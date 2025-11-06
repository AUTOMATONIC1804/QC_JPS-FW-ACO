"""
aco_jps_runner.py
--------------------------------------------------
ACO station optimization (JPS variant)
✅ Uses only existing FW nodes (no new/snapped points)
✅ Start/end inclusion
✅ ≥500 m spacing (keeps exact k)
✅ 1 km proximity-based POI scoring
✅ Validation report for ALL FW nodes
✅ 1 km buffers per chosen station with POI summary
✅ Deduplicates POIs using '@id' (or fallback keys)
✅ Counts POIs by GEOMETRY INTERSECTION with buffer (touch/overlap counts) — robust to edge cases
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.validation import make_valid

from src.algorithms.aco.aco_utils import load_fw_data, load_fixed_endpoints
from src.algorithms.aco.aco_core_stations import ACOStationOptimizer, ACOStationParams
from src.algorithms.aco.aco_config import ACO_CONFIG


# ---------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------

_EPS_M = 0.05  # 5 cm numeric tolerance in meters for boundary-touch cases

def _make_valid_series(geom):
    """Return a valid geometry (works for Shapely 1.x and 2.x)."""
    try:
        # Shapely 2.x
        fixed = make_valid(geom)
    except Exception:
        # Fallback (older stacks): classic fix
        try:
            fixed = geom.buffer(0)
        except Exception:
            fixed = geom
    return fixed

def _clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop empties, fix invalid, keep CRS."""
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    # Only fix invalids to save time
    mask_invalid = ~gdf.is_valid
    if mask_invalid.any():
        gdf.loc[mask_invalid, "geometry"] = gdf.loc[mask_invalid, "geometry"].apply(_make_valid_series)
    # Drop again if any became empty after fix
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    return gdf

def _query_intersects_or_touches(poi_m: gpd.GeoDataFrame, buf_geom) -> gpd.GeoDataFrame:
    """
    Robust 'intersects or touches' using:
      1) spatial index bbox candidates
      2) precise intersects
      3) fallback: distance <= _EPS_M (catches boundary touches missed by numeric quirks)
    """
    # Use sindex for speed
    if getattr(poi_m, "sindex", None) is None:
        return poi_m.iloc[[]]

    # bbox prefilter
    candidates_idx = list(poi_m.sindex.intersection(buf_geom.bounds))
    if not candidates_idx:
        return poi_m.iloc[[]]

    cand = poi_m.iloc[candidates_idx].copy()

    # Exact intersects
    intersects_mask = cand.intersects(buf_geom)

    # Fallback boundary touch (distance ~ 0)
    # Only compute for the non-intersecting subset to save time
    if (~intersects_mask).any():
        cand_non = cand.loc[~intersects_mask]
        # distance to polygon buffer is 0 when touching boundary
        dist = cand_non.distance(buf_geom)
        touch_mask = dist <= _EPS_M
        # merge masks
        intersects_mask.loc[touch_mask.index] = intersects_mask.loc[touch_mask.index] | touch_mask

    return cand.loc[intersects_mask]


# ---------------------------------------------------------
# Helper: Deduplicate POIs but KEEP original geometry
# ---------------------------------------------------------

def _prep_poi_dedup_keep_geom(poi_gdf_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Deduplicate POIs while preserving original geometry (points/polygons/multipolygons).
    Priority for dedup key:
      1) '@id'  2) 'id'  3) (name || Category)
    """
    gdf = poi_gdf_m.copy()

    if "@id" in gdf.columns:
        gdf["_dedup_key"] = gdf["@id"].astype(str)
    elif "id" in gdf.columns:
        gdf["_dedup_key"] = gdf["id"].astype(str)
    else:
        gdf["_dedup_key"] = (
            gdf.get("name", "").astype(str).str.lower().fillna("")
            + "||"
            + gdf.get("Category", "").astype(str).str.lower().fillna("")
        )

    gdf = gdf.drop_duplicates(subset=["_dedup_key"]).reset_index(drop=True)
    return gdf


# ---------------------------------------------------------
# Helper: Compute POI proximity scores for FW nodes (ROBUST INTERSECTS/TOUCHES)
# ---------------------------------------------------------

def _compute_proximity_scores(nodes_xy, poi_path, radius_m=1000.0) -> np.ndarray:
    """
    Sum of NormalizedScore of all POIs whose geometry intersects OR touches
    the buffer disk (radius_m) around each FW node (EPSG:3857), with robust fixes.
    """
    print("📍 Computing 1 km POI proximity scores (mode=intersects|touches, robust)...")
    poi_gdf = gpd.read_file(poi_path)
    if poi_gdf.crs is None:
        poi_gdf.set_crs("EPSG:4326", inplace=True)

    # Project → clean → dedup
    poi_m = poi_gdf.to_crs("EPSG:3857")
    poi_m = _clean_geoms(poi_m)
    poi_m = _prep_poi_dedup_keep_geom(poi_m)

    nodes_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(*zip(*nodes_xy)), crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    scores = np.zeros(len(nodes_gdf), dtype=float)

    for i, pt in enumerate(nodes_gdf.geometry):
        # Clean the buffer too, and add tiny epsilon growth to catch numeric misses
        buf = pt.buffer(radius_m + _EPS_M)
        buf = _make_valid_series(buf)

        nearby = _query_intersects_or_touches(poi_m, buf)

        if len(nearby) > 0 and "NormalizedScore" in nearby.columns:
            scores[i] = float(nearby["NormalizedScore"].sum())

    # normalize for stability
    maxv = scores.max() if scores.size else 0.0
    if maxv > 0:
        scores = scores / maxv
    return scores


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("🚆 ACO Station Optimization (JPS variant)")
    k = int(input("Enter desired number of stations (k): ").strip())

    # ---- Paths
    POINTS_FILE = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson"
    DIST_FILE   = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_D.npy"
    ROADS_FILE  = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_roads.geojson"
    PATH_FILE   = r"D:\Quezon_City\data\outputs\jps_path.geojson"
    POI_FILE    = r"D:\Quezon_City\data\processed\qc_pois_final_scored.geojson"
    OUTPUT_DIR  = Path(r"D:\Quezon_City\data\outputs\aco")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load FW data (nodes as list[(lon,lat)], D as np.ndarray)
    nodes_xy, D, _, _ = load_fw_data(POINTS_FILE, DIST_FILE, roads_file=ROADS_FILE)

    # ---- Compute POI proximity score per node (ROBUST INTERSECTS/TOUCHES)
    poi_scores = _compute_proximity_scores(nodes_xy, POI_FILE, radius_m=1000.0)

    # ---- Start/End indices
    start_idx, end_idx = load_fixed_endpoints(PATH_FILE, nodes_xy)

    # ---- ACO parameters
    valid_keys = {f.name for f in ACOStationParams.__dataclass_fields__.values()}
    params_kwargs = {k: v for k, v in ACO_CONFIG.items() if k in valid_keys}
    params = ACOStationParams(**params_kwargs)

    # ---- Run ACO
    optimizer = ACOStationOptimizer(
        D=D,
        poi_scores=poi_scores,
        params=params,
        start_idx=start_idx,
        end_idx=end_idx,
        k_target=k,
        min_spacing_m=500.0,
    )
    result = optimizer.run()
    chosen_idx = result["best_subset"]
    best_fit = result["best_fitness"]

    # =========================================================
    # Outputs
    # =========================================================

    # ---- Stations (exact FW nodes)
    stations_lon = [nodes_xy[i][0] for i in chosen_idx]
    stations_lat = [nodes_xy[i][1] for i in chosen_idx]
    stations_gdf = gpd.GeoDataFrame(
        {
            "order": list(range(1, len(chosen_idx) + 1)),
            "fw_index": chosen_idx,
            "poi_norm": [float(poi_scores[i]) for i in chosen_idx],
        },
        geometry=gpd.points_from_xy(stations_lon, stations_lat),
        crs="EPSG:4326",
    )
    stations_path = OUTPUT_DIR / "aco_jps_stations.geojson"
    stations_gdf.to_file(stations_path, driver="GeoJSON")

    # ---- Prepare POIs in metric CRS, clean + dedup, for buffers/validation
    poi_src = gpd.read_file(POI_FILE)
    if poi_src.crs is None:
        poi_src.set_crs("EPSG:4326", inplace=True)
    poi_m = poi_src.to_crs("EPSG:3857")
    poi_m = _clean_geoms(poi_m)
    poi_m = _prep_poi_dedup_keep_geom(poi_m)

    # ---- Buffers for visual validation (ROBUST INTERSECTS/TOUCHES)
    print("🟢 Generating 1 km buffers for station POI validation (mode=intersects|touches, robust)...")
    stations_m = stations_gdf.to_crs("EPSG:3857")
    buffers, poi_counts, poi_sums, poi_cats = [], [], [], []
    for geom in stations_m.geometry:
        buf = _make_valid_series(geom.buffer(1000.0 + _EPS_M))
        near = _query_intersects_or_touches(poi_m, buf)
        buffers.append(buf)
        poi_counts.append(int(len(near)))
        poi_sums.append(float(near["NormalizedScore"].sum()) if len(near) > 0 else 0.0)
        cats = near["Category"].value_counts().head(6).to_dict() if len(near) > 0 else {}
        poi_cats.append(str(cats))

    buffers_gdf = gpd.GeoDataFrame(
        {
            "fw_index": chosen_idx,
            "poi_count": poi_counts,
            "poi_score_sum": poi_sums,
            "top_categories": poi_cats,
            "radius_m": 1000,
            "count_mode": "intersects|touches",
            "dedup_key": "@id_then_id_then_name||Category",
            "tolerance_m": _EPS_M,
        },
        geometry=buffers,
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    buffers_path = OUTPUT_DIR / "aco_jps_station_buffers.geojson"
    buffers_gdf.to_file(buffers_path, driver="GeoJSON")

    # ---- Validation report for ALL FW nodes (ROBUST INTERSECTS/TOUCHES)
    print("🧾 Generating validation report (all FW nodes, mode=intersects|touches, robust)...")
    nodes_m = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(*zip(*nodes_xy)), crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    report_records = []
    for i, node in enumerate(nodes_m.geometry):
        buf = _make_valid_series(node.buffer(1000.0 + _EPS_M))
        near = _query_intersects_or_touches(poi_m, buf)
        node_wgs = gpd.GeoSeries([node], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        report_records.append({
            "fw_index": i,
            "is_station": int(i in chosen_idx),
            "poi_count": int(len(near)),
            "poi_score_sum": float(near["NormalizedScore"].sum()) if len(near) > 0 else 0.0,
            "lon": float(node_wgs.x),
            "lat": float(node_wgs.y),
        })

    report_gdf = gpd.GeoDataFrame(
        report_records,
        geometry=gpd.points_from_xy(
            [r["lon"] for r in report_records], [r["lat"] for r in report_records]
        ),
        crs="EPSG:4326",
    )
    report_geojson = OUTPUT_DIR / "aco_jps_validation_report.geojson"
    report_csv = OUTPUT_DIR / "aco_jps_validation_report.csv"
    report_gdf.to_file(report_geojson, driver="GeoJSON")
    report_gdf.drop(columns="geometry").to_csv(report_csv, index=False)

    # ---- Summary
    print("\n✅ Outputs generated successfully:")
    print(f"📍 Stations (points) → {stations_path}")
    print(f"🗺️ Buffers (1 km, robust intersects|touches) → {buffers_path}")
    print(f"🧾 Full validation CSV → {report_csv}")
    print(f"🏁 Best fitness: {best_fit:.4f}")


if __name__ == "__main__":
    main()
