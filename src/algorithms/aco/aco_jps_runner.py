"""
aco_jps_runner.py
--------------------------------------------------
ACO station optimization (JPS variant) with:
✅ Start/end inclusion
✅ ≥500 m spacing but maintains k total
✅ 1 km proximity-based POI scoring (from POI file)
✅ Route-following snapped output
✅ Uses ONLY FW nodes (no new/generated points)
"""

from pathlib import Path
import warnings
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split, linemerge
from shapely.errors import ShapelyDeprecationWarning

from src.algorithms.aco.aco_utils import load_fw_data, load_fixed_endpoints
from src.algorithms.aco.aco_core_stations import ACOStationOptimizer, ACOStationParams
from src.algorithms.aco.aco_config import ACO_CONFIG


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def _load_route_line(path_file: str) -> LineString:
    gdf = gpd.read_file(path_file).to_crs("EPSG:4326")
    # union_all is preferred; fallback to unary_union if needed
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    if geom.geom_type == "LineString":
        return geom
    coords = []
    for ls in geom.geoms:
        coords.extend(list(ls.coords))
    return LineString(coords)


def _compute_proximity_scores(nodes_gdf: gpd.GeoDataFrame, poi_path: str, radius_m: float = 1000.0) -> np.ndarray:
    """
    Compute proximity-based POI score for each node (within 1 km buffer).
    nodes_gdf: GeoDataFrame (EPSG:4326) with Point geometry.
    """
    print("📍 Computing 1 km POI proximity scores...")
    poi_gdf = gpd.read_file(poi_path)
    if poi_gdf.crs is None:
        poi_gdf.set_crs("EPSG:4326", inplace=True)
    poi_m = poi_gdf.to_crs("EPSG:3857")

    nodes = nodes_gdf.copy()
    if nodes.crs is None:
        nodes.set_crs("EPSG:4326", inplace=True)
    nodes_m = nodes.to_crs("EPSG:3857")

    norms = np.zeros(len(nodes_m), dtype=float)
    if len(poi_m) == 0:
        return norms

    # ensure score column exists
    score_col = "NormalizedScore" if "NormalizedScore" in poi_m.columns else None
    if score_col is None:
        return norms

    for i, pt in enumerate(nodes_m.geometry):
        buf = pt.buffer(radius_m)
        nearby = poi_m[poi_m.intersects(buf)]
        norms[i] = nearby[score_col].sum() if len(nearby) > 0 else 0.0

    # normalize proximity scores so they’re 0..1 (avoid huge sums)
    maxv = norms.max() if norms.size else 0.0
    if maxv > 0:
        norms = norms / maxv
    return norms


def _snap_points_to_line(points_lonlat, line_lonlat: LineString):
    """Snap given lon/lat point tuples to the route line; return snapped tuples + measures."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    line_m = gpd.GeoSeries([line_lonlat], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    snapped, measures = [], []

    for (x, y) in points_lonlat:
        try:
            pt_m = gpd.GeoSeries.from_xy([x], [y], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            if not pt_m.is_valid or pt_m.is_empty:
                snapped.append((x, y))
                measures.append(0.0)
                continue
            m = line_m.project(pt_m)
            q = line_m.interpolate(m)
            q_wgs = gpd.GeoSeries([q], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
            snapped.append((q_wgs.x, q_wgs.y))
            measures.append(float(m))
        except Exception:
            snapped.append((x, y))
            measures.append(0.0)

    return snapped, measures, line_m


def _merge_close_stations(snapped_points, measures, poi_scores, min_spacing_m=100, k_target=None):
    """Greedy keep stations ≥ min_spacing_m apart along measure; then top-up to k by POI."""
    if not snapped_points or not measures:
        return snapped_points, measures

    kept_pts, kept_meas, kept_scores = [snapped_points[0]], [measures[0]], [poi_scores[0]]

    for i in range(1, len(snapped_points)):
        if (measures[i] - kept_meas[-1]) >= min_spacing_m:
            kept_pts.append(snapped_points[i])
            kept_meas.append(measures[i])
            kept_scores.append(poi_scores[i])

    if k_target and len(kept_pts) < k_target:
        deficit = k_target - len(kept_pts)
        # remaining by POI (exclude already kept by measure)
        kept_set = set(zip(kept_pts, kept_meas))
        remaining = [
            (p, m, s)
            for p, m, s in zip(snapped_points, measures, poi_scores)
            if (p, m) not in kept_set
        ]
        # pick highest POI to fill
        remaining_sorted = sorted(remaining, key=lambda x: x[2], reverse=True)[:deficit]
        for p, m, s in remaining_sorted:
            kept_pts.append(p)
            kept_meas.append(m)
            kept_scores.append(s)

        # sort again by measure
        order = np.argsort(kept_meas)
        kept_pts = [kept_pts[i] for i in order]
        kept_meas = [kept_meas[i] for i in order]

    return kept_pts, kept_meas


def _build_route_by_measures(line_m, measures_sorted):
    """Create path polyline by splitting original route at station measures and merging segments."""
    if not measures_sorted:
        return gpd.GeoSeries([], crs="EPSG:3857")

    pts = [line_m.interpolate(m) for m in measures_sorted]
    if len(pts) < 2:
        return gpd.GeoSeries([line_m], crs="EPSG:3857")

    parts = split(line_m, MultiPoint(pts))
    segments = []
    intervals = list(zip(measures_sorted[:-1], measures_sorted[1:]))

    for seg in parts.geoms:
        mid_m = line_m.project(seg.centroid)
        for a, b in intervals:
            if a <= mid_m <= b:
                segments.append(seg)
                break

    if not segments:
        segments = [line_m]

    merged = linemerge(segments)
    return gpd.GeoSeries([merged], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("🚆 ACO Station Optimization (JPS variant)")
    k = int(input("Enter desired number of stations (k): ").strip())

    POINTS_FILE = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson"
    DIST_FILE   = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_D.npy"
    ROADS_FILE  = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_roads.geojson"
    PATH_FILE   = r"D:\Quezon_City\data\outputs\jps_path.geojson"
    POI_FILE    = r"D:\Quezon_City\data\processed\qc_pois_final_scored.geojson"
    OUTPUT_DIR  = Path(r"D:\Quezon_City\data\outputs\aco")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load FW & POI data (nodes as GeoDataFrame (EPSG:4326), D as np.ndarray)
    nodes_raw, D, _, _ = load_fw_data(POINTS_FILE, DIST_FILE, roads_file=ROADS_FILE)

    # ✅ Convert to GeoDataFrame if it’s just a list of (lon, lat)
    if isinstance(nodes_raw, list):
        nodes_gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in nodes_raw], crs="EPSG:4326")
    else:
        nodes_gdf = nodes_raw

    # Proximity scores (0..1) around each FW node
    prox_scores = _compute_proximity_scores(nodes_gdf, POI_FILE, radius_m=1000.0)
    poi_scores = prox_scores  # (extend later with base/category if desired)

    # Convert GeoDataFrame geometry into (lon, lat) tuples for compatibility
    node_coords = [(pt.x, pt.y) for pt in nodes_gdf.geometry]
    start_idx, end_idx = load_fixed_endpoints(PATH_FILE, node_coords)


    # ACO params (only algorithmic keys from config)
    valid_keys = {f.name for f in ACOStationParams.__dataclass_fields__.values()}
    params_kwargs = {k: v for k, v in ACO_CONFIG.items() if k in valid_keys}
    params = ACOStationParams(**params_kwargs)

    # Run ACO (uses ONLY FW nodes — no generated points)
    optimizer = ACOStationOptimizer(
        D=D,
        poi_scores=poi_scores,
        params=params,
        start_idx=start_idx,
        end_idx=end_idx,
        k_target=k,
        min_spacing_m=500.0
    )
    result = optimizer.run()

    # Extract chosen FW nodes (keep as existing nodes, not new coords)
    ordered_idx = result["best_subset"]
    chosen_nodes = nodes_gdf.iloc[ordered_idx].reset_index(drop=True)

    # Snap chosen nodes to the route for visualization AND build path between them along the route
    base_line_wgs = _load_route_line(PATH_FILE)
    chosen_lonlat = [(pt.x, pt.y) for pt in chosen_nodes.geometry]
    snapped_lonlat, measures, base_line_m = _snap_points_to_line(chosen_lonlat, base_line_wgs)

    # Ensure ≥500 m spacing while keeping exact k
    order = np.argsort(measures)
    measures_sorted = [measures[i] for i in order]
    snapped_sorted  = [snapped_lonlat[i] for i in order]
    idx_sorted      = [ordered_idx[i] for i in order]
    poi_for_sorted  = [poi_scores[i] for i in idx_sorted]

    snapped_sorted, measures_sorted = _merge_close_stations(
        snapped_sorted, measures_sorted, poi_for_sorted, min_spacing_m=500.0, k_target=k
    )

    route_wgs = _build_route_by_measures(base_line_m, measures_sorted)

    # Save stations (snapped) and route
    stations_gdf = gpd.GeoDataFrame(
        {
            "order": list(range(1, len(snapped_sorted) + 1)),
            "fw_index": [idx_sorted[i] for i in range(len(snapped_sorted))],
            "poi_norm": [round(poi_for_sorted[i], 6) for i in range(len(snapped_sorted))],
        },
        geometry=gpd.points_from_xy(*zip(*snapped_sorted)),
        crs="EPSG:4326",
    )
    stations_gdf.to_file(OUTPUT_DIR / "aco_jps_stations.geojson", driver="GeoJSON")

    gpd.GeoDataFrame({"id": [0]}, geometry=[route_wgs], crs="EPSG:4326").to_file(
        OUTPUT_DIR / "aco_jps_path.geojson", driver="GeoJSON"
    )

    print("\n✅ Saved outputs:")
    print(f"  • Stations (≥500 m apart, exact k={k}) → {OUTPUT_DIR / 'aco_jps_stations.geojson'}")
    print(f"  • Route line (snapped JPS path)       → {OUTPUT_DIR / 'aco_jps_path.geojson'}")

    # ----- Compute breakdown -----
    subset_idx = result["best_subset"]
    poi_sum = np.sum([poi_scores[i] for i in subset_idx])
    dist_sum = sum(
    [D[subset_idx[i], subset_idx[i + 1]] for i in range(len(subset_idx) - 1)]
    )
    k_used = len(subset_idx)
    fitness = (
        params.beta_poi * poi_sum
        - params.alpha_dist * dist_sum
        + params.gamma_station * k_used
    )

    print("\n📊 Detailed Final Metrics:")
    print(f"   Total POI benefit:   {poi_sum:.4f}")
    print(f"   Total path distance: {dist_sum:.2f} m")
    print(f"   Station count:       {k_used} / target {k}")
    print(f"   ────────────────────────────────────────")
    print(f"   β_poi * POI  = {params.beta_poi * poi_sum:.4f}")
    print(f"   -α_dist * D  = {-params.alpha_dist * dist_sum:.4f}")
    print(f"   +γ_station*k = {params.gamma_station * k_used:.4f}")
    print(f"   ----------------------------------------")
    print(f"🏁 Final computed fitness: {fitness:.4f}")
    print(f"🧠 Recorded best fitness:  {result['best_fitness']:.4f}")



if __name__ == "__main__":
    main()
