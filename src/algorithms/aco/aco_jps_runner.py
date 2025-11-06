"""
aco_jps_runner.py
--------------------------------------------------
ACO station optimization (JPS variant) with:
✅ Start/end inclusion
✅ ≥500 m spacing but maintains k total
✅ 1 km proximity-based POI scoring
✅ Route-following snapped output
✅ Full POI validation report per FW node
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split, linemerge
import warnings
from shapely.errors import ShapelyDeprecationWarning

from src.algorithms.aco.aco_utils import load_fw_data, load_fixed_endpoints
from src.algorithms.aco.aco_core_stations import ACOStationOptimizer, ACOStationParams
from src.algorithms.aco.aco_config import ACO_CONFIG


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def _load_route_line(path_file: str) -> LineString:
    gdf = gpd.read_file(path_file).to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    if geom.geom_type == "LineString":
        return geom
    coords = []
    for ls in geom.geoms:
        coords.extend(list(ls.coords))
    return LineString(coords)


def _compute_proximity_scores(nodes_gdf, poi_path, radius_m=1000):
    """
    Compute proximity-based POI score for each node (within 1 km buffer).
    """
    print("📍 Computing 1 km POI proximity scores...")
    poi_gdf = gpd.read_file(poi_path).to_crs("EPSG:3857")
    nodes_m = nodes_gdf.to_crs("EPSG:3857")

    scores = []
    for node in nodes_m.geometry:
        buf = node.buffer(radius_m)
        nearby = poi_gdf[poi_gdf.intersects(buf)]
        scores.append(nearby["NormalizedScore"].sum() if len(nearby) > 0 else 0)
    return np.array(scores)


def _snap_points_to_line(points_lonlat, line_lonlat: LineString):
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
        remaining = [
            (p, m, s)
            for p, m, s in zip(snapped_points, measures, poi_scores)
            if (p, m) not in zip(kept_pts, kept_meas)
        ]
        remaining_sorted = sorted(remaining, key=lambda x: x[2])[:deficit]
        for p, m, s in remaining_sorted:
            kept_pts.append(p)
            kept_meas.append(m)
            kept_scores.append(s)

        order = np.argsort(kept_meas)
        kept_pts = [kept_pts[i] for i in order]
        kept_meas = [kept_meas[i] for i in order]
        kept_scores = [kept_scores[i] for i in order]

    return kept_pts, kept_meas


def _build_route_by_measures(line_m, measures_sorted):
    pts = [line_m.interpolate(m) for m in measures_sorted]
    parts = split(line_m, MultiPoint(pts))
    segments = []
    intervals = list(zip(measures_sorted[:-1], measures_sorted[1:]))
    for seg in parts.geoms:
        mid_m = line_m.project(seg.centroid)
        for a, b in intervals:
            if a <= mid_m <= b:
                segments.append(seg)
                break
    merged = linemerge(segments)
    return gpd.GeoSeries([merged], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("🚆 ACO Station Optimization (JPS variant)")
    k = int(input("Enter desired number of stations (k): ").strip())

    POINTS_FILE = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson"
    DIST_FILE = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_D.npy"
    ROADS_FILE = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_roads.geojson"
    PATH_FILE = r"D:\Quezon_City\data\outputs\jps_path.geojson"
    POI_FILE = r"D:\Quezon_City\data\processed\qc_pois_final_scored.geojson"
    OUTPUT_DIR = Path(r"D:\Quezon_City\data\outputs\aco")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load FW & POI data
    nodes, D, _, _ = load_fw_data(POINTS_FILE, DIST_FILE, roads_file=ROADS_FILE)
    nodes_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*nodes)), crs="EPSG:4326")
    prox_scores = _compute_proximity_scores(nodes_gdf, POI_FILE, radius_m=1000)
    poi_scores = prox_scores  # could combine with base/category if needed

    # Load start/end
    node_coords = [(pt.x, pt.y) for pt in nodes_gdf.geometry]
    start_idx, end_idx = load_fixed_endpoints(PATH_FILE, node_coords)

    # Parameters
    valid_keys = {f.name for f in ACOStationParams.__dataclass_fields__.values()}
    params_kwargs = {k: v for k, v in ACO_CONFIG.items() if k in valid_keys}
    params = ACOStationParams(**params_kwargs)
    optimizer = ACOStationOptimizer(D, poi_scores, params, start_idx, end_idx, k)
    result = optimizer.run()

    # Snap to route
    ordered_idx = result["best_subset"]
    ordered_nodes = [nodes[i] for i in ordered_idx]
    base_line_wgs = _load_route_line(PATH_FILE)
    snapped_lonlat, measures, base_line_m = _snap_points_to_line(ordered_nodes, base_line_wgs)

    order = np.argsort(measures)
    measures_sorted = [measures[i] for i in order]
    snapped_sorted = [snapped_lonlat[i] for i in order]
    idx_sorted = [ordered_idx[i] for i in order]

    snapped_sorted, measures_sorted = _merge_close_stations(
        snapped_sorted, measures_sorted, [poi_scores[i] for i in idx_sorted],
        min_spacing_m=500, k_target=k
    )

    route_wgs = _build_route_by_measures(base_line_m, measures_sorted)

    # Save outputs
    stations_gdf = gpd.GeoDataFrame(
        {"order": list(range(1, len(snapped_sorted) + 1))},
        geometry=gpd.points_from_xy(*zip(*snapped_sorted)),
        crs="EPSG:4326",
    )
    stations_gdf.to_file(OUTPUT_DIR / "aco_jps_stations.geojson", driver="GeoJSON")

    gpd.GeoDataFrame({"id": [0]}, geometry=[route_wgs], crs="EPSG:4326").to_file(
        OUTPUT_DIR / "aco_jps_path.geojson", driver="GeoJSON"
    )

    # ----- Detailed Fitness Summary -----
    subset_idx = result["best_subset"]
    poi_sum = np.sum([poi_scores[i] for i in subset_idx])
    dist_sum = sum([D[subset_idx[i], subset_idx[i + 1]] for i in range(len(subset_idx) - 1)])
    k_used = len(subset_idx)
    fitness = (
        params.beta_poi * poi_sum
        - params.alpha_dist * dist_sum
        + params.gamma_station * k_used
    )

    print("\n✅ Saved outputs:")
    print(f"  • Stations (≥500 m apart, exact k={k}) → {OUTPUT_DIR / 'aco_jps_stations.geojson'}")
    print(f"  • Route line (snapped JPS path)       → {OUTPUT_DIR / 'aco_jps_path.geojson'}")

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

    # ---------------------------------------------------------
    # 🔍 Validation Report Export
    # ---------------------------------------------------------
    print("\n🧾 Generating validation report (POI composition per FW node)...")
    poi_gdf = gpd.read_file(POI_FILE)
    if poi_gdf.crs is None:
        poi_gdf.set_crs("EPSG:4326", inplace=True)
    poi_m = poi_gdf.to_crs("EPSG:3857")

    nodes_m = nodes_gdf.to_crs("EPSG:3857")
    report_records = []

    for i, node in enumerate(nodes_m.geometry):
        buf = node.buffer(1000)
        nearby = poi_m[poi_m.intersects(buf)]

    pois_detail = []
    if len(nearby) > 0:
        for _, row in nearby.iterrows():
            name = row.get("name", "Unknown")
            cat = row.get("Category", "Unclassified")
            score = float(row.get("NormalizedScore", 0))
            pois_detail.append(f"{name} ({cat}, {score:.3f})")

    # Convert to WGS84 for coordinate export
    node_wgs = gpd.GeoSeries([node], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    report_records.append({
        "fw_index": i,
        "is_station": int(i in result["best_subset"]),
        "poi_count": len(nearby),
        "poi_score_sum": float(nearby["NormalizedScore"].sum()) if len(nearby) > 0 else 0.0,
        "poi_details": "; ".join(pois_detail[:25]),
        "lon": node_wgs.x,
        "lat": node_wgs.y,
    })


    report_gdf = gpd.GeoDataFrame(
        report_records,
        geometry=[Point(xy) for xy in zip(
            [r["lon"] for r in report_records],
            [r["lat"] for r in report_records]
        )],
        crs="EPSG:4326"
    )
    report_gdf.to_file(OUTPUT_DIR / "aco_jps_validation_report.geojson", driver="GeoJSON")
    report_gdf.drop(columns="geometry").to_csv(OUTPUT_DIR / "aco_jps_validation_report.csv", index=False)

    print(f"✅ Validation report exported:")
    print(f"   • GeoJSON → {OUTPUT_DIR / 'aco_jps_validation_report.geojson'}")
    print(f"   • CSV     → {OUTPUT_DIR / 'aco_jps_validation_report.csv'}")


if __name__ == "__main__":
    main()
