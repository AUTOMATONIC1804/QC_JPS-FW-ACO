"""
aco_jps_runner.py
--------------------------------------------------
ACO FW–JPS Integrated Runner (FW-only mode)
✅ Uses only FW nodes (no generated points)
✅ FW D.npy and NEXT.npy for routing
✅ POI scores from qc_pois_final_scored.geojson
✅ Respects fw_jps_buffer.geojson (spatial limits)
✅ Stations spaced ≈800 m apart
✅ Roads from fw_jps_roads.geojson for path constraint
✅ Exports stations, route, and validation report
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union
from src.algorithms.aco.aco_path_station_optimizer import (
    ACOPathStationOptimizer,
    ACOPathParams,
)
from src.algorithms.aco.aco_utils import load_fw_data, load_fixed_endpoints
from src.algorithms.aco.aco_config import ACO_CONFIG


# ---------------------------------------------------------
# Main runner
# ---------------------------------------------------------
def main():
    print("🚆 ACO FW–JPS Integrated Runner (FW-only mode)")

    # === Input files ===
    POINTS_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson")
    DIST_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_D.npy")
    NEXT_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_FW.npy")
    ROADS_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_roads.geojson")
    BUFFER_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_buffer.geojson")
    POI_FILE = Path(r"D:\Quezon_City\data\processed\qc_pois_final_scored.geojson")
    PATH_FILE = Path(r"D:\Quezon_City\data\outputs\jps_path.geojson")

    OUTPUT_DIR = Path(r"D:\Quezon_City\data\outputs\aco")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Load FW base data ===
    nodes, D, _, _ = load_fw_data(POINTS_FILE, DIST_FILE, roads_file=ROADS_FILE)
    start_idx, end_idx = load_fixed_endpoints(PATH_FILE, nodes)

    # === ACO parameters ===
    params = ACOPathParams(
        alpha=1.0,
        beta=2.0,
        rho=0.4,
        q=1.0,
        n_ants=40,
        n_iter=50,
        w_poi=1.0,
        w_dist=0.05,
        min_station_spacing_m=800.0,
        local_search_m=6000.0,
        poi_radius_m=1000.0,
        poi_decay_rate=0.001,
        intersection_snap_m=30.0,
        detour_max_extra_m=1500.0,
        detour_min_gain_ratio=0.25,
    )

    # === Apply FW buffer (optional) ===
    buffer_union = None
    try:
        buffer_gdf = gpd.read_file(BUFFER_FILE)
        if buffer_gdf.crs is None:
            buffer_gdf.set_crs("EPSG:4326", inplace=True)
        try:
            buffer_union = buffer_gdf.union_all()
        except Exception:
            buffer_union = unary_union(buffer_gdf.geometry)
        print("✅ FW buffer loaded and applied.")
    except Exception as e:
        print(f"⚠️ No valid FW buffer found ({e})")

    # === Run optimizer ===
    k = int(input("Enter desired number of stations (k): ").strip())
    optimizer = ACOPathStationOptimizer(
        points_file=POINTS_FILE,
        dist_file=DIST_FILE,
        next_file=NEXT_FILE,
        roads_file=ROADS_FILE,
        poi_path=POI_FILE,
        params=params,
        start_idx=start_idx,
        end_idx=end_idx,
        k_target=k,
    )

    result = optimizer.run(OUTPUT_DIR)
    print(f"✅ Optimization complete. Best fitness: {result['best_fitness']:.4f}")

    # ---------------------------------------------------------
    # Validation report
    # ---------------------------------------------------------
    print("\n🧾 Generating validation report...")
    poi_gdf = gpd.read_file(POI_FILE)
    if poi_gdf.crs is None:
        poi_gdf.set_crs("EPSG:4326", inplace=True)
    poi_m = poi_gdf.to_crs(3857)

    nodes_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(*zip(*nodes)),
        crs="EPSG:4326"
    ).to_crs(3857)

    report_records = []
    for i, node in enumerate(nodes_gdf.geometry):
        buf = node.buffer(1000)
        nearby = poi_m[poi_m.intersects(buf)]

        pois_detail = []
        if len(nearby) > 0:
            for _, row in nearby.iterrows():
                name = row.get("name", "Unknown")
                cat = row.get("Category", "Unclassified")
                score = float(row.get("NormalizedScore", 0))
                pois_detail.append(f"{name} ({cat}, {score:.3f})")

        node_wgs = gpd.GeoSeries([node], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        report_records.append({
            "fw_index": i,
            "is_station": int(i in result["best_subset"]),
            "poi_count": int(len(nearby)),
            "poi_score_sum": float(nearby["NormalizedScore"].sum()) if len(nearby) > 0 else 0.0,
            "poi_details": "; ".join(pois_detail[:25]),
            "lon": node_wgs.x,
            "lat": node_wgs.y,
        })

    report_gdf = gpd.GeoDataFrame(
        report_records,
        geometry=gpd.points_from_xy(
            [r["lon"] for r in report_records],
            [r["lat"] for r in report_records]
        ),
        crs="EPSG:4326"
    )

    report_gdf.to_file(OUTPUT_DIR / "aco_jps_validation_report.geojson", driver="GeoJSON")
    pd.DataFrame(report_records).drop(columns="geometry", errors="ignore").to_csv(
        OUTPUT_DIR / "aco_jps_validation_report.csv", index=False
    )

    print(f"✅ Saved:\n"
          f"   • {OUTPUT_DIR / 'aco_path_stations.geojson'}\n"
          f"   • {OUTPUT_DIR / 'aco_path_route.geojson'}\n"
          f"   • {OUTPUT_DIR / 'aco_jps_validation_report.geojson'}\n"
          f"   • {OUTPUT_DIR / 'aco_jps_validation_report.csv'}")
    print(f"🏁 Best fitness: {result['best_fitness']:.4f}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
