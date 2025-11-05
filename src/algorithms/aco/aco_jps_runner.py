"""
aco_jps_runner.py
--------------------------------------------------
Runner for ACO-based station optimization using
the JPS-generated Floyd–Warshall outputs.

Uses:
- fw_jps_points.geojson (nodes)
- fw_jps_D.npy          (distance matrix)
- fw_jps_roads.geojson  (road network)
- jps_path.geojson      (original JPS route)
"""

import geopandas as gpd
import numpy as np
from pathlib import Path
from src.algorithms.aco.aco_utils import load_fw_data, load_fixed_endpoints
from src.algorithms.aco.aco_core_stations import ACOStationOptimizer, ACOStationParams
from src.algorithms.aco.aco_config import ACO_CONFIG


# ---------------------------------------------------
# Main runner
# ---------------------------------------------------
def main():
    print("🚆 ACO Station Optimization (JPS variant)")

    # --- User input ---
    k = int(input("Enter desired number of stations (k): "))

    # --- File paths ---
    POINTS_FILE = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson"
    DIST_FILE   = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_D.npy"
    ROADS_FILE  = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_roads.geojson"
    PATH_FILE   = r"D:\Quezon_City\data\outputs\jps_path.geojson"
    OUTPUT_DIR  = Path(r"D:\Quezon_City\data\outputs\aco")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load FW dataset ---
    nodes, D, poi_scores, roads_gdf = load_fw_data(POINTS_FILE, DIST_FILE, roads_file=ROADS_FILE)

    # --- Load route start/end ---
    start_idx, end_idx = load_fixed_endpoints(PATH_FILE, nodes)

    # --- Initialize ACO parameters ---
    params = ACOStationParams(
        alpha=ACO_CONFIG["alpha"],
        beta=ACO_CONFIG["beta"],
        rho=ACO_CONFIG["rho"],
        Q=ACO_CONFIG["Q"],
        iterations=ACO_CONFIG["iterations"],
        num_ants=ACO_CONFIG["num_ants"],
        alpha_dist=ACO_CONFIG["alpha_dist"],
        beta_poi=ACO_CONFIG["beta_poi"],
        gamma_station=ACO_CONFIG["gamma_station"],
        seed=ACO_CONFIG["seed"],
    )

    # --- Run optimizer ---
    optimizer = ACOStationOptimizer(
        D=D,
        poi_scores=poi_scores,
        params=params,
        start_idx=start_idx,
        end_idx=end_idx,
        k_target=k,
    )

    result = optimizer.run()

    # --- Extract best subset ---
    best_indices = result["best_subset"]
    best_nodes = [nodes[i] for i in best_indices]

    print("\n✅ Final Results")
    print(f"🏁 Best fitness: {result['best_fitness']:.4f}")
    print(f"📍 Selected stations (node indices): {best_indices}")

    # --- Create GeoDataFrame for selected stations ---
    stations_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [p[0] for p in best_nodes],
            [p[1] for p in best_nodes]
        ),
        crs="EPSG:4326"
    )

    stations_gdf["node_index"] = best_indices
    stations_gdf["fitness"] = result["best_fitness"]

    # --- Save results ---
    out_file = OUTPUT_DIR / "aco_jps_stations.geojson"
    stations_gdf.to_file(out_file, driver="GeoJSON")

    print(f"💾 Saved ACO-selected stations to: {out_file}")
    print("===========================================")


# ---------------------------------------------------
# Entry point
# ---------------------------------------------------
if __name__ == "__main__":
    main()
