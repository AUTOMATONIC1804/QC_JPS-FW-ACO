"""
aco_runner_astar.py
-------------------
ACO station optimization for the A* Search variant.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import LineString

from src.algorithms.aco.aco_utils import load_fw_data, load_fixed_endpoints
from src.algorithms.aco.aco_core_stations import ACOStationOptimizer, ACOStationParams
from src.algorithms.aco.aco_config import ACO_CONFIG

# === File Paths (A* Variant) ===
POINTS_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_astar_points.geojson")
ROADS_FILE  = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_astar_roads.geojson")
DIST_FILE   = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_astar_D.npy")
PATH_FILE   = Path(r"D:\Quezon_City\data\outputs\astar_path.geojson")
OUTPUT_DIR  = Path(r"D:\Quezon_City\data\outputs\aco")

def main():
    print("🛰️ ACO Station Optimization (A* variant)")
    k = int(input("Enter desired number of stations (k): ").strip())

    nodes, D, poi_scores = load_fw_data(POINTS_FILE, DIST_FILE, roads_file=ROADS_FILE)
    start_idx, end_idx = load_fixed_endpoints(PATH_FILE, nodes)
    print(f"📍 Fixed endpoints: start #{start_idx}, end #{end_idx}")

    params = ACOStationParams(**ACO_CONFIG)
    optimizer = ACOStationOptimizer(
        nodes, D, poi_scores, k_target=k, params=params,
        start_idx=start_idx, end_idx=end_idx
    )

    result = optimizer.run()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_points = OUTPUT_DIR / "aco_optimal_stations_astar.geojson"
    out_route  = OUTPUT_DIR / "aco_optimal_route_astar.geojson"
    out_csv    = OUTPUT_DIR / "aco_summary_astar.csv"

    gdf_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([p[0] for p in result["stations"]], [p[1] for p in result["stations"]]),
        crs="EPSG:4326"
    )
    gdf_points["order"] = list(range(1, len(gdf_points)+1))
    gdf_points.to_file(out_points, driver="GeoJSON")

    route_line = LineString([(p[0], p[1]) for p in result["stations"]])
    gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326").to_file(out_route, driver="GeoJSON")

    pd.DataFrame([{
        "Variant": "A*",
        "Fitness": round(result["fitness"], 4),
        "Stations": len(gdf_points),
        "StartIndex": start_idx,
        "EndIndex": end_idx,
    }]).to_csv(out_csv, index=False)

    print(f"✅ Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
