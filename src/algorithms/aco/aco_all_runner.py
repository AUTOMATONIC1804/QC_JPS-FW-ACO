"""
aco_runner_all.py
-----------------
Runs ACO station selection for all algorithm variants (JPS, A*, Dijkstra).
Requires:
- fw_<variant>_points.geojson
- fw_<variant>_D.npy
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from src.algorithms.aco.aco_utils import load_fw_data
from src.algorithms.aco.aco_core_stations import ACOStationOptimizer, ACOStationParams
from src.algorithms.aco.aco_config import ACO_CONFIG

VARIANTS = {
    "jps": ("fw_jps_points.geojson", "fw_jps_D.npy"),
    "astar": ("fw_astar_points.geojson", "fw_astar_D.npy"),
    "dijkstra": ("fw_dijkstra_points.geojson", "fw_dijkstra_D.npy"),
}


def run_variant(name: str, k: int):
    print(f"\n🚆 Running ACO Station Optimization for {name.upper()} variant...")

    data_dir = Path(ACO_CONFIG["data_dir"])
    output_dir = Path(ACO_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    points_file, dist_file = VARIANTS[name]
    nodes, D, poi_scores = load_fw_data(data_dir / points_file, data_dir / dist_file)

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

    optimizer = ACOStationOptimizer(nodes, D, poi_scores, k_target=k, params=params)
    result = optimizer.run()

    # Save selected stations as GeoJSON
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([p[0] for p in result["stations"]], [p[1] for p in result["stations"]]),
        crs="EPSG:4326"
    )
    out_file = output_dir / f"aco_optimal_stations_{name}.geojson"
    gdf.to_file(out_file, driver="GeoJSON")
    print(f"✅ Saved {len(gdf)} stations → {out_file}")

    return {"Variant": name.upper(), "Fitness": round(result["fitness"], 4), "Stations": len(gdf)}


def main():
    print("=== ACO Unified Runner ===")
    k = int(input("Enter desired number of stations (k): ").strip())
    results = []

    for variant in VARIANTS.keys():
        try:
            stats = run_variant(variant, k)
            results.append(stats)
        except Exception as e:
            print(f"⚠️ {variant.upper()} failed: {e}")

    if results:
        df = pd.DataFrame(results)
        out_summary = Path(ACO_CONFIG["output_dir"]) / "aco_summary.csv"
        df.to_csv(out_summary, index=False)
        print("\n📊 Summary:")
        print(df)
        print(f"💾 Saved summary → {out_summary}")


if __name__ == "__main__":
    main()
