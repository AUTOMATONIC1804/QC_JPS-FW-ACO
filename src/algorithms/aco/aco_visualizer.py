"""
aco_visualizer.py
-----------------
Plots ACO-selected stations and POIs without contextily.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed/")
OUTPUT_DIR = Path("data/output/")
VARIANTS = ["jps", "astar", "dijkstra"]

def plot_variant(variant):
    pois = gpd.read_file(DATA_DIR / "qc_pois_final_scored.geojson")
    stations = gpd.read_file(OUTPUT_DIR / f"aco_optimal_stations_{variant}.geojson")

    fig, ax = plt.subplots(figsize=(8, 8))
    pois.plot(ax=ax, markersize=5, color="orange", alpha=0.4, label="POIs")
    stations.plot(ax=ax, color="red", markersize=80, marker="o", label="ACO Stations")

    ax.set_title(f"ACO Station Selection – {variant.upper()}")
    ax.legend()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"aco_{variant}_map.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🗺️ Saved → {out_path}")


def main():
    for v in VARIANTS:
        try:
            plot_variant(v)
        except Exception as e:
            print(f"⚠️ Skipped {v.upper()}: {e}")


if __name__ == "__main__":
    main()
