"""
aco_summary.py
---------------------
Aggregates and summarizes ACO results from all FW variants
into a thesis-ready CSV and printed summary.

Requires:
- aco_summary.csv (produced by aco_runner_all.py)
- aco_optimal_stations_<variant>.geojson
- fw_<variant>_D.npy
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

BASE = Path("data/")
PROC = BASE / "processed"
OUT = BASE / "output"

def haversine_m(p1, p2):
    R = 6371008.8
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lon2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(min(1, sqrt(a)))

def mean_station_distance(gdf):
    coords = [(p.x, p.y) for p in gdf.geometry]
    dsum, count = 0, 0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dsum += haversine_m(coords[i], coords[j])
            count += 1
    return dsum / max(count, 1)

def compute_variant_stats(variant):
    stations = gpd.read_file(OUT / f"aco_optimal_stations_{variant}.geojson")
    fw_points = gpd.read_file(PROC / f"fw_{variant}_points.geojson")
    poi_scores = []
    for _, st in stations.iterrows():
        near = fw_points.distance(st.geometry).idxmin()
        poi_scores.append(fw_points.loc[near, "poi_score"])
    mean_poi = np.mean(poi_scores)
    mean_dist = mean_station_distance(stations)
    return {"Variant": variant.upper(), "MeanPOI": round(mean_poi, 4), "MeanInterStationDist_m": round(mean_dist, 2)}

def main():
    summary_path = OUT / "aco_summary.csv"
    df_summary = pd.read_csv(summary_path)
    extra_stats = [compute_variant_stats(v.lower()) for v in ["jps", "astar", "dijkstra"]]
    df_extra = pd.DataFrame(extra_stats)
    df_final = pd.merge(df_summary, df_extra, on="Variant", how="left")
    df_final.to_csv(OUT / "aco_summary_detailed.csv", index=False)
    print("📊 ACO Comparison Summary:")
    print(df_final)
    print(f"💾 Saved → {OUT}/aco_summary_detailed.csv")

if __name__ == "__main__":
    main()
