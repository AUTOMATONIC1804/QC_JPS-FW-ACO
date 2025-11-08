# src/algorithms/aco/aco_jps_runner.py
"""
ACO + JPS Integrated Runner
----------------------------
1. Loads candidate points (from JPS–FW pipeline)
2. Merges start/end nodes from JPS path (removing nearby duplicates)
3. Computes effort matrix based on POIs within 1 km buffers
4. Runs Ant Colony Optimization (station selection)
5. Locks start/end points
6. Enforces path order along the JPS route
7. Exports full node comparison CSV (chosen + unchosen)
8. Exports 1 km buffers for chosen stations (for visual verification)
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
from shapely.geometry import LineString
from src.algorithms.aco.aco_effort_matrix import compute_effort_matrix, EffortParams
from src.algorithms.aco.aco_station_selector import select_optimal_stations
from src.algorithms.aco.poi_scores import load_pois_and_weights

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
# ======================================================
# Helper functions
# ======================================================

def merge_jps_points_with_path(points_gdf, route_gdf, proximity_threshold_m=100):
    """Merge FW candidate nodes with JPS path start/end nodes, removing near duplicates."""
    start_geom = route_gdf.loc[route_gdf["role"] == "start", "geometry"].iloc[0]
    end_geom = route_gdf.loc[route_gdf["role"] == "goal", "geometry"].iloc[0]

    points_m = points_gdf.to_crs("EPSG:3857")
    start_m = gpd.GeoSeries([start_geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    end_m = gpd.GeoSeries([end_geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    start_dists = points_m.distance(start_m)
    end_dists = points_m.distance(end_m)
    mask_near = (start_dists < proximity_threshold_m) | (end_dists < proximity_threshold_m)
    cleaned = points_gdf.loc[~mask_near].copy()

    merged = gpd.GeoDataFrame(pd.concat([
        gpd.GeoDataFrame({"fw_index": [-1], "role": ["start"], "geometry": [start_geom]}, crs="EPSG:4326"),
        cleaned.to_crs("EPSG:4326"),
        gpd.GeoDataFrame({"fw_index": [-2], "role": ["goal"], "geometry": [end_geom]}, crs="EPSG:4326")
    ], ignore_index=True), crs="EPSG:4326")

    return merged


def _route_line_from_geojson(route_gdf):
    """Extract the main LineString path from jps_path.geojson."""
    if "role" in route_gdf.columns:
        line_rows = route_gdf[route_gdf["role"] == "path"]
        if not line_rows.empty:
            return line_rows.geometry.iloc[0]
    lines = [g for g in route_gdf.geometry if isinstance(g, LineString)]
    return lines[0] if lines else None


def _sort_indices_along_line(points_gdf, line, indices):
    """Sort node indices along the direction of the JPS route line."""
    pts = points_gdf.to_crs("EPSG:3857")
    line_m = LineString(line) if not isinstance(line, LineString) else line
    dists = [(i, line_m.project(pts.geometry.iloc[i])) for i in indices]
    dists.sort(key=lambda x: x[1])
    return [i for i, _ in dists]


def _enforce_exact_k(sorted_indices, k, start_idx, end_idx):
    """Trim or subsample to ensure exactly k nodes (with start & end)."""
    keep = [i for i in sorted_indices if i not in (start_idx, end_idx)]
    result = [start_idx]
    internal_needed = max(0, k - 2)
    if len(keep) > internal_needed:
        sel_pos = np.linspace(0, len(keep) - 1, num=internal_needed, dtype=int)
        chosen = [keep[p] for p in sel_pos]
    else:
        chosen = keep
    result += chosen + [end_idx]
    final = []
    for i in result:
        if i not in final:
            final.append(i)
    return final[:k]


# ======================================================
# MAIN PIPELINE
# ======================================================

def run_aco_jps(
    n_stations: int = 9,
    method: str = "jps",
    buffer_radius_m: float = 1000.0,
    output_dir: str = "data/outputs/aco",
):
    print("=== 🧠 Running ACO + JPS Integrated Optimization ===")

    base_fw_dir = Path("data/outputs/floyd_warshall")
    points_path = base_fw_dir / f"fw_{method}_points.geojson"
    D_path = base_fw_dir / f"fw_{method}_D.npy"
    FW_path = base_fw_dir / f"fw_{method}_FW.npy"

    points_gdf = gpd.read_file(points_path)
    route_gdf = gpd.read_file("data/outputs/jps_path.geojson")

    # Merge FW nodes + start/end points
    print("🧩 Merging start/end nodes with FW candidates...")
    points_gdf = merge_jps_points_with_path(points_gdf, route_gdf)
    print(f"📍 Merged total nodes: {len(points_gdf)}")

    D = np.load(D_path)
    FW = np.load(FW_path)
    pois_gdf, poi_weights = load_pois_and_weights()

    # Effort matrix
    params = EffortParams(method=method, processed_dir=str(base_fw_dir), buffer_radius_m=buffer_radius_m)
    E, node_stats = compute_effort_matrix(params, pois_gdf, poi_weights)
    E = np.where(np.isfinite(E), E, FW * 1.2)
    E[~np.isfinite(E)] = 1e9

    start_idx = points_gdf[points_gdf["role"] == "start"].index[0]
    end_idx = points_gdf[points_gdf["role"] == "goal"].index[0]
    print(f"🔒 Start idx={start_idx}, End idx={end_idx}")

    user_input = input(f"Enter number of stations [default={n_stations}]: ").strip()
    if user_input:
        n_stations = int(user_input)

    aco_params = {
        "n_ants": 40, "n_iterations": 80,
        "alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 1.0,
        "ideal_spacing_min": 500, "ideal_spacing_max": 800,
        "buffer_radius": 1000, "target_coverage": 0.9,
        "weights": {"spacing": 0.4, "coverage": 0.3, "poi": 0.3},
        "seed": 42,
    }

    best_route, summary = select_optimal_stations(
        E=E, node_stats=node_stats, points_gdf=points_gdf,
        corridor_gdf=route_gdf, start_idx=start_idx, end_idx=end_idx, params=aco_params,
    )

    line = _route_line_from_geojson(route_gdf.to_crs("EPSG:3857"))
    all_candidates = list(dict.fromkeys(best_route + [start_idx, end_idx]))
    ordered = _sort_indices_along_line(points_gdf, line, all_candidates)
    best_route = _enforce_exact_k(ordered, n_stations, start_idx, end_idx)

    os.makedirs(output_dir, exist_ok=True)

    # --- Export chosen stations
    chosen = points_gdf.iloc[best_route].copy()
    chosen["fw_index"] = best_route
    chosen.to_crs("EPSG:4326").to_file(f"{output_dir}/aco_jps_stations.geojson", driver="GeoJSON")

    # --- Export route
    coords = [points_gdf.iloc[i].geometry for i in best_route if not points_gdf.iloc[i].geometry.is_empty]
    if len(coords) > 1:
        line = LineString(coords)
        gpd.GeoDataFrame({"role": ["path"], "geometry": [line]}, crs="EPSG:3857").to_crs("EPSG:4326").to_file(
            f"{output_dir}/aco_jps_path.geojson", driver="GeoJSON")
    print(f"✅ Saved final route and stations in {output_dir}")

    # --- Export station buffers
    print("🟢 Generating 1 km buffers for station verification...")
    pois_m = pois_gdf.to_crs("EPSG:3857")
    stations_m = chosen.to_crs("EPSG:3857")
    buffers, poi_counts, poi_scores, poi_cats = [], [], [], []

    for geom in stations_m.geometry:
        buf = geom.buffer(1000)
        near = pois_m[pois_m.intersects(buf)]
        buffers.append(buf)
        poi_counts.append(int(len(near)))
        if len(near) > 0 and "score" in near.columns:
            poi_scores.append(float(near["score"].sum()))
        else:
            poi_scores.append(0.0)
        cats = near["category"].value_counts().head(5).to_dict() if "category" in near.columns else {}
        poi_cats.append(str(cats))

    buffers_gdf = gpd.GeoDataFrame(
        {
            "fw_index": best_route,
            "poi_count": poi_counts,
            "poi_score_sum": poi_scores,
            "top_categories": poi_cats,
            "radius_m": 1000,
        },
        geometry=buffers,
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")

    buffers_gdf.to_file(f"{output_dir}/aco_jps_station_buffers.geojson", driver="GeoJSON")
    print(f"✅ Saved 1 km station buffers → {output_dir}/aco_jps_station_buffers.geojson")


    # --- Export CSV (all nodes)
    all_nodes = points_gdf.to_crs("EPSG:4326").copy()
    all_nodes["fw_index"] = np.arange(len(all_nodes))
    all_nodes["is_station"] = all_nodes["fw_index"].isin(best_route).astype(int)
    all_nodes["poi_count"] = all_nodes["fw_index"].map(lambda i: node_stats.get(str(i), {}).get("count", 0))
    all_nodes["poi_score"] = all_nodes["fw_index"].map(lambda i: node_stats.get(str(i), {}).get("score", 0))
    all_nodes["poi_norm"] = all_nodes["fw_index"].map(lambda i: node_stats.get(str(i), {}).get("score_norm", 0))
    all_nodes["lon"], all_nodes["lat"] = all_nodes.geometry.x, all_nodes.geometry.y
    all_nodes.drop(columns="geometry").to_csv(Path(output_dir) / "aco_jps_all_nodes.csv", index=False)
    print("🧾 Exported aco_jps_all_nodes.csv (chosen + unchosen)")


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_stations", type=int, default=9)
    ap.add_argument("--method", type=str, default="jps")
    ap.add_argument("--buffer_radius_m", type=float, default=1000.0)
    ap.add_argument("--output_dir", type=str, default="data/outputs/aco")
    args = ap.parse_args()
    run_aco_jps(**vars(args))
