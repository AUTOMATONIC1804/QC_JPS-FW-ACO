# src/algorithms/aco/aco_jps_runner.py
"""
ACO + JPS Integrated Runner
----------------------------
1. Loads candidate points (from JPS–FW pipeline)
2. Computes effort matrix based on POIs within 1 km buffers
3. Runs Ant Colony Optimization (station selection)
4. Locks start/end points
5. Rebuilds final route via JPS
"""

import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from src.algorithms.aco.aco_effort_matrix import compute_effort_matrix, EffortParams
from src.algorithms.aco.aco_station_selector import select_optimal_stations
from src.algorithms.aco.poi_scores import load_pois_and_weights


# ======================================================
# Helper functions
# ======================================================

def ensure_start_end(route, start_idx, end_idx):
    """Ensure the route starts and ends at the correct indices."""
    r = list(route)
    if not r:
        return [start_idx, end_idx] if start_idx != end_idx else [start_idx]
    if r[0] != start_idx:
        if start_idx not in r:
            r = [start_idx] + r
        else:
            r.remove(start_idx)
            r = [start_idx] + r
    if r[-1] != end_idx:
        if end_idx not in r:
            r = r + [end_idx]
        else:
            r.remove(end_idx)
            r = r + [end_idx]
    # Deduplicate consecutive indices
    dedup = []
    for v in r:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    return dedup


def fill_to_k(route, k, E, node_stats, points_gdf, ideal_min=500, ideal_max=800):
    """Greedily insert intermediate stations until reaching K total nodes."""
    if not route:
        return route
    route = list(route)
    n = len(points_gdf)
    available = set(range(n)) - set(route)
    points_m = points_gdf.to_crs("EPSG:3857")

    def dist_idx(i, j):
        return points_m.iloc[i].geometry.distance(points_m.iloc[j].geometry)

    while len(route) < k and available:
        best_gain = None
        best_insert = None
        for j in list(available):
            poi_bonus = node_stats.get(str(j), {}).get("score_norm", 0.0)
            for pos in range(len(route) - 1):
                a, b = route[pos], route[pos + 1]
                ea = E[a, j]
                eb = E[j, b]
                eab = E[a, b]
                if not np.isfinite(ea) or not np.isfinite(eb):
                    continue
                if dist_idx(a, j) < 0.7 * ideal_min or dist_idx(j, b) < 0.7 * ideal_min:
                    continue
                gain = (ea + eb - eab) * (1 - 0.25 * poi_bonus)
                if best_gain is None or gain < best_gain:
                    best_gain = gain
                    best_insert = (pos + 1, j)
        if best_insert is None:
            # Relax spacing if needed
            for j in list(available):
                poi_bonus = node_stats.get(str(j), {}).get("score_norm", 0.0)
                for pos in range(len(route) - 1):
                    a, b = route[pos], route[pos + 1]
                    ea = E[a, j]
                    eb = E[j, b]
                    eab = E[a, b]
                    if not np.isfinite(ea) or not np.isfinite(eb):
                        continue
                    gain = (ea + eb - eab) * (1 - 0.25 * poi_bonus)
                    if best_gain is None or gain < best_gain:
                        best_gain = gain
                        best_insert = (pos + 1, j)
            if best_insert is None:
                break
        pos, j = best_insert
        route.insert(pos, j)
        available.discard(j)
    return route


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

    # --------------------------------------------------
    # 1️⃣ Load JPS/Dijkstra/A* candidate nodes and matrices
    # --------------------------------------------------
    base_fw_dir = Path("data/outputs/floyd_warshall")

    if method == "jps":
        points_path = base_fw_dir / "fw_jps_points.geojson"
        D_path = base_fw_dir / "fw_jps_D.npy"
        FW_path = base_fw_dir / "fw_jps_FW.npy"
    elif method == "dijkstra":
        points_path = base_fw_dir / "fw_dijkstra_points.geojson"
        D_path = base_fw_dir / "fw_dijkstra_D.npy"
        FW_path = base_fw_dir / "fw_dijkstra_FW.npy"
    elif method == "astar":
        points_path = base_fw_dir / "fw_astar_points.geojson"
        D_path = base_fw_dir / "fw_astar_D.npy"
        FW_path = base_fw_dir / "fw_astar_FW.npy"
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not points_path.exists():
        raise FileNotFoundError(f"Missing candidate points: {points_path}")

    points_gdf = gpd.read_file(points_path)
    print(f"📍 Loaded {len(points_gdf)} candidate nodes")

    D = np.load(D_path)
    FW = np.load(FW_path)

    # --------------------------------------------------
    # 2️⃣ Load POIs and category weights
    # --------------------------------------------------
    pois_gdf, poi_weights = load_pois_and_weights()

    # --------------------------------------------------
    # 3️⃣ Compute Effort Matrix (E)
    # --------------------------------------------------
    params = EffortParams(
        method=method,
        processed_dir=str(base_fw_dir),
        buffer_radius_m=buffer_radius_m,
    )

    E, node_stats = compute_effort_matrix(params, pois_gdf, poi_weights)
    print(f"📊 Effort matrix shape: {E.shape}")

    # --- Connectivity repair: replace inf with FW fallback ---
    E = np.where(np.isfinite(E), E, FW * 1.2)
    E[~np.isfinite(E)] = 1e9
    finite_ratio = np.isfinite(E).sum() / E.size
    print(f"🔍 Effort matrix connectivity: {finite_ratio:.2%} finite entries ({np.isfinite(E).sum()} / {E.size})")

    # --------------------------------------------------
    # 4️⃣ Select start & end indices
    # --------------------------------------------------
    route_path = Path("data/outputs/jps_path.geojson")
    if not route_path.exists():
        raise FileNotFoundError(f"Missing route file: {route_path}")

    route_gdf = gpd.read_file(route_path)
    start_pt = route_gdf[route_gdf["role"] == "start"].geometry.iloc[0]
    goal_pt = route_gdf[route_gdf["role"] == "goal"].geometry.iloc[0]

    points_m = points_gdf.to_crs("EPSG:3857")
    start_idx = points_m.distance(start_pt).idxmin()
    end_idx = points_m.distance(goal_pt).idxmin()
    # Find nearest candidate points
    points_m = points_gdf.to_crs("EPSG:3857")
    start_idx = points_m.distance(start_pt).idxmin()
    end_idx = points_m.distance(goal_pt).idxmin()

    # --- Safety patch: prevent start=end ---
    if start_idx == end_idx:
        print(f"⚠️ Start and end snapped to the same point ({start_idx}). Fixing…")
    # find the farthest point instead
        dists = points_m.distance(goal_pt)
        end_idx = int(np.argmax(dists.values))
        print(f"✅ Reassigned end_idx={end_idx}")
    else:
        print(f"🔒 Locked start_idx={start_idx}, end_idx={end_idx}")

    # --------------------------------------------------
    # 5️⃣ Station count input
    # --------------------------------------------------
    try:
        user_input = input(
            f"Enter number of stations to select (including start and end) [default={n_stations}]: "
        ).strip()
        if user_input:
            n_stations = int(user_input)
    except Exception:
        pass
    print(f"🎯 Selecting {n_stations} stations (including start/end).")

    # --------------------------------------------------
    # 6️⃣ ACO parameters
    # --------------------------------------------------
    aco_params = {
        "n_ants": 40,
        "n_iterations": 80,
        "alpha": 1.0,
        "beta": 3.0,
        "rho": 0.5,
        "Q": 1.0,
        "ideal_spacing_min": 500,
        "ideal_spacing_max": 800,
        "buffer_radius": 1000,
        "target_coverage": 0.9,
        "weights": {"spacing": 0.4, "coverage": 0.3, "poi": 0.3},
        "seed": 42,
    }

    # --------------------------------------------------
    # 7️⃣ Run ACO
    # --------------------------------------------------
    best_route, summary = select_optimal_stations(
        E=E,
        node_stats=node_stats,
        points_gdf=points_gdf,
        corridor_gdf=route_gdf,
        start_idx=start_idx,
        end_idx=end_idx,
        params=aco_params,
    )

    # --- enforce start/end and fill up ---
    best_route = ensure_start_end(best_route, start_idx, end_idx)
    if len(best_route) < n_stations:
        best_route = fill_to_k(best_route, n_stations, E, node_stats, points_gdf,
                               ideal_min=aco_params["ideal_spacing_min"],
                               ideal_max=aco_params["ideal_spacing_max"])

    if len(best_route) < 2:
        print("❌ Still not enough stations to build a path after repair.")
        return

    print(f"✅ Best route indices (repaired): {best_route}")
    print(f"📊 Summary: {json.dumps(summary, indent=2)}")

    # --------------------------------------------------
    # 8️⃣ Export chosen stations
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    chosen = points_gdf.iloc[best_route]
    chosen.to_crs("EPSG:4326").to_file(
        f"{output_dir}/aco_jps_stations.geojson", driver="GeoJSON"
    )
    print(f"✅ Saved chosen stations → {output_dir}/aco_jps_stations.geojson")

    # --------------------------------------------------
    # 9️⃣ Rebuild route connecting chosen stations
    # --------------------------------------------------
    print(f"🚉 Rebuilding route using {len(best_route)} stations via JPS…")
    coords = [pt for pt in [points_gdf.iloc[i].geometry for i in best_route] if pt and not pt.is_empty]
    unique_coords = []
    for c in coords:
        if not unique_coords or not c.equals(unique_coords[-1]):
            unique_coords.append(c)

    if len(unique_coords) > 1:
        line = LineString(unique_coords)
        gdf = gpd.GeoDataFrame(
            {"role": ["path"] + ["station"] * len(unique_coords),
             "geometry": [line] + unique_coords},
            crs="EPSG:3857",
        )
        gdf.to_crs("EPSG:4326").to_file(
            f"{output_dir}/aco_jps_path.geojson", driver="GeoJSON"
        )
        print(f"✅ Saved final route → {output_dir}/aco_jps_path.geojson")
    else:
        print("⚠️ Not enough unique coordinates to build a LineString — skipping route export.")

    # --------------------------------------------------
    # 🔟 Save summary
    # --------------------------------------------------
    summary_path = Path(output_dir) / "aco_jps_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved summary → {summary_path}")


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
