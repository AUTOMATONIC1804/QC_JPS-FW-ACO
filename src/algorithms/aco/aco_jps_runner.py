# src/algorithms/aco/aco_jps_runner.py
"""
ACO + JPS Integrated Runner
----------------------------
1. Loads candidate points (from JPS–FW pipeline)
2. Computes effort matrix based on POIs within 1 km buffers
3. Runs Ant Colony Optimization (station selection)
4. Locks start/end points
5. Enforces path order along the JPS route
6. Rebuilds final route
7. Exports detailed POI stats for each chosen station
"""

import os
import json
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
    """Ensure route starts and ends at locked indices."""
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


def _enforce_exact_k(sorted_indices, k, start_idx, end_idx, node_stats):
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
    # Deduplicate
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

    # --------------------------------------------------
    # 1️⃣ Load Floyd–Warshall outputs (method-specific)
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
    # 2️⃣ Load POIs and weights
    # --------------------------------------------------
    pois_gdf, poi_weights = load_pois_and_weights()

    # --------------------------------------------------
    # 3️⃣ Compute Effort Matrix
    # --------------------------------------------------
    params = EffortParams(
        method=method,
        processed_dir=str(base_fw_dir),
        buffer_radius_m=buffer_radius_m,
    )
    E, node_stats = compute_effort_matrix(params, pois_gdf, poi_weights)
    print(f"📊 Effort matrix shape: {E.shape}")

    # --- Connectivity repair ---
    E = np.where(np.isfinite(E), E, FW * 1.2)
    E[~np.isfinite(E)] = 1e9
    print(f"🔍 Connectivity repaired. Finite ratio: {(np.isfinite(E).sum() / E.size):.2%}")

    # --------------------------------------------------
    # 4️⃣ Get start and end nodes from JPS route
    # --------------------------------------------------
    route_path = Path("data/outputs/jps_path.geojson")
    if not route_path.exists():
        raise FileNotFoundError(f"Missing JPS path: {route_path}")

    route_gdf = gpd.read_file(route_path)
    start_pt = route_gdf[route_gdf["role"] == "start"].geometry.iloc[0]
    goal_pt = route_gdf[route_gdf["role"] == "goal"].geometry.iloc[0]

    points_m = points_gdf.to_crs("EPSG:3857")
    start_idx = points_m.distance(start_pt).idxmin()
    end_idx = points_m.distance(goal_pt).idxmin()

    if start_idx == end_idx:
        print(f"⚠️ Start and end snapped to same node ({start_idx}); reassigning end.")
        dists = points_m.distance(goal_pt)
        end_idx = int(np.argmax(dists.values))
    print(f"🔒 Locked start_idx={start_idx}, end_idx={end_idx}")

    # --------------------------------------------------
    # 5️⃣ Station count
    # --------------------------------------------------
    try:
        user_input = input(
            f"Enter number of stations to select (including start/end) [default={n_stations}]: "
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
    # 7️⃣ Run ACO selection
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

    # --- Order and enforce exact count ---
    line = _route_line_from_geojson(route_gdf.to_crs("EPSG:3857"))
    if line is None:
        raise RuntimeError("Could not extract path LineString from jps_path.geojson.")

    all_candidates = list(dict.fromkeys(best_route + [start_idx, end_idx]))
    ordered = _sort_indices_along_line(points_gdf, line, all_candidates)
    best_route = _enforce_exact_k(ordered, n_stations, start_idx, end_idx, node_stats)

    # --------------------------------------------------
    # 🧠 Print detailed station POI info
    # --------------------------------------------------
    print("\n🚉 Detailed Station Summary:")
    for i, idx in enumerate(best_route):
        stats = node_stats.get(str(idx), {})
        top_pois = ", ".join(stats.get("top_pois", [])[:5]) or "None"
        print(f"  {i+1:02d}. Node {idx} → Count={stats.get('count', 0):.0f}, "
              f"Score={stats.get('score', 0):.2f}, Norm={stats.get('score_norm', 0):.2f}")
        print(f"     Top POIs: {top_pois}")

    # --------------------------------------------------
    # 8️⃣ Export stations with POI details
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    chosen = points_gdf.iloc[best_route].copy()
    chosen["fw_index"] = best_route
    chosen["poi_count"] = [node_stats.get(str(i), {}).get("count", 0) for i in best_route]
    chosen["poi_score"] = [node_stats.get(str(i), {}).get("score", 0) for i in best_route]
    chosen["poi_norm"] = [node_stats.get(str(i), {}).get("score_norm", 0) for i in best_route]
    chosen["top_pois"] = [", ".join(node_stats.get(str(i), {}).get("top_pois", [])[:5]) for i in best_route]
    chosen["order"] = np.arange(1, len(chosen) + 1)
    chosen.to_crs("EPSG:4326").to_file(f"{output_dir}/aco_jps_stations.geojson", driver="GeoJSON")
    print(f"✅ Saved chosen stations → {output_dir}/aco_jps_stations.geojson")

    # --------------------------------------------------
    # 9️⃣ Rebuild route
    # --------------------------------------------------
    coords = [points_gdf.iloc[i].geometry for i in best_route if not points_gdf.iloc[i].geometry.is_empty]
    if len(coords) > 1:
        line = LineString(coords)
        gdf = gpd.GeoDataFrame({"role": ["path"], "geometry": [line]}, crs="EPSG:3857")
        gdf.to_crs("EPSG:4326").to_file(f"{output_dir}/aco_jps_path.geojson", driver="GeoJSON")
        print(f"✅ Saved final route → {output_dir}/aco_jps_path.geojson")
    else:
        print("⚠️ Not enough points to create route line.")

    # --------------------------------------------------
    # 🔟 Save extended summary
    # --------------------------------------------------
    summary["selected_nodes"] = best_route
    summary["node_stats"] = {str(i): node_stats.get(str(i), {}) for i in best_route}
    with open(Path(output_dir) / "aco_jps_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved summary → {output_dir}/aco_jps_summary.json")


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
