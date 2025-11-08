# src/algorithms/aco/aco_jps_runner.py
"""
ACO + JPS Integrated Runner
----------------------------
1. Loads candidate points (from JPS–FW pipeline)
2. Computes effort matrix based on POIs within 1 km buffers
3. Runs Ant Colony Optimization (station selection)
4. Locks start/end points (uses 'role' in FW points if present; else snaps to JPS path)
5. Enforces path order along the JPS route
6. Exports full node comparison CSV (chosen + unchosen)
7. Exports 1 km buffers for chosen stations (for visual verification)
8. Tracks ACO convergence (cost vs. iteration)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from src.algorithms.aco.aco_effort_matrix import compute_effort_matrix, EffortParams
from src.algorithms.aco.aco_station_selector import select_optimal_stations
from src.algorithms.aco.poi_scores import load_pois_and_weights

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# ======================================================
# Helpers
# ======================================================

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
    line_m = line if isinstance(line, LineString) else LineString(line)
    dists = [(i, line_m.project(pts.geometry.iloc[i])) for i in indices]
    dists.sort(key=lambda x: x[1])
    return [i for i, _ in dists]


def _enforce_exact_k(sorted_indices, k, start_idx, end_idx):
    """Trim/subsample to ensure exactly k nodes while keeping start/end."""
    keep = [i for i in sorted_indices if i not in (start_idx, end_idx)]
    result = [start_idx]
    internal_needed = max(0, k - 2)
    if len(keep) > internal_needed:
        sel_pos = np.linspace(0, len(keep) - 1, num=internal_needed, dtype=int)
        chosen = [keep[p] for p in sel_pos]
    else:
        chosen = keep
    result += chosen + [end_idx]
    # Deduplicate while preserving order
    final = []
    for i in result:
        if i not in final:
            final.append(i)
    return final[:k]


# ======================================================
# MAIN
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

    if not points_path.exists():
        raise FileNotFoundError(f"Missing candidate points: {points_path}")

    points_gdf = gpd.read_file(points_path)
    route_gdf = gpd.read_file("data/outputs/jps_path.geojson")

    print(f"📍 Loaded {len(points_gdf)} candidate nodes")

    # Load matrices (for connectivity patch later)
    D = np.load(D_path)
    FW = np.load(FW_path)

    # POIs + weights
    pois_gdf, poi_weights = load_pois_and_weights()

    # Effort matrix (built against the SAME FW node set on disk)
    params = EffortParams(method=method, processed_dir=str(base_fw_dir), buffer_radius_m=buffer_radius_m)
    E, node_stats = compute_effort_matrix(params, pois_gdf, poi_weights)
    print(f"📊 Effort matrix shape: {E.shape}")

    # Connectivity repair (use FW fallback where needed)
    E = np.where(np.isfinite(E), E, FW * 1.2)
    E[~np.isfinite(E)] = 1e9
    print(f"🔍 Connectivity repaired. Finite ratio: {(np.isfinite(E).sum() / E.size):.2%}")

    # Determine start/end
    if "role" in points_gdf.columns and {"start", "end"}.issubset(set(points_gdf["role"].dropna().unique())):
        start_idx = int(points_gdf.index[points_gdf["role"] == "start"][0])
        end_idx = int(points_gdf.index[points_gdf["role"] == "end"][0])
        print("🔒 Start/end found in FW points (role column).")
    else:
        # Fallback: snap to path start/goal
        start_pt = route_gdf[route_gdf["role"] == "start"].geometry.iloc[0]
        goal_pt = route_gdf[route_gdf["role"] == "goal"].geometry.iloc[0]
        pts_m = points_gdf.to_crs("EPSG:3857")
        start_idx = pts_m.distance(start_pt).idxmin()
        end_idx = pts_m.distance(goal_pt).idxmin()
        if start_idx == end_idx:
            dists = pts_m.distance(goal_pt)
            end_idx = int(np.argmax(dists.values))
        print("🔒 Start/end snapped from JPS path.")

    print(f"   → start_idx={start_idx}, end_idx={end_idx}")

    # Station count (interactive)
    try:
        user_input = input(
            f"Enter number of stations (including start/end) [default={n_stations}]: "
        ).strip()
        if user_input:
            n_stations = int(user_input)
    except Exception:
        pass
    print(f"🎯 Selecting {n_stations} stations.")

    # ACO hyperparams
    aco_params = {
        "n_ants": 40, "n_iterations": 80,
        "alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 1.0,
        "ideal_spacing_min": 500, "ideal_spacing_max": 800,
        "buffer_radius": 1000, "target_coverage": 0.9,
        "weights": {"spacing": 0.4, "coverage": 0.3, "poi": 0.3},
        "seed": 42,
    }

    # Run ACO
    best_route, summary = select_optimal_stations(
        E=E,
        node_stats=node_stats,
        points_gdf=points_gdf,
        corridor_gdf=route_gdf,
        start_idx=start_idx,
        end_idx=end_idx,
        params=aco_params,
    )

    # Save convergence data (if present)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = summary.get("history", {})
    best_costs = history.get("best_cost", [])
    mean_costs = history.get("mean_cost", [])
    if best_costs:
        df_hist = pd.DataFrame(
            {"iteration": np.arange(1, len(best_costs) + 1),
             "best_cost": best_costs,
             "mean_cost": mean_costs if len(mean_costs) == len(best_costs) else [np.nan]*len(best_costs)}
        )
        df_hist.to_csv(output_dir / "aco_convergence.csv", index=False)
        print("📈 Saved ACO convergence log → aco_convergence.csv")

        plt.figure()
        plt.plot(df_hist["iteration"], df_hist["best_cost"], label="Best Cost")
        if df_hist["mean_cost"].notna().any():
            plt.plot(df_hist["iteration"], df_hist["mean_cost"], "--", label="Mean Cost")
        plt.xlabel("Iteration"); plt.ylabel("Cost")
        plt.title("ACO Convergence Over Iterations")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(output_dir / "aco_convergence.png", dpi=200)
        plt.close()
        print("📉 Saved ACO convergence plot → aco_convergence.png")

    # Order along the corridor and enforce exact K
    line = _route_line_from_geojson(route_gdf.to_crs("EPSG:3857"))
    if line is None:
        raise RuntimeError("Could not extract LineString from jps_path.geojson.")
    all_candidates = list(dict.fromkeys(best_route + [start_idx, end_idx]))
    ordered = _sort_indices_along_line(points_gdf, line, all_candidates)
    best_route = _enforce_exact_k(ordered, n_stations, start_idx, end_idx)

    # Export chosen stations
    chosen = points_gdf.iloc[best_route].copy()
    chosen.loc[:, "fw_index"] = best_route
    chosen.to_crs("EPSG:4326").to_file(output_dir / "aco_jps_stations.geojson", driver="GeoJSON")
    print("✅ Saved chosen stations → aco_jps_stations.geojson")

    # Export route line
    coords = [points_gdf.iloc[i].geometry for i in best_route if not points_gdf.iloc[i].geometry.is_empty]
    if len(coords) > 1:
        route_line = LineString(coords)
        gpd.GeoDataFrame({"role": ["path"], "geometry": [route_line]}, crs="EPSG:3857") \
           .to_crs("EPSG:4326") \
           .to_file(output_dir / "aco_jps_path.geojson", driver="GeoJSON")
        print("✅ Saved final route → aco_jps_path.geojson")
    else:
        print("⚠️ Not enough points to create route line.")

    # Export 1 km buffers for verification
    print("🟢 Generating 1 km buffers for station verification...")
    pois_m = pois_gdf.to_crs("EPSG:3857")
    stations_m = chosen.to_crs("EPSG:3857")
    buffers, poi_counts, poi_scores, poi_cats = [], [], [], []
    for geom in stations_m.geometry:
        buf = geom.buffer(1000)
        near = pois_m[pois_m.intersects(buf)]
        buffers.append(buf)
        poi_counts.append(int(len(near)))
        poi_scores.append(float(near["score"].sum()) if "score" in near.columns and len(near) else 0.0)
        cats = near["category"].value_counts().head(5).to_dict() if "category" in near.columns and len(near) else {}
        poi_cats.append(str(cats))

    buffers_gdf = gpd.GeoDataFrame(
        {"fw_index": best_route,
         "poi_count": poi_counts,
         "poi_score_sum": poi_scores,
         "top_categories": poi_cats,
         "radius_m": 1000},
        geometry=buffers, crs="EPSG:3857"
    ).to_crs("EPSG:4326")
    buffers_gdf.to_file(output_dir / "aco_jps_station_buffers.geojson", driver="GeoJSON")
    print("✅ Saved 1 km station buffers → aco_jps_station_buffers.geojson")

    # Export all-nodes CSV (chosen + unchosen)
    all_nodes = points_gdf.to_crs("EPSG:4326").copy()
    all_nodes.loc[:, "fw_index"] = np.arange(len(all_nodes))
    chosen_set = set(best_route)
    all_nodes.loc[:, "is_station"] = all_nodes["fw_index"].apply(lambda i: int(i in chosen_set))
    all_nodes.loc[:, "poi_count"] = all_nodes["fw_index"].map(lambda i: node_stats.get(str(i), {}).get("count", 0))
    all_nodes.loc[:, "poi_score"] = all_nodes["fw_index"].map(lambda i: node_stats.get(str(i), {}).get("score", 0))
    all_nodes.loc[:, "poi_norm"] = all_nodes["fw_index"].map(lambda i: node_stats.get(str(i), {}).get("score_norm", 0))
    all_nodes.loc[:, "lon"] = all_nodes.geometry.x
    all_nodes.loc[:, "lat"] = all_nodes.geometry.y
    all_nodes.drop(columns="geometry").to_csv(output_dir / "aco_jps_all_nodes.csv", index=False)
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
