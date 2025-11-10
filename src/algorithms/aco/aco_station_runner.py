# src/algorithms/aco/aco_station_runner.py
"""
ACO Station Runner (helper module)
----------------------------
Provides run_aco_jps(...) as an importable helper. This module is not intended
to be executed as a script; remove the CLI entry if you previously ran it with
python -m ...
"""

from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.algorithms.aco.aco_effort_matrix import compute_effort_matrix, EffortParams
from src.algorithms.aco.aco_station_selector import select_optimal_stations
from src.algorithms.aco.poi_scores import load_pois_and_weights

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

__all__ = ["run_aco_jps"]

# ------------------------ small helpers ------------------------

def _route_line_from_geojson(route_gdf: gpd.GeoDataFrame):
    """Extract the main LineString path from jps_path.geojson."""
    if "role" in route_gdf.columns:
        line_rows = route_gdf[route_gdf["role"] == "path"]
        if not line_rows.empty:
            return line_rows.geometry.iloc[0]
    lines = [g for g in route_gdf.geometry if isinstance(g, LineString)]
    return lines[0] if lines else None


def _sort_indices_along_line(points_gdf: gpd.GeoDataFrame, line: LineString, indices):
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
    final = []
    for i in result:
        if i not in final:
            final.append(i)
    return final[:k]


# ------------------------ main helper ------------------------

def run_aco_jps(
    n_stations: int = 9,
    method: str = "jps",
    buffer_radius_m: float = 1000.0,
    output_dir: str = "data/outputs/aco",
):
    """
    Run ACO + JPS integrated optimization and write outputs to output_dir.
    This function is intended to be imported and called from other code.
    """
    print("=== 🧠 Running ACO + JPS Integrated Optimization ===")

    base_fw_dir = Path("data/outputs/floyd_warshall")
    points_path_full = base_fw_dir / f"fw_{method}_points.geojson"
    D_path = base_fw_dir / f"fw_{method}_D.npy"
    FW_path = base_fw_dir / f"fw_{method}_FW.npy"
    route_path = Path("data/outputs/jps_path.geojson")
    detour_path = Path("data/outputs/aco/debug_detour_nodes.geojson")

    # 1) Load full FW nodes in canonical order (these define row/col order for D/FW/E_full)
    if not points_path_full.exists():
        raise FileNotFoundError(f"Missing candidate points: {points_path_full}")
    points_full = gpd.read_file(points_path_full).to_crs("EPSG:4326")
    print(f"📍 Loaded {len(points_full)} FW nodes (full set, canonical order)")

    if not route_path.exists():
        raise FileNotFoundError(f"Missing JPS path: {route_path}")
    route_gdf = gpd.read_file(route_path)

    # 2) Detour filter → produce feasible ORIGINAL indices (in FW order)
    if detour_path.exists():
        print("🧭 Using debug_detour_nodes.geojson as authoritative filter…")
        detour_ll = gpd.read_file(detour_path).to_crs("EPSG:4326")
        if "detour_label" not in detour_ll.columns:
            raise RuntimeError("❌ debug_detour_nodes.geojson missing 'detour_label' column.")

        detour_ll["__is_feasible__"] = detour_ll["detour_label"].astype(str).str.lower().isin(
            ["feasible", "true", "1"]
        )

        # Prefer direct fw_index if present
        if "fw_index" in detour_ll.columns:
            idx_raw = detour_ll["fw_index"].dropna().astype(int).tolist()
            idx_raw = [i for i in idx_raw if 0 <= i < len(points_full)]
            feas_mask = detour_ll["__is_feasible__"].values
            feas_orig_idx = [i for i, ok in zip(idx_raw, feas_mask) if ok]
            # Always preserve explicit start/end from detour file if present
            if "role" in detour_ll.columns:
                se_idx = detour_ll.loc[
                    detour_ll["role"].astype(str).str.lower().isin(["start", "end", "goal"]),
                    "fw_index"
                ].dropna().astype(int).tolist()
                feas_orig_idx = sorted(set(feas_orig_idx + se_idx))
        else:
            # No fw_index → nearest snap to FW nodes in EPSG:3857 (metric)
            print("ℹ️ detour file has no 'fw_index' → snapping detour to FW nodes (≤ 50 m)…")
            detour_m = detour_ll.to_crs("EPSG:3857")
            points_full_m = points_full.to_crs("EPSG:3857")
            joined = gpd.sjoin_nearest(
                detour_m[["geometry", "__is_feasible__", *(["role"] if "role" in detour_ll.columns else [])]],
                points_full_m.reset_index()[["index", "geometry"]],
                how="left",
                max_distance=50,
            )
            feas_rows = joined[joined["__is_feasible__"] == True]
            feas_orig_idx = feas_rows["index"].dropna().astype(int).tolist()
            if "role" in detour_ll.columns:
                se_rows = joined[joined.get("role", "").astype(str).str.lower().isin(["start", "end", "goal"])]
                se_idx = se_rows["index"].dropna().astype(int).tolist()
                feas_orig_idx = sorted(set(feas_orig_idx + se_idx))

        feas_orig_idx = sorted(set(feas_orig_idx))
        if len(feas_orig_idx) == 0:
            raise RuntimeError("❌ Detour filter resulted in 0 feasible nodes.")
        print(f"✅ Feasible detour nodes mapped to FW indices: {len(feas_orig_idx)}")
    else:
        print("⚠️ debug_detour_nodes.geojson not found — using all FW nodes.")
        feas_orig_idx = list(range(len(points_full)))

    # Build feasible node frame in original order
    points_feas = points_full.iloc[feas_orig_idx].copy().reset_index(drop=True)

    # Carry role info from the original FW points if present
    if "role" in points_full.columns:
        role_map = points_full["role"]
        roles_on_feas = [role_map.iloc[i] if i < len(role_map) else None for i in feas_orig_idx]
        points_feas["role"] = roles_on_feas

    # If detour file has role, fill missing roles via nearest (in EPSG:3857)
    if detour_path.exists():
        detour_ll2 = gpd.read_file(detour_path)
        if "role" in detour_ll2.columns:
            detour_roles_m = detour_ll2.to_crs("EPSG:3857")[["geometry", "role"]]
            feas_m = points_feas.to_crs("EPSG:3857")[["geometry"]]
            j = gpd.sjoin_nearest(feas_m, detour_roles_m, how="left", max_distance=50)
            # only fill NA values
            if "role" in points_feas.columns:
                points_feas["role"] = points_feas["role"].fillna(j["role"])
            else:
                points_feas["role"] = j["role"]

    print(f"📍 Candidate nodes after detour filtering: {len(points_feas)}")

    # 3) Load matrices for the FULL set (we will slice later)
    D_full = np.load(D_path)
    FW_full = np.load(FW_path)

    # 4) Build FULL effort matrix against FULL FW set (consistent with points_full order)
    pois_gdf, poi_weights = load_pois_and_weights()
    E_full, node_stats_full = compute_effort_matrix(
        EffortParams(method=method, processed_dir=str(base_fw_dir), buffer_radius_m=buffer_radius_m),
        pois_gdf,
        poi_weights
    )
    if E_full.shape[0] != len(points_full):
        raise RuntimeError("Effort matrix size does not match full FW points length.")

    # 5) Slice effort matrix + node stats down to feasible subset
    E = E_full[np.ix_(feas_orig_idx, feas_orig_idx)]
    node_stats = {
        str(new_i): node_stats_full.get(
            str(old_i), {"score_norm": 0.0, "count": 0.0, "score": 0.0}
        )
        for new_i, old_i in enumerate(feas_orig_idx)
    }

    # Connectivity repair on the sliced matrix
    FW_slice = FW_full[np.ix_(feas_orig_idx, feas_orig_idx)]
    E = np.where(np.isfinite(E), E, FW_slice * 1.2)
    E[~np.isfinite(E)] = 1e9
    print(f"📊 Effort matrix (feasible-only) shape: {E.shape}")
    print(f"🔍 Connectivity repaired. Finite ratio: {(np.isfinite(E).sum() / E.size):.2%}")

    # 6) Determine start/end on the feasible subset
    def _find_se_idx_from_roles(gdf):
        start_candidates = gdf.index[gdf.get("role", "").astype(str).str.lower().eq("start")]
        end_candidates = gdf.index[gdf.get("role", "").astype(str).str.lower().isin(["goal", "end"])]
        return (int(start_candidates[0]) if len(start_candidates) else None,
                int(end_candidates[0]) if len(end_candidates) else None)

    start_idx, end_idx = _find_se_idx_from_roles(points_feas)

    if start_idx is None or end_idx is None:
        print("⚠️ Start/goal missing on feasible set — snapping from JPS path…")
        path_start = route_gdf[route_gdf["role"] == "start"].geometry.iloc[0]
        path_goal = route_gdf[route_gdf["role"] == "goal"].geometry.iloc[0]
        pts_m = points_feas.to_crs("EPSG:3857")
        start_idx = int(pts_m.distance(path_start).idxmin())
        end_idx = int(pts_m.distance(path_goal).idxmin())
        if start_idx == end_idx:
            dists = pts_m.distance(path_goal)
            end_idx = int(np.argmax(dists.values))
    print(f"🔒 Start/end confirmed → start_idx={start_idx}, end_idx={end_idx}")

    # 7) Ask for station count
    try:
        user_input = input(
            f"Enter number of stations (including start/end) [default={n_stations}]: "
        ).strip()
        if user_input:
            n_stations = int(user_input)
            print(f"✅ Station count manually set to {n_stations}.")
        else:
            print(f"ℹ️ Using default station count: {n_stations}")
    except Exception:
        print(f"⚠️ Invalid input, using default station count ({n_stations}).")

    # 8) ACO params
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

    # 9) Run ACO on feasible-only set
    best_route, summary = select_optimal_stations(
        E=E,
        node_stats=node_stats,
        points_gdf=points_feas,     # aligned with E rows/cols
        corridor_gdf=route_gdf,
        start_idx=start_idx,
        end_idx=end_idx,
        params=aco_params,
    )

    # 10) Outputs & convergence logs
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    history = summary.get("history", {})
    best_costs = history.get("best_cost", [])
    mean_costs = history.get("mean_cost", [])
    if best_costs:
        df_hist = pd.DataFrame({
            "iteration": np.arange(1, len(best_costs) + 1),
            "best_cost": best_costs,
            "mean_cost": mean_costs if len(mean_costs) == len(best_costs) else [np.nan]*len(best_costs)
        })
        df_hist.to_csv(outdir / "aco_convergence.csv", index=False)
        print("📈 Saved ACO convergence log → aco_convergence.csv")

        plt.figure()
        plt.plot(df_hist["iteration"], df_hist["best_cost"], label="Best Cost")
        if df_hist["mean_cost"].notna().any():
            plt.plot(df_hist["iteration"], df_hist["mean_cost"], "--", label="Mean Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("ACO Convergence Over Iterations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / "aco_convergence.png", dpi=200)
        plt.close()
        print("📉 Saved ACO convergence plot → aco_convergence.png")

    # 11) Order along corridor and enforce exact K
    line = _route_line_from_geojson(route_gdf.to_crs("EPSG:3857"))
    if line is None:
        raise RuntimeError("Could not extract LineString from jps_path.geojson.")

    all_candidates = list(dict.fromkeys(best_route + [start_idx, end_idx]))
    ordered = _sort_indices_along_line(points_feas, line, all_candidates)

    # 🧩 FIX: If ACO only returned start/end, fill with evenly spaced nodes
    if len(ordered) < n_stations:
        print(f"⚠️ ACO returned only {len(ordered)} nodes; filling up to {n_stations}...")
        remaining_needed = n_stations - len(ordered)
        feasible_idxs = list(range(len(points_feas)))

        # Pick evenly spaced filler indices (excluding those already in route)
        filler_idxs = np.linspace(
            0, len(feasible_idxs) - 1,
            num=remaining_needed + 2,
            dtype=int
        )[1:-1]
        filler_idxs = [i for i in filler_idxs if i not in ordered]

        ordered += filler_idxs
        ordered = sorted(set(ordered), key=lambda i: line.project(points_feas.to_crs("EPSG:3857").geometry.iloc[i]))

    best_route = _enforce_exact_k(ordered, n_stations, start_idx, end_idx)


    # 12) Export chosen stations
    chosen = points_feas.iloc[best_route].copy()
    chosen.loc[:, "sub_index"] = best_route  # index within feasible subproblem
    # Also carry back original FW index for traceability
    chosen.loc[:, "fw_index"] = [feas_orig_idx[i] for i in best_route]
    chosen.to_crs("EPSG:4326").to_file(outdir / "aco_jps_stations.geojson", driver="GeoJSON")
    print("✅ Saved chosen stations → aco_jps_stations.geojson")

    # 14) Export 1 km buffers for verification
    print("🟢 Generating 1 km buffers for station verification…")
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
        {"sub_index": best_route,
         "fw_index": [feas_orig_idx[i] for i in best_route],
         "poi_count": poi_counts,
         "poi_score_sum": poi_scores,
         "top_categories": poi_cats,
         "radius_m": 1000},
        geometry=buffers, crs="EPSG:3857"
    ).to_crs("EPSG:4326")
    buffers_gdf.to_file(outdir / "aco_jps_station_buffers.geojson", driver="GeoJSON")
    print("✅ Saved 1 km station buffers → aco_jps_station_buffers.geojson")

    # 15) Export all-nodes CSV (feasible-only universe = what ACO evaluated)
    all_nodes = points_feas.to_crs("EPSG:4326").copy()
    all_nodes.loc[:, "sub_index"] = np.arange(len(all_nodes))
    all_nodes.loc[:, "fw_index"] = feas_orig_idx
    chosen_set = set(best_route)
    all_nodes.loc[:, "is_station"] = all_nodes["sub_index"].apply(lambda i: int(i in chosen_set))
    all_nodes.loc[:, "poi_count"] = all_nodes["sub_index"].map(
        lambda i: node_stats.get(str(i), {}).get("count", 0)
    )
    all_nodes.loc[:, "poi_score"] = all_nodes["sub_index"].map(
        lambda i: node_stats.get(str(i), {}).get("score", 0)
    )
    all_nodes.loc[:, "poi_norm"] = all_nodes["sub_index"].map(
        lambda i: node_stats.get(str(i), {}).get("score_norm", 0)
    )
    all_nodes.loc[:, "lon"] = all_nodes.geometry.x
    all_nodes.loc[:, "lat"] = all_nodes.geometry.y
    all_nodes.drop(columns="geometry").to_csv(outdir / "aco_jps_all_nodes.csv", index=False)
    print("🧾 Exported aco_jps_all_nodes.csv (feasible universe: chosen + unchosen)")

    # Return core results for callers
    return {
        "chosen_stations_gdf": chosen,
        "buffers_gdf": buffers_gdf,
        "all_nodes_df": all_nodes,
        "best_route": best_route,
        "summary": summary,
    }


# ===================================================
# Prevent direct execution
# ===================================================
if __name__ == "__main__":
    raise RuntimeError(
        "This module is a helper and should not be executed directly. "
        "Use 'aco_jps_runner.py' instead, which calls run_aco_jps() internally."
    )