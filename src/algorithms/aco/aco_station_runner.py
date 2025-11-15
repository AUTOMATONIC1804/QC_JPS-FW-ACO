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
import time

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
    """Extract the main LineString path from a route GeoJSON file."""
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

def _enforce_minimum_spacing(points_gdf: gpd.GeoDataFrame, ordered_indices: list, 
                             min_spacing_m: float, start_idx: int, end_idx: int, 
                             target_count: int) -> list:
    """
    Enforce minimum spacing while maintaining target station count.
    Priority: exact station count > spacing constraint.
    """
    if len(ordered_indices) <= 2:
        return ordered_indices
    
    points_m = points_gdf.to_crs("EPSG:3857")
    filtered = []
    
    # Separate start/end from regular stations
    regular_stations = [idx for idx in ordered_indices if idx not in (start_idx, end_idx)]
    
    # Always include start first
    if start_idx in ordered_indices:
        filtered.append(start_idx)
    
    # Process regular stations with spacing constraint, but ensure we reach target
    needed_regular = max(0, target_count - 2)  # -2 for start/end
    
    for i, current_idx in enumerate(regular_stations):
        if len(filtered) >= target_count:
            break  # Already have enough stations
        
        # Calculate how many stations we still need (including end)
        remaining_needed = target_count - len(filtered) - (1 if end_idx not in filtered else 0)
        remaining_candidates = len(regular_stations) - i - 1  # Stations after current
        
        # Check spacing to last added station
        last_idx = filtered[-1] if filtered else None
        if last_idx is not None:
            dist = points_m.geometry.iloc[last_idx].distance(points_m.geometry.iloc[current_idx])
        else:
            dist = float('inf')  # No previous station, always add first one
        
        if dist >= min_spacing_m:
            # Good spacing, add it
            filtered.append(current_idx)
        elif remaining_needed > remaining_candidates:
            # We need this station to reach target count (not enough remaining candidates)
            # Accept it even if spacing is violated
            filtered.append(current_idx)
        elif dist >= min_spacing_m * 0.6:
            # Moderate spacing violation, but acceptable if we need stations
            if remaining_needed > 0:
                filtered.append(current_idx)
        # else: skip (too close and we have alternatives)
    
    # Always include end
    if end_idx in ordered_indices and end_idx not in filtered:
        # Check spacing to last station
        if filtered:
            last_idx = filtered[-1]
            dist_to_end = points_m.geometry.iloc[last_idx].distance(points_m.geometry.iloc[end_idx])
            # Always add end, but warn if spacing is very tight
            if dist_to_end < min_spacing_m * 0.5 and len(filtered) >= target_count:
                # If we're at target and end is very close, consider removing last regular station
                if len(filtered) > 1 and filtered[-1] not in (start_idx, end_idx):
                    filtered.pop()
        filtered.append(end_idx)
    
    # If we still don't have enough stations, fill from remaining candidates
    if len(filtered) < target_count:
        remaining = [idx for idx in ordered_indices if idx not in filtered]
        needed = target_count - len(filtered)
        # Add remaining stations in order (spacing may be violated, but count is priority)
        for idx in remaining[:needed]:
            filtered.append(idx)
    
    # Ensure both start and end are present
    if start_idx not in filtered:
        filtered.insert(0, start_idx)
    if end_idx not in filtered:
        filtered.append(end_idx)
    
    return filtered

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
    Run ACO station optimization for the specified algorithm method.
    Supports: "jps", "dijkstra", "astar"
    This function is intended to be imported and called from other code.
    """
    method_upper = method.upper() if method == "astar" else method.capitalize()
    print(f"=== 🧠 RUNNING ACO + {method_upper.upper()} INTEGRATED OPTIMIZATION ===")
    aco_total_start = time.perf_counter()

    interactive_wait_s = 0.0

    base_fw_dir = Path("data/outputs/floyd_warshall")
    points_path_full = base_fw_dir / f"fw_{method}_points.geojson"
    D_path = base_fw_dir / f"fw_{method}_D.npy"
    FW_path = base_fw_dir / f"fw_{method}_FW.npy"
    route_path = Path(f"data/outputs/{method}_path.geojson")
    method_slug = method.lower()
    detour_path = Path(f"data/outputs/aco/{method_slug}_detour_debug.geojson")

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
        print(f"🧭 Using {detour_path.name} as authoritative filter…")
        detour_ll = gpd.read_file(detour_path).to_crs("EPSG:4326")
        if "detour_label" not in detour_ll.columns:
            raise RuntimeError(f"❌ {detour_path.name} missing 'detour_label' column.")

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
            print(f"ℹ️ {detour_path.name} has no 'fw_index' → snapping detour to FW nodes (≤ 50 m)…")
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
        print(f"⚠️ {detour_path.name} not found — using all FW nodes.")
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
    wait_start = time.perf_counter()
    try:
        user_input = input(
            f"Enter number of stations (including start/end) [default={n_stations}]: "
        ).strip()
    except Exception:
        interactive_wait_s += time.perf_counter() - wait_start
        print(f"⚠️ Invalid input, using default station count ({n_stations}).")
    else:
        interactive_wait_s += time.perf_counter() - wait_start
        if user_input:
            try:
                n_stations = int(user_input)
                print(f"✅ Station count manually set to {n_stations}.")
            except Exception:
                print(f"⚠️ Invalid input, using default station count ({n_stations}).")
        else:
            print(f"ℹ️ Using default station count: {n_stations}")

    # 8) ACO params
    corridor_line_metric = _route_line_from_geojson(route_gdf.to_crs("EPSG:3857"))
    corridor_len_m = corridor_line_metric.length if corridor_line_metric is not None else None
    if corridor_len_m and n_stations > 1:
        target_spacing = corridor_len_m / (n_stations - 1)
        spacing_min = max(250.0, target_spacing * 0.85)
        spacing_max = target_spacing * 1.15
    else:
        spacing_min = 500.0
        spacing_max = 800.0

    aco_params = {
        "n_ants": 50,
        "n_iterations": 100,
        "alpha": 1.0,
        "beta": 3.0,
        "rho": 0.5,
        "Q": 1.0,
        "ideal_spacing_min": spacing_min,
        "ideal_spacing_max": spacing_max,
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
        df_hist.to_csv(outdir / f"aco_{method}_convergence.csv", index=False)
        print(f"📈 Saved ACO convergence log → aco_{method}_convergence.csv")

        plt.figure()
        plt.plot(df_hist["iteration"], df_hist["best_cost"], label="Best Cost")
        if df_hist["mean_cost"].notna().any():
            plt.plot(df_hist["iteration"], df_hist["mean_cost"], "--", label="Mean Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title(f"ACO Convergence Over Iterations ({method_upper})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"aco_{method}_convergence.png", dpi=200)
        plt.close()
        print(f"📉 Saved ACO convergence plot → aco_{method}_convergence.png")

    # 11) Order along corridor and enforce minimum spacing
    line = _route_line_from_geojson(route_gdf.to_crs("EPSG:3857"))
    if line is None:
        raise RuntimeError(f"Could not extract LineString from {method}_path.geojson.")

    all_candidates = list(dict.fromkeys(best_route + [start_idx, end_idx]))
    ordered = _sort_indices_along_line(points_feas, line, all_candidates)

    # 🧩 FIX: If ACO only returned start/end, fill with evenly spaced nodes
    if len(ordered) < n_stations:
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

    # Enforce minimum spacing while maintaining exact station count
    min_spacing = aco_params.get("ideal_spacing_min", 250.0)
    ordered = _enforce_minimum_spacing(points_feas, ordered, min_spacing, start_idx, end_idx, n_stations)
    print(f"📏 After minimum spacing enforcement ({min_spacing:.0f} m, target={n_stations}): {len(ordered)} stations")

    best_route = _enforce_exact_k(ordered, n_stations, start_idx, end_idx)


    # 12) Export chosen stations
    chosen = points_feas.iloc[best_route].copy()
    chosen.loc[:, "sub_index"] = best_route  # index within feasible subproblem
    # Also carry back original FW index for traceability
    chosen.loc[:, "fw_index"] = [feas_orig_idx[i] for i in best_route]
    chosen.to_crs("EPSG:4326").to_file(outdir / f"aco_{method}_stations.geojson", driver="GeoJSON")
    print(f"✅ Saved chosen stations → aco_{method}_stations.geojson")

    # 14) Export 1 km buffers for verification
    print("🟢 Generating 1 km buffers for station verification…")
    pois_m = pois_gdf.to_crs("EPSG:3857")
    stations_m = chosen.to_crs("EPSG:3857")
    buffers, poi_counts, poi_scores, poi_cats = [], [], [], []
    station_top_poi_pairs = []
    for sub_idx, geom in zip(best_route, stations_m.geometry):
        buf = geom.buffer(1000)
        near = pois_m[pois_m.intersects(buf)]
        buffers.append(buf)
        poi_counts.append(int(len(near)))
        poi_scores.append(float(near["score"].sum()) if "score" in near.columns and len(near) else 0.0)
        cats = near["category"].value_counts().head(10).to_dict() if "category" in near.columns and len(near) else {}
        poi_cats.append(str(cats))

        if len(near) and "NormalizedScore" in near.columns:
            near = near.copy()
            near["__norm_score__"] = pd.to_numeric(near["NormalizedScore"], errors="coerce").fillna(0.0)

            id_field = "@id" if "@id" in near.columns else ("id" if "id" in near.columns else None)
            if id_field:
                near["_geom_priority"] = near.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                near = (
                    near.sort_values(
                        by=[id_field, "_geom_priority", "__norm_score__"],
                        ascending=[True, False, False],
                    )
                    .drop_duplicates(subset=[id_field], keep="first")
                    .drop(columns="_geom_priority")
                    .reset_index(drop=True)
                )

            top_rows = near.sort_values("__norm_score__", ascending=False).head(10)
            if not top_rows.empty:
                name_series = None
                for col in ["name", "name_left", "name_right"]:
                    if col in top_rows.columns:
                        name_series = top_rows[col] if name_series is None else name_series.fillna(top_rows[col])

                if name_series is None:
                    # No name columns found -> no labels
                    formatted = ""
                else:
                    # Normalize and strip strings
                    labels = name_series.fillna("").astype(str).str.strip()

                    # Skip blanks and "Unnamed*" entries in favor of real names
                    filtered_pairs = []
                    for label, score in zip(labels, top_rows["__norm_score__"]):
                        if not label:
                            continue
                        # skip common autogenerated "Unnamed" labels (case-insensitive)
                        if label.lower().startswith("unnamed"):
                            continue
                        filtered_pairs.append((label, score))

                    # If no valid named entries after filtering, fall back to any non-blank labels
                    if not filtered_pairs:
                        for label, score in zip(labels, top_rows["__norm_score__"]):
                            if label:
                                filtered_pairs.append((label, score))
                    formatted = "; ".join(f"{lbl} ({sc:.3f})" for lbl, sc in filtered_pairs)
            else:
                formatted = ""
        else:
            formatted = "; ".join(node_stats.get(str(sub_idx), {}).get("top_pois", []))
        station_top_poi_pairs.append((sub_idx, formatted))

    buffers_gdf = gpd.GeoDataFrame(
        {"sub_index": best_route,
         "fw_index": [feas_orig_idx[i] for i in best_route],
         "poi_count": poi_counts,
         "poi_score_sum": poi_scores,
         "top_categories": poi_cats,
         "radius_m": 1000},
        geometry=buffers, crs="EPSG:3857"
    ).to_crs("EPSG:4326")
    buffers_gdf.to_file(outdir / f"aco_{method}_station_buffers.geojson", driver="GeoJSON")
    print(f"✅ Saved 1 km station buffers → aco_{method}_station_buffers.geojson")

    # Update chosen stations with top POI scorers (NormalizedScore-based)
    top_poi_map = {idx: label for idx, label in station_top_poi_pairs}
    chosen.loc[:, "top_pois"] = [top_poi_map.get(idx, "") for idx in best_route]

    # 15) Export all-nodes CSV (feasible-only universe = what ACO evaluated)
    all_nodes = points_feas.to_crs("EPSG:4326").copy()
    all_nodes.loc[:, "sub_index"] = np.arange(len(all_nodes))
    all_nodes.loc[:, "fw_index"] = feas_orig_idx
    chosen_set = set(best_route)
    all_nodes.loc[:, "poi_count"] = all_nodes["sub_index"].map(
        lambda i: node_stats.get(str(i), {}).get("count", 0)
    )
    all_nodes.loc[:, "poi_score"] = all_nodes["sub_index"].map(
        lambda i: node_stats.get(str(i), {}).get("score", 0)
    )
    all_nodes.loc[:, "poi_norm"] = all_nodes["sub_index"].map(
        lambda i: node_stats.get(str(i), {}).get("score_norm", 0)
    )
    all_nodes.loc[:, "top_pois"] = all_nodes["sub_index"].map(
        lambda i: "; ".join(node_stats.get(str(i), {}).get("top_pois", []))
    )
    if top_poi_map:
        mask = all_nodes["sub_index"].isin(top_poi_map.keys())
        all_nodes.loc[mask, "top_pois"] = all_nodes.loc[mask, "sub_index"].map(top_poi_map)
    all_nodes.loc[:, "lon"] = all_nodes.geometry.x
    all_nodes.loc[:, "lat"] = all_nodes.geometry.y
    all_nodes.drop(columns="geometry").to_csv(outdir / f"aco_{method}_candidate_stations.csv", index=False)
    print(f"🧾 Exported aco_{method}_candidate_stations.csv (Feasible Nodes)")

    # 16) Export chosen stations CSV (only the final selected stations)
    chosen_stations_df = all_nodes[all_nodes["sub_index"].isin(best_route)].copy()

    # -------------------------------------------------------------
    # >>> NEW FEATURE: Average poi_norm score (over 1, rounded 2dp)
    # -------------------------------------------------------------
    avg_norm = float(chosen_stations_df["poi_norm"].mean())
    formatted_score = f"{avg_norm:.2f}/1"
    chosen_stations_df.loc[:, "avg_poi_norm_score"] = formatted_score
    # -------------------------------------------------------------

    chosen_stations_df.drop(columns="geometry").to_csv(outdir / f"aco_{method}_stations_list.csv", index=False)
    print(f"🧾 Exported aco_{method}_stations_list.csv (Final Stations List)")

    aco_total_elapsed = max(0.0, time.perf_counter() - aco_total_start - interactive_wait_s)
    aco_total_ms = aco_total_elapsed * 1000.0
    print(f"[OK] ACO total runtime: {aco_total_ms:.2f} ms ({aco_total_elapsed:.2f} s)")

    # Return core results for callers
    return {
        "chosen_stations_gdf": chosen,
        "buffers_gdf": buffers_gdf,
        "all_nodes_df": all_nodes,
        "best_route": best_route,
        "summary": summary,
        "interactive_wait_s": interactive_wait_s,
        "compute_time_s": aco_total_elapsed,
    }


# ===================================================
# Prevent direct execution
# ===================================================
if __name__ == "__main__":
    raise RuntimeError(
        "This module is a helper and should not be executed directly. "
        "Use 'aco_jps_runner.py' instead, which calls run_aco_jps() internally."
    ) 