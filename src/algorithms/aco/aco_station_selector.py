# src/algorithms/aco/aco_station_selector.py
"""
ACO Station Selector
--------------------
Integrates all ACO components for station optimization.

Workflow:
  1. Load effort matrix + POI node stats.
  2. Apply detour feasibility adjustment (POI score boost/penalty).
  3. Run Ant Colony Optimization with spacing, coverage, POI, and corridor alignment constraints.
  4. Return selected station indices and diagnostics.
"""

from __future__ import annotations

import math
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from typing import Dict, List, Tuple

from src.algorithms.aco.aco_core import AntColony, ACOConfig
from src.algorithms.aco.aco_coverage_utils import (
    compute_coverage_ratio,
    coverage_penalty,
    spacing_penalty,
)

# ---------------------------------------------------------------------
# DETOUR-AWARE ADJUSTMENT
# ---------------------------------------------------------------------
def _compute_detour_features(points_gdf, corridor_gdf, detour_limit=300, rejoin_limit=500):
    """Compute off-corridor detour feasibility using JPS corridor geometry."""
    line = corridor_gdf.to_crs("EPSG:3857").geometry.iloc[0]
    pts = points_gdf.to_crs("EPSG:3857").copy()

    pts["d_path"] = pts.geometry.distance(line)
    pts["s_proj"] = pts.geometry.apply(lambda p: line.project(p))
    total_len = line.length

    feasible = []
    for _, row in pts.iterrows():
        if row["d_path"] <= detour_limit:
            feasible.append(True)
            continue

        # Try rejoining downstream within rejoin_limit
        p_proj = line.interpolate(min(row["s_proj"] + rejoin_limit, total_len))
        rejoin_dist = row.geometry.distance(p_proj)
        feasible.append(rejoin_dist <= rejoin_limit)

    pts["detour_feasible"] = feasible
    return pts


# ---------------------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------------------
def select_optimal_stations(
    E: np.ndarray,
    node_stats: Dict[str, Dict[str, float]],
    points_gdf: gpd.GeoDataFrame,
    corridor_gdf: gpd.GeoDataFrame,
    start_idx: int,
    end_idx: int,
    params: Dict,
) -> Tuple[List[int], Dict[str, float]]:
    """Run ACO to select optimal stations along a corridor."""
    print("\n[ACO] Starting station selection optimization...")

    # -----------------------------------------------------------------
    # CONFIG
    cfg = ACOConfig(
        n_ants=params.get("n_ants", 40),
        n_iterations=params.get("n_iterations", 100),
        alpha=params.get("alpha", 1.0),
        beta=params.get("beta", 3.0),
        rho=params.get("rho", 0.5),
        Q=params.get("Q", 1.0),
        start_idx=start_idx,
        end_idx=end_idx,
        allow_revisit=False,
        seed=params.get("seed", None),
    )

    ideal_min = params.get("ideal_spacing_min", 500)
    ideal_max = params.get("ideal_spacing_max", 800)
    buffer_radius = params.get("buffer_radius", 1000)
    target_cov = params.get("target_coverage", 0.9)

    w_spacing = params.get("weights", {}).get("spacing", 0.4)
    w_coverage = params.get("weights", {}).get("coverage", 0.3)
    w_poi = params.get("weights", {}).get("poi", 0.3)

    # -----------------------------------------------------------------
    # DETOUR FEASIBILITY + SCORE ADJUSTMENT
    # -----------------------------------------------------------------
    print("🧭 Evaluating off-corridor detour feasibility...")
    points_with_detours = _compute_detour_features(points_gdf, corridor_gdf)

    detour_ok_count = 0
    detour_no_count = 0

    for i, row in points_with_detours.iterrows():
        idx = str(i)
        if idx not in node_stats:
            continue

        detour_ok = bool(row.get("detour_feasible", False))
        d_path = float(row.get("d_path", 0))
        score_norm = node_stats[idx].get("score_norm", 0.0)

        if detour_ok:
            detour_ok_count += 1
            adj_factor = 1 + 0.25 * (1 - min(d_path, 300) / 300)
            node_stats[idx]["score_norm"] = score_norm * adj_factor
        else:
            detour_no_count += 1
            node_stats[idx]["score_norm"] = score_norm * 0.1

    print(f"✅ Detour-feasible nodes: {detour_ok_count} | ❌ Non-feasible: {detour_no_count}")

    # -----------------------------------------------------------------
    # BUILD COST FUNCTION
    # -----------------------------------------------------------------
    points_m = points_gdf.to_crs("EPSG:3857")
    corridor_m = corridor_gdf.to_crs("EPSG:3857")

    def route_cost_fn(route: List[int]) -> Tuple[float, bool]:
        """Composite cost for an ant’s route."""
        total_effort = 0.0
        invalid_edges = 0

        for u, v in zip(route[:-1], route[1:]):
            c = E[u, v]
            if not np.isfinite(c):
                total_effort += 1e6
                invalid_edges += 1
            else:
                total_effort += c

        if invalid_edges > len(route) * 0.4:
            return math.inf, False

        # ---------------------- SPACING ----------------------
        spacing_vals = []
        for u, v in zip(route[:-1], route[1:]):
            pu = points_m.iloc[u].geometry
            pv = points_m.iloc[v].geometry
            dist = pu.distance(pv)
            spacing_vals.append(spacing_penalty(dist, ideal_min, ideal_max, w=1.0))
        spacing_penalty_avg = np.mean(spacing_vals) if spacing_vals else 1.0

        # ---------------------- COVERAGE ----------------------
        selected_points = points_gdf.iloc[route]
        coverage_ratio = compute_coverage_ratio(selected_points, corridor_m, buffer_radius)
        coverage_pen = coverage_penalty(coverage_ratio, target_cov, w=1.0)

        # ---------------------- POI REWARD ----------------------
        poi_vals = [node_stats.get(str(i), {}).get("score_norm", 0.0) for i in route]
        avg_poi = np.mean(poi_vals) if poi_vals else 0.0

        # ---------------------- CORRIDOR ALIGNMENT ----------------------
        try:
            line = corridor_m.geometry.iloc[0]
            projections = [line.project(points_m.iloc[i].geometry) for i in route]
            diffs = np.diff(projections)
            line_len = line.length

            # measure backward and overshoot
            backward_excess = np.sum(np.abs(diffs[diffs < -50]))  # meters reversed
            overshoot_excess = max(0, projections[-1] - line_len)

            backward_ratio = backward_excess / max(line_len, 1)
            overshoot_ratio = overshoot_excess / max(line_len, 1)

            # strong penalty
            alignment_penalty = 1 + (1000 * backward_ratio) + (500 * overshoot_ratio)
            alignment_penalty = min(alignment_penalty, 1e6)

            # Debug (optional)
            if alignment_penalty > 5:
                print(f"[⚠️ Alignment] penalty={alignment_penalty:.2f}, backward={backward_excess:.1f}m, overshoot={overshoot_excess:.1f}m")

        except Exception:
            alignment_penalty = 1.0

        # ---------------------- COMBINE ----------------------
        total_cost = total_effort
        total_cost *= (1 + w_spacing * (spacing_penalty_avg - 1))
        total_cost *= (1 + w_coverage * (coverage_pen - 1))
        total_cost *= (1 - w_poi * avg_poi)
        total_cost *= alignment_penalty  # enforce corridor direction

        return float(total_cost), True

    # -----------------------------------------------------------------
    # RUN ACO
    # -----------------------------------------------------------------
    colony = AntColony(E, cfg)
    best_route, best_cost, history = colony.run(route_cost_fn=route_cost_fn)

    # -----------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------
    if not best_route:
        print("[ACO] No valid route found.")
        return [], {"best_cost": np.inf, "history": history}

    print(f"[ACO] Best route cost: {best_cost:.4f}")
    print(f"[ACO] Best route length: {len(best_route)} stations")

    best_points = points_gdf.iloc[best_route]
    coverage_ratio = compute_coverage_ratio(best_points, corridor_gdf, buffer_radius)
    spacing_stats = _route_spacing_stats(best_points)

    valid_poi_scores = [
        node_stats.get(str(i), {}).get("score_norm", 0.0)
        for i in best_route
        if str(i) in node_stats
    ]

    summary = {
        "best_cost": float(best_cost),
        "num_stations": len(best_route),
        "coverage_ratio": float(coverage_ratio),
        "spacing_mean_m": spacing_stats["mean"],
        "spacing_min_m": spacing_stats["min"],
        "spacing_max_m": spacing_stats["max"],
        "avg_poi_norm": float(np.mean(valid_poi_scores) if valid_poi_scores else 0.0),
        "history": history,
    }

    return best_route, summary


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _route_spacing_stats(points_gdf: gpd.GeoDataFrame) -> dict:
    """Compute inter-station spacing along a route."""
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs("EPSG:4326")
    points_m = points_gdf.to_crs("EPSG:3857")

    dists = []
    for i in range(len(points_m) - 1):
        a = points_m.iloc[i].geometry
        b = points_m.iloc[i + 1].geometry
        dists.append(a.distance(b))

    if not dists:
        return {"mean": 0, "min": 0, "max": 0}

    return {
        "mean": float(np.mean(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "all": dists,
    }
