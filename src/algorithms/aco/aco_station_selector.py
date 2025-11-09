# src/algorithms/aco/aco_station_selector.py
"""
ACO Station Selector
--------------------
Integrates all ACO components for station optimization.

Workflow:
  1) Accept pre-sliced effort matrix E and node_stats (feasible-only if runner filtered).
  2) (Optional) Enforce detour feasibility again if `detour_label` is present on points_gdf.
  3) Run ACO with spacing + coverage + POI + alignment penalties.
  4) Return selected station indices and diagnostic stats.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from src.algorithms.aco.aco_core import AntColony, ACOConfig
from src.algorithms.aco.aco_coverage_utils import (
    compute_coverage_ratio,
    coverage_penalty,
    spacing_penalty,
)


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

    # -------------------- config --------------------
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

    # Alignment penalty weights (meters → cost addend via multipliers)
    # Mild by default; tune as needed.
    align_cfg = params.get("align", {})
    lambda_back = float(align_cfg.get("lambda_back", 0.001))      # cost per meter of backtracking
    lambda_over = float(align_cfg.get("lambda_over", 0.002))      # cost per meter of overshoot
    proj_tol_m = float(align_cfg.get("proj_tol_m", 10.0))         # tolerance for overshoot

    # -------------------- feasibility guard (detour_label) --------------------
    # If the runner already filtered to feasible-only nodes, this is a no-op.
    # We keep it here as a hard guardrail.
    feasible_mask = np.ones(len(points_gdf), dtype=bool)
    if "detour_label" in points_gdf.columns:
        lab = (
            points_gdf["detour_label"]
            .astype(str)
            .str.lower()
            .isin(["feasible", "true", "1"])
            .values
        )
        feasible_mask &= lab

    # Always force start/end to be feasible
    feasible_mask[start_idx] = True
    feasible_mask[end_idx] = True

    # -------------------- precompute geometry/projections --------------------
    points_m = points_gdf.to_crs("EPSG:3857")
    corridor_m = corridor_gdf.to_crs("EPSG:3857")
    if len(corridor_m) == 0:
        raise RuntimeError("corridor_gdf is empty.")
    line: LineString = corridor_m.geometry.iloc[0]  # the JPS path LineString

    # Projection of each candidate onto the corridor (meters along-line)
    s_proj = np.array([line.project(geom) for geom in points_m.geometry], dtype=float)
    s_start = float(s_proj[start_idx])
    s_end = float(s_proj[end_idx])
    s_min, s_max = (min(s_start, s_end), max(s_start, s_end))

    # -------------------- cost function --------------------
    def route_cost_fn(route: List[int]) -> Tuple[float, bool]:
        """Composite cost for an ant route."""
        if not route:
            return math.inf, False

        # Hard feasibility: any non-feasible node kills the route
        for r in route:
            if not feasible_mask[r]:
                return math.inf, False

        # Base effort on E (sum of edges; invalid edges penalized)
        total_effort = 0.0
        invalid_edges = 0
        for u, v in zip(route[:-1], route[1:]):
            c = E[u, v]
            if not np.isfinite(c):
                total_effort += 1e6  # heavy penalty for invalid links
                invalid_edges += 1
            else:
                total_effort += float(c)

        if invalid_edges > len(route) * 0.4:
            return math.inf, False

        # Spacing penalty (pairwise)
        spacing_vals = []
        for u, v in zip(route[:-1], route[1:]):
            duv = points_m.geometry.iloc[u].distance(points_m.geometry.iloc[v])
            spacing_vals.append(spacing_penalty(duv, ideal_min, ideal_max, w=1.0))
        spacing_penalty_avg = float(np.mean(spacing_vals)) if spacing_vals else 1.0

        # Coverage penalty (buffers along the corridor)
        selected_points = points_gdf.iloc[route]
        coverage_ratio = float(compute_coverage_ratio(selected_points, corridor_m, buffer_radius))
        cov_pen = coverage_penalty(coverage_ratio, target_cov, w=1.0)

        # POI reward (higher poi_norm reduces cost)
        poi_vals = [node_stats.get(str(i), {}).get("score_norm", 0.0) for i in route]
        avg_poi = float(np.mean(poi_vals)) if poi_vals else 0.0

        # Alignment penalty: discourage big backtracks and overshooting the goal
        s_route = s_proj[route]
        diffs = np.diff(s_route)

        # Backtracking = negative diffs (in meters)
        back_m = float(np.sum(np.abs(diffs[diffs < 0]))) if diffs.size else 0.0

        # Overshoot = how far past the goal projection we end (measured against the “end” side)
        # Identify the corridor direction by comparing start/end projections.
        end_target = s_max  # we want to end within [s_min, s_max]
        last_proj = float(s_route[-1])
        overshoot_m = max(0.0, last_proj - end_target - proj_tol_m)

        align_add = lambda_back * back_m + lambda_over * overshoot_m

        # Combine multiplicatively for spacing/coverage/poi, add alignment
        total_cost = total_effort
        total_cost *= (1 + w_spacing * (spacing_penalty_avg - 1))
        total_cost *= (1 + w_coverage * (cov_pen - 1))
        total_cost *= (1 - w_poi * avg_poi)
        total_cost += align_add

        return float(total_cost), True

    # -------------------- run ACO --------------------
    colony = AntColony(E, cfg)
    best_route, best_cost, history = colony.run(route_cost_fn=route_cost_fn)

    # -------------------- report --------------------
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
        "num_stations": int(len(best_route)),
        "coverage_ratio": float(coverage_ratio),
        "spacing_mean_m": spacing_stats["mean"],
        "spacing_min_m": spacing_stats["min"],
        "spacing_max_m": spacing_stats["max"],
        "avg_poi_norm": float(np.mean(valid_poi_scores) if valid_poi_scores else 0.0),
        "history": history,  # <-- for convergence CSV/plot
    }

    return best_route, summary


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _route_spacing_stats(points_gdf: gpd.GeoDataFrame) -> dict:
    """Compute inter-station spacing along a route (meters)."""
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs("EPSG:4326")
    points_m = points_gdf.to_crs("EPSG:3857")

    dists = []
    for i in range(len(points_m) - 1):
        a = points_m.geometry.iloc[i]
        b = points_m.geometry.iloc[i + 1]
        dists.append(a.distance(b))

    if not dists:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "all": dists,
    }
