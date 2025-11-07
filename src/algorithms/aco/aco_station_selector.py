# src/algorithms/aco/aco_station_selector.py
"""
ACO Station Selector
--------------------
Integrates all ACO components for station optimization.

Workflow:
  1. Load effort matrix + POI node stats.
  2. Optionally build detour feasibility mask.
  3. Run Ant Colony Optimization using spacing + coverage + POI penalties.
  4. Return selected station indices and diagnostic stats.

References:
  Adapted from Korzeń & Kruszyna (2025) – modified effort-based ACO.
"""

from __future__ import annotations

import json
import math
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional

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
    """
    Run ACO to select optimal stations along a corridor.

    Args:
        E: effort matrix (n x n)
        node_stats: dict of POI stats per node (from aco_effort_matrix)
        points_gdf: GeoDataFrame of candidate station nodes
        corridor_gdf: GeoDataFrame of main route/corridor geometry
        start_idx: starting node index
        end_idx: ending node index
        params: dict containing:
            - n_ants, n_iterations, alpha, beta, rho, Q
            - ideal_spacing_min, ideal_spacing_max
            - buffer_radius, target_coverage
            - weights: dict of relative weights (spacing, coverage, poi)

    Returns:
        best_route (list of node indices)
        summary_stats (dict of key metrics)
    """
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
    # BUILD COST FUNCTION
    # -----------------------------------------------------------------
    points_m = points_gdf.to_crs("EPSG:3857")
    corridor_m = corridor_gdf.to_crs("EPSG:3857")

    def route_cost_fn(route: List[int]) -> Tuple[float, bool]:
        """
        Compute the composite cost for a given route (ant's path).
        """
        # --- 1. base effort cost (with inf-handling)
        total_effort = 0.0
        invalid_edges = 0

        for u, v in zip(route[:-1], route[1:]):
            c = E[u, v]  # ✅ define before using
            if not np.isfinite(c):
                # Heavy penalty for disconnected pairs
                total_effort += 1e6
                invalid_edges += 1
            else:
                total_effort += c

        # Reject routes that are mostly disconnected
        if invalid_edges > len(route) * 0.4:
            return math.inf, False

        # --- 2. spacing penalties
        spacing_vals = []
        for u, v in zip(route[:-1], route[1:]):
            pu = points_m.iloc[u].geometry
            pv = points_m.iloc[v].geometry
            dist = pu.distance(pv)
            spacing_vals.append(spacing_penalty(dist, ideal_min, ideal_max, w=1.0))
        spacing_penalty_avg = np.mean(spacing_vals) if spacing_vals else 1.0

        # --- 3. coverage ratio penalty
        selected_points = points_gdf.iloc[route]
        coverage_ratio = compute_coverage_ratio(selected_points, corridor_m, buffer_radius)
        coverage_pen = coverage_penalty(coverage_ratio, target_cov, w=1.0)

        # --- 4. POI reward (higher score = lower effective cost)
        poi_vals = [node_stats.get(str(i), {}).get("score_norm", 0.0) for i in route]
        avg_poi = np.mean(poi_vals) if poi_vals else 0.0

        # --- 5. Combine weighted total cost
        total_cost = total_effort
        total_cost *= (1 + w_spacing * (spacing_penalty_avg - 1))
        total_cost *= (1 + w_coverage * (coverage_pen - 1))
        total_cost *= (1 - w_poi * avg_poi)

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
        return [], {"best_cost": np.inf}

    print(f"[ACO] Best route cost: {best_cost:.4f}")
    print(f"[ACO] Best route length: {len(best_route)} stations")

    best_points = points_gdf.iloc[best_route]
    coverage_ratio = compute_coverage_ratio(best_points, corridor_gdf, buffer_radius)
    spacing_stats = _route_spacing_stats(best_points)

    summary = {
        "best_cost": best_cost,
        "num_stations": len(best_route),
        "coverage_ratio": coverage_ratio,
        "spacing_mean_m": spacing_stats["mean"],
        "spacing_min_m": spacing_stats["min"],
        "spacing_max_m": spacing_stats["max"],
        "avg_poi_norm": float(np.mean([node_stats[str(i)]["score_norm"] for i in best_route if str(i) in node_stats])),
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
        "all": dists
    }
