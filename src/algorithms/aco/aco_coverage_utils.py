# -*- coding: utf-8 -*-
"""
ACO Coverage & Spacing Utilities
--------------------------------
Utilities for evaluating spacing, coverage, and detour feasibility
for station-selection Ant Colony Optimization.

All operations are compatible with EPSG:4326 datasets but project
temporarily to EPSG:3857 (meters) for distance and buffer calculations.

Functions:
  - spacing_penalty(distance_m)
  - compute_spacing_stats(points_gdf)
  - compute_coverage_ratio(stations_gdf, corridor_gdf, buffer_radius=1000)
  - coverage_penalty(coverage_ratio)
  - build_detour_mask(points_gdf, corridor_gdf, buffer_radius=1000)
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString


# ---------------------------------------------------------------------
# SPACING UTILITIES
# ---------------------------------------------------------------------

def spacing_penalty(distance_m: float,
                    ideal_min: float = 500,
                    ideal_max: float = 800,
                    w: float = 0.5) -> float:
    """
    Return spacing penalty multiplier.
      - penalty = 1 when within ideal range
      - penalty > 1 when too close or too far
    """
    if np.isnan(distance_m):
        return 2.0  # heavy penalty if unknown

    if distance_m < ideal_min:
        penalty = 1 + w * ((ideal_min - distance_m) / ideal_min)
    elif distance_m > ideal_max:
        penalty = 1 + w * ((distance_m - ideal_max) / ideal_max)
    else:
        penalty = 1.0

    return float(max(penalty, 1.0))


def compute_spacing_stats(points_gdf: gpd.GeoDataFrame) -> dict:
    """
    Compute inter-point distances along the sequence of stations.
    Assumes points are ordered along the route (e.g., by route sequence).
    """
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs("EPSG:4326")
    points_m = points_gdf.to_crs("EPSG:3857")

    dists = []
    for i in range(len(points_m) - 1):
        a = points_m.iloc[i].geometry
        b = points_m.iloc[i + 1].geometry
        dists.append(a.distance(b))

    return {
        "mean": float(np.mean(dists)) if dists else 0.0,
        "min": float(np.min(dists)) if dists else 0.0,
        "max": float(np.max(dists)) if dists else 0.0,
        "all": dists
    }


# ---------------------------------------------------------------------
# COVERAGE UTILITIES
# ---------------------------------------------------------------------

def compute_coverage_ratio(stations_gdf: gpd.GeoDataFrame,
                           corridor_gdf: gpd.GeoDataFrame,
                           buffer_radius: float = 1000) -> float:
    """
    Compute the fraction of the corridor's total length covered by
    station buffers of given radius (meters).

    Returns: coverage_ratio (0..1)
    """
    if stations_gdf.crs is None:
        stations_gdf = stations_gdf.set_crs("EPSG:4326")
    if corridor_gdf.crs is None:
        corridor_gdf = corridor_gdf.set_crs("EPSG:4326")

    stations_m = stations_gdf.to_crs("EPSG:3857")
    corridor_m = corridor_gdf.to_crs("EPSG:3857")

    # dissolve all buffers into one geometry
    buffer_union = stations_m.buffer(buffer_radius).unary_union

    # intersection length between corridor and union of buffers
    corridor_line = corridor_m.unary_union
    if not isinstance(corridor_line, LineString):
        # if MultiLineString, merge
        corridor_line = LineString([pt for geom in corridor_m.geometry for pt in geom.coords])

    covered = corridor_line.intersection(buffer_union)
    coverage_len = 0.0
    if covered.is_empty:
        coverage_len = 0.0
    elif isinstance(covered, LineString):
        coverage_len = covered.length
    else:
        # MultiLineString or GeometryCollection
        coverage_len = sum(g.length for g in covered.geoms if hasattr(g, "length"))

    total_len = corridor_line.length
    ratio = coverage_len / total_len if total_len > 0 else 0.0

    return float(max(0.0, min(1.0, ratio)))


def coverage_penalty(coverage_ratio: float,
                     target: float = 0.9,
                     w: float = 0.3) -> float:
    """
    Returns a multiplier penalty based on coverage ratio.
    - If coverage >= target, penalty = 1.0
    - Otherwise, penalty = 1 + w * (1 - coverage/target)
    """
    if coverage_ratio >= target:
        return 1.0
    return 1.0 + w * (1 - coverage_ratio / target)


# ---------------------------------------------------------------------
# DETOUR UTILITIES
# ---------------------------------------------------------------------

def build_detour_mask(points_gdf: gpd.GeoDataFrame,
                      corridor_gdf: gpd.GeoDataFrame,
                      buffer_radius: float = 1000) -> np.ndarray:
    """
    Build a boolean mask (n x n) indicating feasible detours:
    True = node pair has a path that stays within corridor buffer.

    Simplified heuristic: both nodes must fall inside the same buffer zone
    around the corridor polygon.
    """
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs("EPSG:4326")
    if corridor_gdf.crs is None:
        corridor_gdf = corridor_gdf.set_crs("EPSG:4326")

    points_m = points_gdf.to_crs("EPSG:3857")
    corridor_m = corridor_gdf.to_crs("EPSG:3857")

    n = len(points_m)
    mask = np.zeros((n, n), dtype=bool)

    # precompute corridor buffer polygon
    corridor_buffer = corridor_m.buffer(buffer_radius).unary_union

    # mark feasible pairs
    for i in range(n):
        pi = points_m.iloc[i].geometry
        for j in range(n):
            pj = points_m.iloc[j].geometry
            if corridor_buffer.contains(pi) and corridor_buffer.contains(pj):
                mask[i, j] = True
            else:
                mask[i, j] = False

    return mask
