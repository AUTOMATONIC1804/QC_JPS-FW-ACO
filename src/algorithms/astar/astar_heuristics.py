"""
src/algorithms/astar/astar_heuristics.py
Heuristic functions for A*.
Currently uses octile distance (ideal for grids allowing diagonal movement).
"""

from math import sqrt


def octile_distance(a, b):
    """Compute octile distance (diagonal-allowed)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    F = sqrt(2) - 1
    return F * min(dx, dy) + max(dx, dy)
