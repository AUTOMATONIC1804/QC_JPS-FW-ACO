"""
src/algorithms/jps/jps_heuristics.py
Unified heuristic functions for grid-based pathfinding.
Both JPS and A* use the same Octile distance heuristic for uniformity.
"""

from math import sqrt


def octile_distance(a, b):
    """
    Compute Octile distance — admissible heuristic for 8-directional movement.

    Parameters
    ----------
    a, b : tuple(int, int)
        Grid cell coordinates (row, col)

    Returns
    -------
    float
        Approximate distance between cells assuming:
        - Orthogonal moves cost = 1
        - Diagonal moves cost = √2
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    F = sqrt(2) - 1
    return F * min(dx, dy) + max(dx, dy)
