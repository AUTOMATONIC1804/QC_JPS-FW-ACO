"""
src/algorithms/jps/jps_heuristics.py
Heuristic functions for JPS.
"""

import math

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def octile(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (2**0.5 - 1) * min(dx, dy)
