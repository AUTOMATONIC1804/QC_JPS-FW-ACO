"""
src/algorithms/metrics_utils.py
Utility functions for measuring runtime and path length in meters.
"""

import time
from shapely.geometry import LineString
from src.jps.grid_utils import cell_to_coords


def measure_runtime(func, *args, **kwargs):
    """Measure execution time (ms) of a pathfinding function."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return result, elapsed_ms


def compute_path_length(path, transform):
    """Compute total path length (in meters) from a list of grid cells."""
    if not path:
        return 0.0
    coords = [cell_to_coords(r, c, transform) for r, c in path]
    line = LineString(coords)
    return line.length
