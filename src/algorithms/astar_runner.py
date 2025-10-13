"""
src/algorithms/astar_runner.py
A* algorithm on QC grid (based on JPS runner).
References:
- GeeksforGeeks: https://www.geeksforgeeks.org/dsa/a-search-algorithm/
- DataCamp: https://www.datacamp.com/tutorial/a-star-algorithm
"""

import heapq
import json
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import deque
from pyproj import Transformer
from shapely.geometry import Point, LineString, mapping

from src.jps.jps_grid import Grid
from src.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell
from src.algorithms.metrics_utils import measure_runtime, compute_path_length


# -------------------------
# Supporting Functions
# -------------------------
def snap_to_nearest_road(grid, start_cell, max_radius=50):
    """If start_cell is obstacle, find nearest road (value=1) via BFS."""
    r0, c0 = start_cell
    if grid.matrix[r0, c0] == 1:
        return start_cell

    rows, cols = grid.matrix.shape
    visited = set()
    q = deque([(r0, c0, 0)])
    while q:
        r, c, d = q.popleft()
        if d > max_radius:
            break
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if 0 <= r < rows and 0 <= c < cols and grid.matrix[r, c] == 1:
            return (r, c)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < rows and 0 <= cc < cols:
                q.append((rr, cc, d+1))
    raise ValueError(f"No nearby road within {max_radius} cells")


def octile_distance(a, b):
    """Octile distance heuristic (diagonal-allowed)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    F = sqrt(2) - 1
    return F * min(dx, dy) + max(dx, dy)


def astar_search(grid, start, goal):
    """Perform A* search on grid with 8-directional movement."""
    rows, cols = grid.matrix.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: octile_distance(start, goal)}

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid.matrix[neighbor[0], neighbor[1]] == 0:
                    continue  # obstacle
                tentative_g = g_score[current] + (sqrt(2) if dr != 0 and dc != 0 else 1)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + octile_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path


# -------------------------
# Main Benchmark Function
# -------------------------
def run_astar_benchmark(
    tif_path="data/processed/qc_grid_clean.tif",
    start_coords=(121.0596, 14.7324),
    goal_coords=(121.080857, 14.59297),
    output_dir="data/outputs"
):
    """Run A* on QC grid and return metrics."""
    print("🚀 Running A* on QC grid...")

    # Load grid
    grid_arr, transform, crs = load_clean_grid(
        tif_path=tif_path,
        preview_png=f"{output_dir}/grid_preview.png",
        preview_geojson=f"{output_dir}/grid_preview.geojson",
    )
    grid = Grid(grid_arr)

    # Convert lon/lat → EPSG:3857 grid cells
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    sx, sy = transformer.transform(*start_coords)
    gx, gy = transformer.transform(*goal_coords)
    start = coords_to_cell(sx, sy, transform)
    goal = coords_to_cell(gx, gy, transform)

    start = snap_to_nearest_road(grid, start)
    goal = snap_to_nearest_road(grid, goal)
    print(f"🎯 Start: {start}, Goal: {goal}")

    # Run A* with runtime measurement
    path, runtime_ms = measure_runtime(astar_search, grid, start, goal)
    if not path:
        print("❌ No path found by A*.")
        return {"algorithm": "A*", "runtime_ms": None, "path_length_m": None, "steps": None}

    path_length_m = compute_path_length(path, transform)
    print(f"[OK] A* completed — Runtime: {runtime_ms:.2f} ms, Path length: {path_length_m:.2f} m")

    # --- Visualization (white bg, black roads, blue path) ---
    plt.figure(figsize=(10, 12), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    ax.imshow(grid.matrix, cmap="gray", interpolation="none", origin="upper")

    rows, cols = zip(*path)
    ax.plot(cols, rows, color="#007BFF", linewidth=2.8, label="A* Path", zorder=3)

    ax.scatter(start[1], start[0], s=120, facecolor="#4CAF50", edgecolors="black",
               linewidth=1.2, zorder=4, label="Start")
    ax.scatter(goal[1], goal[0], s=140, facecolor="red", marker="X",
               zorder=4, label="Goal")

    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_alpha(1.0)
    for text in leg.get_texts():
        text.set_color("black")

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{output_dir}/astar_path.png", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()

    # --- GeoJSON Export ---
    coords = [cell_to_coords(r, c, transform) for r, c in path]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(LineString(coords)), "properties": {"algorithm": "A*"}},
            {"type": "Feature", "geometry": mapping(Point(cell_to_coords(*start, transform))), "properties": {"role": "start"}},
            {"type": "Feature", "geometry": mapping(Point(cell_to_coords(*goal, transform))), "properties": {"role": "goal"}},
        ],
    }
    with open(f"{output_dir}/astar_path.geojson", "w") as f:
        json.dump(geojson, f)

    return {
        "algorithm": "A*",
        "runtime_ms": float(runtime_ms),
        "path_length_m": float(path_length_m),
        "steps": len(path),
    }
