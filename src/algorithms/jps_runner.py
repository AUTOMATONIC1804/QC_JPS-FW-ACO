"""
src/algorithms/jps_runner.py
Run the Jump Point Search (JPS) algorithm on the QC grid and collect metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pyproj import Transformer
from shapely.geometry import Point, LineString, mapping

from src.algorithms.jps.jps_grid import Grid
from src.algorithms.jps.jps_main import jump_point_search
from src.algorithms.jps.jps_heuristics import octile
from src.algorithms.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell
from src.algorithms.metrics_utils import measure_runtime, compute_path_length


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


def run_jps_benchmark(
    tif_path="data/processed/qc_grid_clean.tif",
    start_coords=(121.05153128195097, 14.652763030203197),
    goal_coords=(121.080857, 14.59297),
    output_dir="data/outputs"
):
    """Run JPS on QC grid and return metrics for comparison."""

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

    print(f"🚀 Running JPS from {start} → {goal}")

    # Run algorithm and measure runtime
    (path, runtime_ms) = measure_runtime(jump_point_search, grid, start, goal, octile)
    if not path:
        print("❌ No path found by JPS.")
        return {
            "algorithm": "JPS",
            "runtime_ms": None,
            "path_length_m": None,
            "steps": None,
        }

    # Compute path metrics
    path_length_m = compute_path_length(path, transform)
    print(f"[OK] JPS completed — Runtime: {runtime_ms:.2f} ms, Path length: {path_length_m:.2f} m")

    # --- Visualization ---

    plt.figure(figsize=(10, 12), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # show grid: black roads on white background
    ax.imshow(grid.matrix, cmap="gray", interpolation="none", origin="upper")

    # JPS path (yellow)
    rows, cols = zip(*path)
    ax.plot(cols, rows, color="#FFD600", linewidth=2.8, label="JPS Path", zorder=3)

    # Start / Goal markers
    ax.scatter(start[1], start[0], s=120, facecolor="#4CAF50", edgecolors="black",
        linewidth=1.2, zorder=4, label="Start")
    ax.scatter(goal[1], goal[0], s=140, facecolor="red", marker="X",
        zorder=4, label="Goal")

    # Legend: white background, black border, top-right
    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_alpha(1.0)
    for text in leg.get_texts():
        text.set_color("black")

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{output_dir}/jps_path.png", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()




    # --- GeoJSON Export ---
    coords = [cell_to_coords(r, c, transform) for r, c in path]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(LineString(coords)), "properties": {"algorithm": "JPS"}},
            {"type": "Feature", "geometry": mapping(Point(cell_to_coords(*start, transform))), "properties": {"role": "start"}},
            {"type": "Feature", "geometry": mapping(Point(cell_to_coords(*goal, transform))), "properties": {"role": "goal"}},
        ],
    }
    with open(f"{output_dir}/jps_path.geojson", "w") as f:
        json.dump(geojson, f)

    return {
        "algorithm": "JPS",
        "runtime_ms": float(runtime_ms),
        "path_length_m": float(path_length_m),
        "steps": len(path),
    }
