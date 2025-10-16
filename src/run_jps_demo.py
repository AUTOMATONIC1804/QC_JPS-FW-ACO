"""
src/run_jps_demo.py
Run a demo of Jump Point Search (JPS) on either the QC grid or a toy grid.
Now supports:
- Manual start/goal via lon/lat (auto-projected to EPSG:3857).
- Snap-to-nearest road if clicked on obstacle.
- Always exports start & goal to PNG and GeoJSON (even if no path found).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import random
from collections import deque
from pyproj import Transformer
from shapely.geometry import Point, LineString, mapping

from src.algorithms.jps.jps_grid import Grid
from src.algorithms.jps.jps_main import jump_point_search
from src.algorithms.jps.jps_heuristics import octile
from src.algorithms.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell


def pick_random_points(grid, n=2, seed=None):
    """Pick n random passable cells (value=1)."""
    rows, cols = np.where(grid.matrix == 1)
    indices = list(zip(rows, cols))
    rng = random.Random(seed)
    return rng.sample(indices, n)


def snap_to_nearest_road(grid, start_cell, max_radius=50):
    """
    If start_cell is obstacle, search outward until we find nearest road cell (value=1).
    Uses BFS within a given radius.
    """
    r0, c0 = start_cell
    if grid.matrix[r0, c0] == 1:
        return start_cell  # already on road

    rows, cols = grid.matrix.shape
    visited = set()
    q = deque([(r0, c0, 0)])  # (row, col, dist)

    while q:
        r, c, d = q.popleft()
        if d > max_radius:
            break
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if 0 <= r < rows and 0 <= c < cols and grid.matrix[r, c] == 1:
            return (r, c)

        # Expand 8-neighbors
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < rows and 0 <= cc < cols:
                q.append((rr, cc, d+1))

    raise ValueError(f"No road cell found near {start_cell} within radius {max_radius}")


def run_demo(use_qc=True, manual=False, start_coords=None, goal_coords=None):
    if use_qc:
        # Load QC grid + metadata
        grid_arr, transform, crs = load_clean_grid(
            tif_path="data/processed/qc_grid_clean.tif",
            preview_png="data/outputs/qc_grid_preview.png",
            preview_geojson="data/outputs/qc_grid_preview.geojson"
        )
        grid = Grid(grid_arr)

        if manual and start_coords and goal_coords:
            # Reproject lon/lat → EPSG:3857 (to match raster CRS)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            sx, sy = transformer.transform(*start_coords)
            gx, gy = transformer.transform(*goal_coords)

            # Convert to grid cells
            start = coords_to_cell(sx, sy, transform)
            goal = coords_to_cell(gx, gy, transform)

            # Snap to nearest road if obstacle
            start = snap_to_nearest_road(grid, start)
            goal = snap_to_nearest_road(grid, goal)

            print(f"🎯 Manual Start (snapped): {start} from {start_coords}")
            print(f"🎯 Manual Goal  (snapped): {goal} from {goal_coords}")
        else:
            # Pick random start & goal
            start, goal = pick_random_points(grid)
            print(f"🎲 Random Start: {start}, Goal: {goal}")

    else:
        # Toy grid
        grid_arr = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ])
        grid = Grid(grid_arr)
        start, goal = (0, 0), (4, 4)
        transform, crs = None, None

    # Run JPS
    path = jump_point_search(grid, start, goal, heuristic=octile)

    # --- Visualization ---
    plt.figure(figsize=(8, 10))
    plt.imshow(grid.matrix, cmap="gray", interpolation="none")
    plt.scatter(start[1], start[0], color="lime", s=100, label="Start", edgecolors="black", zorder=3)
    plt.scatter(goal[1], goal[0], color="red", s=100, label="Goal", marker="x", zorder=3)

    if path:
        print(f"✅ Path found with {len(path)} steps")
        path_rows, path_cols = zip(*path)
        plt.plot(path_cols, path_rows, color="yellow", linewidth=2, label="JPS Path", zorder=2)
    else:
        print("❌ No path found!")
        plt.title("JPS Demo (No Path Found)")

    plt.legend(facecolor="white", framealpha=0.8, loc="upper right", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("data/outputs/jps_demo.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("[OK] Saved PNG visualization → data/outputs/jps_demo.png")

    # --- GeoJSON export ---
    if use_qc and transform:
        features = []

        # Add start/goal as points
        start_coords_real = cell_to_coords(start[0], start[1], transform)
        goal_coords_real = cell_to_coords(goal[0], goal[1], transform)
        features.append({
            "type": "Feature",
            "geometry": mapping(Point(start_coords_real)),
            "properties": {"role": "start"}
        })
        features.append({
            "type": "Feature",
            "geometry": mapping(Point(goal_coords_real)),
            "properties": {"role": "goal"}
        })

        # Add path if it exists
        if path:
            coords = [cell_to_coords(r, c, transform) for r, c in path]
            features.append({
                "type": "Feature",
                "geometry": mapping(LineString(coords)),
                "properties": {"algorithm": "JPS"}
            })

        geojson = {"type": "FeatureCollection", "features": features}
        with open("data/outputs/jps_demo.geojson", "w") as f:
            json.dump(geojson, f)
        print("[OK] Saved GeoJSON (with start/goal) → data/outputs/jps_demo.geojson")


if __name__ == "__main__":
    # Example 1: Random points
    # run_demo(use_qc=True, manual=False)

    # Example 2: Manual with lon/lat coords (from QGIS)
    run_demo(
        use_qc=True,
        manual=True,
        start_coords=(121.0596, 14.7324),   # lon, lat
        goal_coords=(121.080857, 14.59297)  # lon, lat
    )
