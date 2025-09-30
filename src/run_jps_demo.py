"""
Run a demo of Jump Point Search (JPS) on a grid.

Steps:
1. Load a test grid (either toy or QC grid from QGIS).
2. Pick start/end points.
3. Run JPS (from jps_main).
4. Save results as PNG + optional GeoJSON for QGIS.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from src.jps.jps_grid import Grid
from src.jps.jps_main import jump_point_search
from src.jps.jps_heuristics import octile
from src.jps.grid_utils import load_clean_grid, cell_to_coords  # utils we made earlier


def run_demo(use_qc=True):
    if use_qc:
        # Load Quezon City cleaned grid
        grid_arr, transform, crs = load_clean_grid("data/inputs/processed/qc_grid_clean.tif")
        grid = Grid(grid_arr)

        # Pick start/goal (row, col) -> must be on passable cells (value=1)
        start = (100, 150)
        goal = (300, 400)
    else:
        # Toy example grid (5x5, with a wall in the middle)
        grid_arr = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ])
        grid = Grid(grid_arr)
        start = (0, 0)
        goal = (4, 4)
        transform = None
        crs = None

    # Run JPS
    path = jump_point_search(grid, start, goal, heuristic=octile)

    if path is None:
        print("❌ No path found!")
        return

    print(f"✅ Path found with {len(path)} steps")

    # Plot result
    plt.imshow(grid.matrix, cmap="gray", interpolation="none")
    path_rows, path_cols = zip(*path)
    plt.plot(path_cols, path_rows, color="yellow", linewidth=2, label="JPS Path")
    plt.scatter(start[1], start[0], color="green", marker="o", label="Start")
    plt.scatter(goal[1], goal[0], color="red", marker="x", label="Goal")
    plt.legend()
    plt.title("JPS Demo")
    plt.savefig("data/outputs/jps_demo.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved PNG visualization to data/outputs/jps_demo.png")

    # Save GeoJSON if real QC grid
    if use_qc and transform:
        coords = [cell_to_coords(r, c, transform) for r, c in path]
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"algorithm": "JPS"},
                }
            ],
        }
        with open("data/outputs/jps_demo.geojson", "w") as f:
            json.dump(geojson, f)
        print("[OK] Saved GeoJSON path to data/outputs/jps_demo.geojson")


if __name__ == "__main__":
    run_demo(use_qc=True)  # start with toy example; set to True for QC grid
