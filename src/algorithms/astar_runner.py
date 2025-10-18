"""
src/algorithms/astar_runner.py
A* algorithm on QC grid (based on JPS structure).
References:
- GeeksforGeeks: https://www.geeksforgeeks.org/dsa/a-search-algorithm/
- DataCamp: https://www.datacamp.com/tutorial/a-star-algorithm
"""

import json
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import Point, LineString, mapping

from src.algorithms.jps.jps_grid import Grid
from src.algorithms.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell
from src.algorithms.metrics_utils import measure_runtime, compute_path_length
from src.algorithms.astar.astar_main import astar_search
from src.algorithms.astar.astar_utils import snap_to_nearest_road


def run_astar_benchmark(
    tif_path="data/processed/qc_grid_clean.tif",
    start_coords=(121.00210794876448, 14.618161542775779),
    goal_coords=(121.03115460465105, 14.655409073297683),
    output_dir="data/outputs"
):
    """Run A* on QC grid and return performance metrics."""
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

    # Run A* with timing
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
