"""
src/algorithms/astar_runner.py
A* algorithm on QC grid (based on JPS structure).
Now accepts (lat, lon) input for convenience when copying from QGIS.
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
    start_coords=(14.73545509,121.06668635),  
    goal_coords=(14.6522274,121.0477103),    
    output_dir="data/outputs"
):
    """Run A* on QC grid and return performance metrics (lat, lon input version)."""
    import os, traceback, geopandas as gpd, matplotlib.pyplot as plt
    from shapely.geometry import Point, LineString

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 🟦 Running A* Benchmark ===")
    try:
        # -------------------------------------------------------
        print("[1] Loading QC grid...")
        grid_arr, transform, crs = load_clean_grid(
            tif_path=tif_path,
            preview_png=f"{output_dir}/grid_preview.png",
            preview_geojson=f"{output_dir}/grid_preview.geojson",
        )
        grid = Grid(grid_arr)

        # -------------------------------------------------------
        print("[2] Preparing coordinates (EPSG:4326 → EPSG:3857)...")
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # 🔁 Swap (lat, lon) → (lon, lat) before transforming
        sx, sy = transformer.transform(start_coords[1], start_coords[0])
        gx, gy = transformer.transform(goal_coords[1], goal_coords[0])

        start = coords_to_cell(sx, sy, transform)
        goal = coords_to_cell(gx, gy, transform)
        start = snap_to_nearest_road(grid, start)
        goal = snap_to_nearest_road(grid, goal)

        print(f"   Start cell: {start}, Goal cell: {goal}")

        # -------------------------------------------------------
        print("[3] Running A* algorithm...")
        path, runtime_ms = measure_runtime(astar_search, grid, start, goal)
        if not path:
            print("❌ No path found by A*.")
            return {"algorithm": "A*", "runtime_ms": None, "path_length_m": None, "steps": None}

        path_length_m = compute_path_length(path, transform)
        print(f"[OK] Path found: {len(path)} steps, {path_length_m:.2f} m, {runtime_ms:.2f} ms")

        # -------------------------------------------------------
        print("[4] Rendering path visualization...")
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
        print(f"✅ Saved visualization → {output_dir}/astar_path.png")

        # -------------------------------------------------------
        print("[5] Building GeoJSON (EPSG:4326)...")
        coords = [cell_to_coords(r, c, transform) for r, c in path]
        line = LineString(coords)
        start_pt = Point(cell_to_coords(*start, transform))
        goal_pt = Point(cell_to_coords(*goal, transform))

        gdf = gpd.GeoDataFrame(
            {"role": ["path", "start", "goal"], "geometry": [line, start_pt, goal_pt]},
            crs="EPSG:3857",
        )

        # Convert to EPSG:4326 for GeoJSON export
        gdf.to_crs("EPSG:4326").to_file(f"{output_dir}/astar_path.geojson", driver="GeoJSON")
        print(f"✅ Saved route GeoJSON → {output_dir}/astar_path.geojson")

        # -------------------------------------------------------
        print("[6] Returning metrics summary...")
        return {
            "algorithm": "A*",
            "runtime_ms": float(runtime_ms),
            "path_length_m": float(path_length_m),
            "steps": len(path),
        }

    except Exception as e:
        print("\n❌ A* failed with error:")
        traceback.print_exc()
        print("❌ Error message:", e)
        return {"algorithm": "A*", "runtime_ms": None, "path_length_m": None, "steps": None}
