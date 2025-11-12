"""
src/algorithms/jps_runner.py
Run the Jump Point Search (JPS) algorithm on the QC grid and collect metrics.
Now includes total runtime measurement and unified octile distance heuristic.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from pyproj import Transformer
from shapely.geometry import Point, LineString, mapping

from src.algorithms.jps.jps_grid import Grid
from src.algorithms.jps.jps_main import jump_point_search
from src.algorithms.jps.jps_heuristics import octile_distance  # ✅ unified heuristic
from src.algorithms.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell
from src.algorithms.metrics_utils import compute_path_length


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
    start_coords=None,
    goal_coords=None,
    output_dir="data/outputs"
):
    """Run JPS on QC grid and return metrics for comparison (lat, lon input version)."""
    import traceback, geopandas as gpd, os, matplotlib.pyplot as plt
    from shapely.geometry import Point, LineString

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 🟨 Running Jump Point Search (JPS) Benchmark ===")
    
    # Get user input for coordinates if not provided
    interactive_wait_s = 0.0
    if start_coords is None or goal_coords is None:
        wait_start = time.perf_counter()
        print("\n📍 Enter coordinates (lat, lon format):")
        try:
            start_input = input("Start coordinates (lat, lon) [default: 14.7327857, 121.0611778]: ").strip()
            if not start_input:
                start_coords = (14.7327857, 121.0611778)
            else:
                start_parts = [x.strip() for x in start_input.split(",")]
                if len(start_parts) != 2:
                    raise ValueError("Start coordinates must be in format: lat, lon")
                start_coords = (float(start_parts[0]), float(start_parts[1]))
            
            goal_input = input("Goal coordinates (lat, lon) [default: 14.656511, 121.031089]: ").strip()
            if not goal_input:
                goal_coords = (14.656511, 121.031089)
            else:
                goal_parts = [x.strip() for x in goal_input.split(",")]
                if len(goal_parts) != 2:
                    raise ValueError("Goal coordinates must be in format: lat, lon")
                goal_coords = (float(goal_parts[0]), float(goal_parts[1]))
            
            interactive_wait_s += time.perf_counter() - wait_start
            print(f"✅ Using coordinates: Start ({start_coords[0]}, {start_coords[1]}), Goal ({goal_coords[0]}, {goal_coords[1]})\n")
        except (ValueError, KeyboardInterrupt) as e:
            interactive_wait_s += time.perf_counter() - wait_start
            if isinstance(e, KeyboardInterrupt):
                raise
            print("⚠️ Invalid input, using default coordinates.")
            start_coords = (14.7327857, 121.0611778)
            goal_coords = (14.656511, 121.031089)
    
    total_start = time.perf_counter()  # ⏱️ Start total runtime timer

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
        sx, sy = transformer.transform(start_coords[1], start_coords[0])
        gx, gy = transformer.transform(goal_coords[1], goal_coords[0])
        start = coords_to_cell(sx, sy, transform)
        goal = coords_to_cell(gx, gy, transform)
        start = snap_to_nearest_road(grid, start)
        goal = snap_to_nearest_road(grid, goal)
        print(f"   Start cell: {start}, Goal cell: {goal}")

        # -------------------------------------------------------
        print("[3] Running Jump Point Search algorithm...")
        t0 = time.perf_counter()
        path = jump_point_search(grid, start, goal)
        runtime_ms = (time.perf_counter() - t0) * 1000
        if not path:
            print("❌ No path found by JPS.")
            total_runtime_ms = (time.perf_counter() - total_start - interactive_wait_s) * 1000
            return {
                "algorithm": "JPS",
                "runtime_ms": None,
                "total_runtime_ms": float(total_runtime_ms),
                "path_length_m": None,
                "steps": None,
            }

        path_length_m = compute_path_length(path, transform)
        print(f"[OK] Path found: {len(path)} steps, {path_length_m:.2f} m, {runtime_ms:.2f} ms")

        # -------------------------------------------------------
        print("[4] Rendering path visualization...")
        plt.figure(figsize=(10, 12), facecolor="white")
        ax = plt.gca()
        ax.set_facecolor("white")
        ax.imshow(grid.matrix, cmap="gray", interpolation="none", origin="upper")
        rows, cols = zip(*path)
        ax.plot(cols, rows, color="#FFD600", linewidth=2.8, label="JPS Path", zorder=3)
        ax.scatter(start[1], start[0], s=120, facecolor="#4CAF50", edgecolors="black", linewidth=1.2, zorder=4, label="Start")
        ax.scatter(goal[1], goal[0], s=140, facecolor="red", marker="X", zorder=4, label="Goal")

        leg = ax.legend(loc="upper right", frameon=True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        for text in leg.get_texts():
            text.set_color("black")

        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"{output_dir}/jps_path.png", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close()
        print(f"✅ Saved visualization → {output_dir}/jps_path.png")

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
        gdf.to_crs("EPSG:4326").to_file(f"{output_dir}/jps_path.geojson", driver="GeoJSON")
        print(f"✅ Saved route GeoJSON → {output_dir}/jps_path.geojson")

        # -------------------------------------------------------
        total_runtime_ms = (time.perf_counter() - total_start - interactive_wait_s) * 1000
        print(f"[OK] Total runtime (load → export): {total_runtime_ms:.2f} ms")

        return {
            "algorithm": "JPS",
            "runtime_ms": float(runtime_ms),
            "total_runtime_ms": float(total_runtime_ms),
            "path_length_m": float(path_length_m),
            "steps": len(path),
        }

    except Exception as e:
        print("\n❌ JPS failed with error:")
        traceback.print_exc()
        print("❌ Error message:", e)
        total_runtime_ms = (time.perf_counter() - total_start - interactive_wait_s) * 1000
        return {
            "algorithm": "JPS",
            "runtime_ms": None,
            "total_runtime_ms": float(total_runtime_ms),
            "path_length_m": None,
            "steps": None,
        }
