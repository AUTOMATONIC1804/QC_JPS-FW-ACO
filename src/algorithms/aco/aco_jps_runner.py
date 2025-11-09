# src/algorithms/aco/aco_jps_runner.py
"""
ACO JPS Runner (Route Generator)
--------------------------------
Generates the *true traversable railway route* between ACO-selected
stations using Jump Point Search (JPS) restricted to the real
road network rasterized from fw_jps_roads.geojson.

Workflow:
  0) Automatically run ACO Station Runner to generate station nodes
  1) Load ACO-selected station nodes
  2) Load and rasterize fw_jps_roads.geojson → binary grid
  3) Convert all to EPSG:3857 (metric CRS)
  4) For each consecutive station pair:
       - Run Jump Point Search (JPS) on raster grid
       - Save per-segment GeoJSON
  5) Merge all path segments into a final route
  6) Export route + report CSV
"""

import os, time, json
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString, Point
from pyproj import Transformer
import rasterio
from rasterio.transform import rowcol

# --- JPS Core Imports ---
from src.algorithms.jps.jps_main import jump_point_search
from src.algorithms.jps.jps_grid import Grid
from src.algorithms.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell

# --- Import ACO Station Runner ---
from src.algorithms.aco.aco_station_runner import run_aco_jps   # ✅ update: import station generator

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# ===================================================
# Raster adapter (for coordinate conversions)
# ===================================================
class GridAdapter:
    def __init__(self, tif_path: str):
        self.ds = rasterio.open(tif_path)
        self.transform = self.ds.transform
        self.crs = self.ds.crs.to_string() if self.ds.crs else WGS84
        self.array = self.ds.read(1)
        self.nodata = self.ds.nodata
        if self.crs != METRIC:
            self.to_raster_crs = Transformer.from_crs(METRIC, self.crs, always_xy=True)
        else:
            self.to_raster_crs = None

    def metric_to_rowcol(self, x_m, y_m):
        if self.to_raster_crs:
            x, y = self.to_raster_crs.transform(x_m, y_m)
        else:
            x, y = x_m, y_m
        r, c = rowcol(self.transform, x, y)
        return int(r), int(c)

    def is_in_bounds(self, r, c):
        return 0 <= r < self.array.shape[0] and 0 <= c < self.array.shape[1]

    def is_passable(self, r, c):
        if not self.is_in_bounds(r, c):
            return False
        v = self.array[r, c]
        if self.nodata is not None and v == self.nodata:
            return False
        return v > 0


# ===================================================
# Helper: Snap to nearest road cell (for safety)
# ===================================================
def snap_to_nearest_road(grid, start_cell, max_radius=50):
    from collections import deque
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
    raise ValueError(f"No nearby road found for {start_cell}")


# ===================================================
# Main Runner
# ===================================================
def run_aco_jps_route(
    stations_fp="data/outputs/aco/aco_jps_stations.geojson",
    roads_fp="data/outputs/floyd_warshall/fw_jps_roads.geojson",
    tif_path="data/processed/qc_grid_clean.tif",
    output_dir="data/outputs/aco"
):
    # -------------------------------
    # 🚉 Stage 0: Generate stations
    # -------------------------------
    print("=== 🚉 Stage 1: Running ACO Station Optimization ===")
    try:
        n_stations = int(input("Enter number of stations (including start/end) [default=9]: ") or 9)
    except ValueError:
        n_stations = 9

    # 🔹 Automatically run the ACO + JPS station optimizer
    print(f"🧠 Running ACO Station Runner with {n_stations} stations...")
    run_aco_jps(n_stations=n_stations, method="jps")
    print("\n✅ Stations generated successfully — proceeding to JPS route generation...\n")

    # -------------------------------
    # 🚆 Stage 1: Route generation
    # -------------------------------
    print("=== 🚆 Stage 2: Running ACO JPS Route Generation ===")

    # --- Setup paths ---
    stations_fp, roads_fp, tif_path = Path(stations_fp), Path(roads_fp), Path(tif_path)
    outdir = Path(output_dir) / "aco_jps_segments"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load station points ---
    if not stations_fp.exists():
        raise FileNotFoundError(f"Missing station file: {stations_fp}")

    stations = gpd.read_file(stations_fp).to_crs(METRIC)
    stations = stations.reset_index(drop=True)
    print(f"📍 Loaded {len(stations)} ACO-selected station nodes.")

    # --- 2. Load raster grid ---
    print("🗺️ Loading QC raster grid...")
    grid_arr, transform, crs = load_clean_grid(tif_path=str(tif_path))
    adapter = GridAdapter(str(tif_path))

    # --- 3. Load and rasterize fw_jps_roads.geojson ---
    print("🛣️ Rasterizing FW JPS roads to binary mask...")
    if not roads_fp.exists():
        raise FileNotFoundError(f"Missing {roads_fp}")

    roads_gdf = gpd.read_file(roads_fp)
    roads_gdf = roads_gdf.to_crs(adapter.crs)

    import rasterio.features
    road_mask = np.zeros_like(adapter.array, dtype=np.uint8)
    shapes = ((geom, 1) for geom in roads_gdf.geometry if not geom.is_empty)
    burned = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=road_mask.shape,
        transform=adapter.transform,
        fill=0,
        dtype=np.uint8
    )
    grid = Grid(burned)
    print(f"✅ Rasterized {len(roads_gdf)} road segments to grid mask.")

    # --- 4. Iterate over station pairs ---
    all_segments = []
    report = []
    total_len = 0.0
    total_time_ms = 0.0

    for i in range(len(stations) - 1):
        s_pt = stations.geometry.iloc[i]
        g_pt = stations.geometry.iloc[i + 1]
        seg_name = f"segment_{i}_{i+1}.geojson"
        seg_path = outdir / seg_name

        sx, sy = s_pt.x, s_pt.y
        gx, gy = g_pt.x, g_pt.y

        start_cell = adapter.metric_to_rowcol(sx, sy)
        goal_cell = adapter.metric_to_rowcol(gx, gy)

        start_cell = snap_to_nearest_road(grid, start_cell)
        goal_cell = snap_to_nearest_road(grid, goal_cell)

        print(f"▶️ Segment {i+1}/{len(stations)-1}: {start_cell} → {goal_cell}")

        t0 = time.perf_counter()
        path = jump_point_search(grid, start_cell, goal_cell)
        t_ms = (time.perf_counter() - t0) * 1000
        total_time_ms += t_ms

        if not path:
            print(f"❌ JPS failed for segment {i}-{i+1}")
            report.append({
                "segment": f"{i}-{i+1}",
                "distance_m": np.nan,
                "runtime_ms": round(t_ms, 2),
                "status": "failed"
            })
            continue

        coords = [cell_to_coords(r, c, transform) for r, c in path]
        line = LineString(coords)
        all_segments.append(line)
        seg_len = line.length
        total_len += seg_len

        # Save segment GeoJSON
        gdf_seg = gpd.GeoDataFrame(
            {"segment": [f"{i}-{i+1}"], "distance_m": [seg_len], "runtime_ms": [t_ms]},
            geometry=[line], crs="EPSG:3857"
        ).to_crs(WGS84)
        gdf_seg.to_file(seg_path, driver="GeoJSON")

        report.append({
            "segment": f"{i}-{i+1}",
            "distance_m": round(seg_len, 2),
            "runtime_ms": round(t_ms, 2),
            "status": "ok"
        })
        print(f"✅ Segment {i}-{i+1}: {seg_len:.2f} m ({t_ms:.1f} ms)")

    # --- 5. Merge all segments into one full route ---
    if all_segments:
        merged = LineString([pt for seg in all_segments for pt in seg.coords])
        full_route = gpd.GeoDataFrame(
            {"role": ["path"], "length_m": [merged.length]}, geometry=[merged], crs="EPSG:3857"
        ).to_crs(WGS84)
        full_route.to_file(Path(output_dir) / "aco_jps_full_route.geojson", driver="GeoJSON")
        print(f"✅ Saved merged route → aco_jps_full_route.geojson")
    else:
        print("⚠️ No valid segments to merge — route incomplete.")

    # --- 6. Save performance report ---
    if report:
        import pandas as pd
        df = pd.DataFrame(report)
        df.loc[len(df.index)] = {
            "segment": "TOTAL",
            "distance_m": round(total_len, 2),
            "runtime_ms": round(total_time_ms, 2),
            "status": "complete"
        }
        df.to_csv(Path(output_dir) / "aco_jps_route_report.csv", index=False)
        print(f"🧾 Saved route report → aco_jps_route_report.csv")

    print("\n=== ✅ ACO JPS Route Generation Complete ===")
    print(f"📏 Total distance: {total_len:.2f} m")
    print(f"⏱️ Total compute time: {total_time_ms:.2f} ms")
    print("=============================================")


# ===================================================
# Entry Point
# ===================================================
if __name__ == "__main__":
    run_aco_jps_route()
