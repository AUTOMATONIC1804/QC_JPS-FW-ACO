# src/algorithms/aco/aco_astar_runner.py
"""
ACO A* Runner (Route Generator)
--------------------------------
Generates the *true traversable railway route* between ACO-selected
stations using A* Search restricted to the real road network
rasterized from fw_astar_roads.geojson.

Workflow:
  0) Automatically run ACO Station Runner to generate station nodes
  1) Load ACO-selected station nodes
  2) Load and rasterize fw_astar_roads.geojson → binary grid
  3) Convert all to EPSG:3857 (metric CRS)
  4) For each consecutive station pair:
       - Run A* Search on raster grid
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

# --- A* Core Imports ---
from src.algorithms.astar.astar_main import astar_search
from src.algorithms.jps.jps_grid import Grid
from src.algorithms.jps.grid_utils import load_clean_grid, cell_to_coords, coords_to_cell
from src.algorithms.astar.astar_utils import snap_to_nearest_road

# --- Import ACO Station Runner ---
from src.algorithms.aco.aco_station_runner import run_aco_jps

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
# Main Runner
# ===================================================
def run_aco_astar_route(
    method: str = "astar",
    stations_fp: str = None,
    roads_fp: str = None,
    tif_path="data/processed/qc_grid_clean.tif",
    output_dir="data/outputs/aco"
):
    """
    Run ACO station optimization + route generation for A*.
    """
    # Auto-construct paths if not provided
    if stations_fp is None:
        stations_fp = f"data/outputs/aco/aco_{method}_stations.geojson"
    if roads_fp is None:
        roads_fp = f"data/outputs/floyd_warshall/fw_{method}_roads.geojson"
    
    method_upper = "A*"
    
    runner_total_start = time.perf_counter()
    
    # -------------------------------
    # 🚉 Stage 0: Generate stations
    # -------------------------------
    print(f"=== 🚉 STAGE 1: RUNNING ACO {method_upper.upper()} STATION OPTIMIZATION ===")
    
    # 🔹 Automatically run the ACO station optimizer
    # (run_aco_jps handles user input for n_stations internally)
    stage1_start = time.perf_counter()
    stage1_result = run_aco_jps(n_stations=9, method=method)
    stage1_raw = time.perf_counter() - stage1_start
    stage1_compute_s = float(stage1_result.get("compute_time_s", stage1_raw)) if stage1_result else stage1_raw
    stage1_wait_s = float(stage1_result.get("interactive_wait_s", 0.0)) if stage1_result else 0.0
    if stage1_compute_s < 0:
        stage1_compute_s = max(0.0, stage1_raw - stage1_wait_s)
    print(f"[OK] Stage 1 runtime: {stage1_compute_s * 1000:.2f} ms ({stage1_compute_s:.2f} s)")
    print(f"\n✅ Stations generated successfully — proceeding to {method_upper} route generation...\n")

    # -------------------------------
    # 🚆 Stage 1: Route generation
    # -------------------------------
    print(f"=== 🚆 Stage 2: Running ACO {method_upper} Route Generation ===")
    stage2_start = time.perf_counter()

    # --- Setup paths ---
    stations_fp, roads_fp, tif_path = Path(stations_fp), Path(roads_fp), Path(tif_path)
    outdir = Path(output_dir) / f"aco_{method}_segments"
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

    # --- 3. Load and rasterize FW roads ---
    print(f"🛣️ Rasterizing FW {method_upper} roads to binary mask...")
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

        t0 = time.time()
        path = astar_search(grid, start_cell, goal_cell)
        t_ms = (time.time() - t0) * 1000
        total_time_ms += t_ms

        if not path:
            print(f"❌ A* failed for segment {i}-{i+1}")
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
        full_route.to_file(Path(output_dir) / f"aco_{method}_full_route.geojson", driver="GeoJSON")
        print(f"✅ Saved merged route → aco_{method}_full_route.geojson")
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
        df.to_csv(Path(output_dir) / f"aco_{method}_route_report.csv", index=False)
        print(f"🧾 Saved route report → aco_{method}_route_report.csv")

    now = time.perf_counter()
    stage2_elapsed = now - stage2_start
    total_compute_s = stage1_compute_s + stage2_elapsed

    print(f"\n=== ✅ ACO {method_upper} Route Generation Complete ===")
    print(f"📏 Total distance: {total_len:.2f} m")
    print(f"[OK] Stage 1 runtime: {stage1_compute_s * 1000:.2f} ms ({stage1_compute_s:.2f} s)")
    print(f"[OK] Stage 2 runtime: {stage2_elapsed * 1000:.2f} ms ({stage2_elapsed:.2f} s)")
    print(f"[OK] ACO total runtime (Stage 1 + Stage 2): {total_compute_s * 1000:.2f} ms ({total_compute_s:.2f} s)")
    print("=============================================")


# ===================================================
# Entry Point
# ===================================================
if __name__ == "__main__":
    run_aco_astar_route()
