"""
aco_jps_runner.py
--------------------------------------------------
Run integrated ACO (path + stations) on FW/JPS graph, with
intersection-aware detours and POI-aware scoring.

Inputs (FW/JPS variant):
  data/outputs/floyd_warshall/
    ├─ fw_jps_points.geojson   (FW nodes)
    ├─ fw_jps_D.npy            (FW distance matrix, meters)
    ├─ fw_jps_FW.npy           (FW next/parent matrix)
    └─ fw_jps_roads.geojson    (roads geometry; used for intersections)

  data/outputs/jps_path.geojson           (to auto-pick start/end)
  data/processed/qc_pois_final_scored.geojson (POIs with NormalizedScore)

Outputs (to data/outputs/aco/):
  ├─ aco_path_stations.geojson
  ├─ aco_path_route.geojson
  ├─ aco_intersections_debug.geojson
  ├─ aco_route_refined.geojson           (optional detour refinement)
  └─ aco_detours_debug.geojson           (optional detour segments)
"""

from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, LineString

from src.algorithms.aco.aco_path_station_optimizer import (
    ACOPathStationOptimizer, ACOPathParams
)
from src.algorithms.aco.aco_detour_refiner import (
    export_refined_route_with_debug, DetourConfig
)
from src.algorithms.aco.aco_config import ACO_CONFIG


def _nearest_fw_index(points_file: str, lon: float, lat: float) -> int:
    nodes = gpd.read_file(points_file)
    if nodes.crs is None:
        nodes.set_crs("EPSG:4326", inplace=True)
    nodes_m = nodes.to_crs(3857)
    p_m = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(3857).iloc[0]
    return int(nodes_m.distance(p_m).idxmin())


def _start_end_from_jps(points_file: str, jps_path_file: str):
    jps = gpd.read_file(jps_path_file)
    if jps.crs is None:
        jps.set_crs("EPSG:4326", inplace=True)
    line = jps.geometry.iloc[0]
    (lon_s, lat_s) = line.coords[0]
    (lon_e, lat_e) = line.coords[-1]
    s = _nearest_fw_index(points_file, lon_s, lat_s)
    e = _nearest_fw_index(points_file, lon_e, lat_e)
    print(f"🔍 Start FW idx: {s} | End FW idx: {e}")
    return s, e


def main():
    print("🚆 ACO Path+Station Optimization (FW/JPS, intersection-aware)")

    # --- Project root & standard paths (adjust base if needed) ---
    BASE = Path(r"D:\Quezon_City")
    FW_DIR = BASE / "data" / "outputs" / "floyd_warshall"

    POINTS_FILE = str(FW_DIR / "fw_jps_points.geojson")
    DIST_FILE   = str(FW_DIR / "fw_jps_D.npy")
    NEXT_FILE   = str(FW_DIR / "fw_jps_FW.npy")
    ROADS_FILE  = str(FW_DIR / "fw_jps_roads.geojson")   # intersections only
    JPS_PATH    = str(BASE / "data" / "outputs" / "jps_path.geojson")
    POI_FILE    = str(BASE / "data" / "processed" / "qc_pois_final_scored.geojson")
    OUTPUT_DIR  = str(BASE / "data" / "outputs" / "aco")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Params: merge config dict into dataclass safely ---
    valid = {name for name in ACOPathParams.__dataclass_fields__.keys()}
    kwargs = {k: v for k, v in ACO_CONFIG.items() if k in valid}
    params = ACOPathParams(**kwargs)

    # --- User input: station count k ---
    k = int(input("Enter desired number of stations (k): ").strip())

    # --- Start/End from JPS endpoints → nearest FW nodes ---
    start_idx, end_idx = _start_end_from_jps(POINTS_FILE, JPS_PATH)

    # --- Run intersection-aware ACO ---
    optimizer = ACOPathStationOptimizer(
        points_file=POINTS_FILE,
        dist_file=DIST_FILE,
        next_file=NEXT_FILE,
        roads_file=ROADS_FILE,
        poi_path=POI_FILE,
        params=params,
        start_idx=start_idx,
        end_idx=end_idx,
        k_target=k,
    )
    result = optimizer.run(OUTPUT_DIR)

    # --- Optional: refine final route by adding acceptable detours ---
    try:
        route_path = Path(OUTPUT_DIR) / "aco_path_route.geojson"
        route_gdf = gpd.read_file(route_path)
        route_line: LineString = route_gdf.geometry.iloc[0]
        export_refined_route_with_debug(
            main_path_line=route_line,
            roads_file=ROADS_FILE,
            pois_file=POI_FILE,
            output_dir=OUTPUT_DIR,
            cfg=DetourConfig(),  # use defaults; tweak in aco_config if desired
        )
    except Exception as e:
        print(f"⚠️ Skipped detour refinement due to error: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
