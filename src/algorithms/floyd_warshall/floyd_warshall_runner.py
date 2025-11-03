"""
floyd_warshall_runner.py
Master runner for Floyd–Warshall (FW) pipeline across:
    - Jump Point Search (JPS)
    - A* Search (ASTAR)
    - Dijkstra
"""

import time, json
import pandas as pd
import numpy as np
from pathlib import Path

# The 3 Algorithms 
from src.algorithms.floyd_warshall.fw_jps_runner import run_fw_vector as run_jps
from src.algorithms.floyd_warshall.fw_astar_runner import run_fw_astar_vector as run_astar
from src.algorithms.floyd_warshall.fw_dijkstra_runner import run_fw_dijkstra as run_dijkstra


# Matrix info for the three algorithms
def get_matrix_info(output_dir: Path, prefix: str):
    D_path = output_dir / f"{prefix}_D.npy"
    FW_path = output_dir / f"{prefix}_FW.npy"
    pts_path = output_dir / f"{prefix}_points.geojson"
    info = {"n_points": None, "matrix_shape": None}
    try:
        if D_path.exists():
            D = np.load(D_path)
            info["matrix_shape"] = list(D.shape)
            info["n_points"] = D.shape[0]
        elif FW_path.exists():
            FW = np.load(FW_path)
            info["matrix_shape"] = list(FW.shape)
            info["n_points"] = FW.shape[0]
        elif pts_path.exists():
            import geopandas as gpd
            pts = gpd.read_file(pts_path)
            info["n_points"] = len(pts)
    except Exception as e:
        print(f"⚠️ Could not read matrix info for {prefix}: {e}")
    return info


def run_fw_pipelines(
    base_output_dir="data/outputs/floyd_warshall",
    jps_path="data/outputs/jps_path.geojson",
    astar_path="data/outputs/astar_path.geojson",
    dijkstra_path="data/outputs/dijkstra_path.geojson",
    roads_vector="data/processed/qc_roads_major_edges.geojson",
    graphml_path=r"D:\Quezon_City\data\processed\qc_roads_major.graphml",
    buffer_m=2000,
    spacing_m=600,
    merge_radius_m=450
):
    """
    Runs all three Floyd–Warshall pipelines (JPS, A*, Dijkstra)
    and compiles a unified report of their results.
    """
    out_dir = Path(base_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0_all = time.perf_counter()

    # JPS Metrics
    print("\n==============================")
    print("Jump Point Search + FW")
    print("==============================")
    t0 = time.perf_counter()
    run_jps(
        route_geojson=jps_path,
        roads_vector=roads_vector,
        buffer_m=buffer_m,
        spacing_m=spacing_m,
        merge_radius_m=merge_radius_m,
        output_dir=base_output_dir
    )
    t1 = (time.perf_counter() - t0) * 1000
    info_jps = get_matrix_info(out_dir, "fw_jps")
    results.append({
        "Algorithm": "JPS",
        "RouteFile": jps_path,
        "RoadSource": "Vector (GeoJSON)",
        "Buffer_m": buffer_m,
        "Spacing_m": spacing_m,
        "MergeRadius_m": merge_radius_m,
        "n_points": info_jps.get("n_points"),
        "MatrixShape": info_jps.get("matrix_shape"),
        "Time_ms": round(t1, 2)
    })
    print(f"JPS runtime: {t1:.2f} ms")

    # A* Metrics
    print("\n==============================")
    print("A* + FW")
    print("==============================")
    t0 = time.perf_counter()
    run_astar(
        route_geojson=astar_path,
        roads_vector=roads_vector,
        buffer_m=buffer_m,
        spacing_m=spacing_m,
        merge_radius_m=merge_radius_m,
        output_dir=base_output_dir
    )
    t1 = (time.perf_counter() - t0) * 1000
    info_astar = get_matrix_info(out_dir, "fw_astar")
    results.append({
        "Algorithm": "A*",
        "RouteFile": astar_path,
        "RoadSource": "Vector (GeoJSON)",
        "Buffer_m": buffer_m,
        "Spacing_m": spacing_m,
        "MergeRadius_m": merge_radius_m,
        "n_points": info_astar.get("n_points"),
        "MatrixShape": info_astar.get("matrix_shape"),
        "Time_ms": round(t1, 2)
    })
    print(f"A* runtime: {t1:.2f} ms")

    # Dijkstra Metrics
    print("\n==============================")
    print("Dijkstra + FW")
    print("==============================")
    t0 = time.perf_counter()
    run_dijkstra(
        route_geojson=dijkstra_path,
        graphml_path=graphml_path,
        buffer_m=buffer_m,
        spacing_m=spacing_m,
        output_dir=base_output_dir
    )
    t1 = (time.perf_counter() - t0) * 1000
    info_dijkstra = get_matrix_info(out_dir, "fw_dijkstra")
    results.append({
        "Algorithm": "Dijkstra",
        "RouteFile": dijkstra_path,
        "RoadSource": "GraphML (OSMnx)",
        "Buffer_m": buffer_m,
        "Spacing_m": spacing_m,
        "MergeRadius_m": "-",
        "n_points": info_dijkstra.get("n_points"),
        "MatrixShape": info_dijkstra.get("matrix_shape"),
        "Time_ms": round(t1, 2)
    })
    print(f"Dijkstra runtime: {t1:.2f} ms")

    # Summary of all three algorithms

    total_time = (time.perf_counter() - t0_all) * 1000
    print("\n")
    print("Summary")
    print("\n")
    print("Runtimes")
    print("JPS:      {:.2f} ms".format(results[0]["Time_ms"]))
    print("A*:       {:.2f} ms".format(results[1]["Time_ms"]))
    print("Dijkstra: {:.2f} ms".format(results[2]["Time_ms"]))
    print(f"Total runtime: {total_time:.2f} ms")
    print("\n")
    print("Matrix Shape")
    print(f"JPS:      {results[0]['MatrixShape']}")
    print(f"A*:       {results[1]['MatrixShape']}")
    print(f"Dijkstra: {results[2]['MatrixShape']}")
    print("\n")

    df = pd.DataFrame(results)
    summary_json = out_dir / "fw_summary.json"
    summary_csv = out_dir / "fw_summary.csv"
    df.to_json(summary_json, orient="records", indent=2)
    df.to_csv(summary_csv, index=False)

    print(f"Table Summary saved to:\n  - {summary_json}\n  - {summary_csv}")

    return df


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    run_fw_pipelines()
