"""
floyd_warshall_runner.py
Master runner for Floyd–Warshall (FW) pipeline across:
    - A* Search (ASTAR)
    - Jump Point Search (JPS)
(All Dijkstra and JPS-Pruned logic removed)
"""

import time, json, inspect
import pandas as pd
import numpy as np
from pathlib import Path

# Only A* and JPS runners remain
from src.algorithms.floyd_warshall.fw_astar_runner import run_fw_astar_vector as run_astar
from src.algorithms.floyd_warshall.fw_jps_runner import run_fw_vector as run_jps


# -------------------------------------------------------
# Matrix info helper
# -------------------------------------------------------
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


def _get_default_param(func, param_name, fallback=None):
    try:
        sig = inspect.signature(func)
        param = sig.parameters.get(param_name)
        if param and param.default is not inspect._empty:
            return param.default
    except (ValueError, TypeError):
        pass
    return fallback


# -------------------------------------------------------
# MAIN PIPELINE (A* first → JPS second)
# -------------------------------------------------------
def run_fw_pipelines(
    base_output_dir="data/outputs/floyd_warshall",
    astar_path="data/outputs/astar_path.geojson",
    jps_path="data/outputs/jps_path.geojson",
    roads_vector="data/processed/qc_roads_major_edges.geojson",
):
    """
    Runs the Floyd–Warshall pipelines for:
        1. A* Search (grid)
        2. Jump Point Search (grid)
    All Dijkstra and JPS-Pruned logic removed.
    """

    out_dir = Path(base_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0_all = time.perf_counter()

    # =====================================================
    # 1. A* + FW  (BASELINE FIRST)
    # =====================================================
    print("\n==============================")
    print("A* + FW")
    print("==============================")

    t0 = time.perf_counter()
    astar_buffer = _get_default_param(run_astar, "buffer_m", "default")
    astar_spacing = _get_default_param(run_astar, "spacing_m", "default")
    astar_merge = _get_default_param(run_astar, "merge_radius_m", "default")

    run_astar(
        route_geojson=astar_path,
        roads_vector=roads_vector,
        output_dir=base_output_dir
    )
    t1 = (time.perf_counter() - t0) * 1000

    info_astar = get_matrix_info(out_dir, "fw_astar")

    results.append({
        "Algorithm": "A*",
        "RouteFile": astar_path,
        "RoadSource": "Vector (GeoJSON)",
        "Buffer_m": astar_buffer,
        "Spacing_m": astar_spacing,
        "MergeRadius_m": astar_merge,
        "n_points": info_astar.get("n_points"),
        "MatrixShape": info_astar.get("matrix_shape"),
        "Time_ms": round(t1, 2)
    })

    print(f"A* runtime: {t1:.2f} ms")

    # =====================================================
    # 2. JPS + FW  (OUR MAIN ALGORITHM)
    # =====================================================
    print("\n==============================")
    print("Jump Point Search + FW")
    print("==============================")

    t0 = time.perf_counter()
    jps_buffer = _get_default_param(run_jps, "buffer_m", "default")
    jps_spacing = _get_default_param(run_jps, "spacing_m", "default")
    jps_merge = _get_default_param(run_jps, "merge_radius_m", "default")

    run_jps(
        route_geojson=jps_path,
        roads_vector=roads_vector,
        output_dir=base_output_dir
    )
    t1 = (time.perf_counter() - t0) * 1000

    info_jps = get_matrix_info(out_dir, "fw_jps")

    results.append({
        "Algorithm": "JPS",
        "RouteFile": jps_path,
        "RoadSource": "Vector (GeoJSON)",
        "Buffer_m": jps_buffer,
        "Spacing_m": jps_spacing,
        "MergeRadius_m": jps_merge,
        "n_points": info_jps.get("n_points"),
        "MatrixShape": info_jps.get("matrix_shape"),
        "Time_ms": round(t1, 2)
    })

    print(f"JPS runtime: {t1:.2f} ms")

    # =====================================================
    # SUMMARY
    # =====================================================
    total_time = (time.perf_counter() - t0_all) * 1000
    print("\nSummary\n")

    print("Runtimes")
    for res in results:
        print("{:<9}: {:>8.2f} ms".format(res["Algorithm"], res["Time_ms"]))
    print(f"Total runtime: {total_time:.2f} ms\n")

    print("Matrix Shape")
    for res in results:
        print(f"{res['Algorithm']:<9}: {res['MatrixShape']}")
    print("\n")

    # Save summary tables
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
