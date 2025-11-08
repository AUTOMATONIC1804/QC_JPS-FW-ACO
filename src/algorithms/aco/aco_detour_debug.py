# src/algorithms/aco/aco_detour_debug.py
# -*- coding: utf-8 -*-
"""
Detour feasibility debugger (uses your existing JPS path)
- Reads fw_<method>_points.geojson
- Reads data/outputs/jps_path.geojson and extracts the LineString with role=='path'
- Computes per-node:
    d_path (distance to path, meters)
    s_proj (projected chainage along path, meters)
    detour_feasible (within detour_limit OR can rejoin within rejoin_limit)
- Writes GeoJSON for quick QGIS inspection
"""

import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

def _load_path_line(jps_path_file: str) -> LineString:
    gdf = gpd.read_file(jps_path_file)
    if "role" in gdf.columns:
        rows = gdf[gdf["role"] == "path"]
        if not rows.empty:
            geom = rows.geometry.iloc[0]
            if isinstance(geom, LineString):
                return geom
            # If it's MultiLineString, merge to a LineString-ish
            return geom if isinstance(geom, LineString) else geom.union
    # Fallback: first LineString in file
    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            return geom
    raise RuntimeError("No LineString path found in jps_path.geojson")

def _compute_detour_features(points_gdf, path_line, detour_limit=300, rejoin_limit=500):
    # project to metric
    pts_m = points_gdf.to_crs("EPSG:3857").copy()
    line_m = gpd.GeoSeries([path_line], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    # distance to path and chainage
    pts_m["d_path"] = pts_m.geometry.distance(line_m)
    pts_m["s_proj"] = pts_m.geometry.apply(lambda p: float(line_m.project(p)))
    total_len = float(line_m.length)

    # simple feasibility: close to path OR can rejoin downstream within rejoin_limit
    feasible = []
    for _, row in pts_m.iterrows():
        dpath = float(row["d_path"])
        if dpath <= detour_limit:
            feasible.append(True)
            continue
        s_proj = float(row["s_proj"])
        s_down = min(s_proj + rejoin_limit, total_len)
        rejoin_pt = line_m.interpolate(s_down)
        feasible.append(row.geometry.distance(rejoin_pt) <= rejoin_limit)

    pts_m["detour_feasible"] = feasible
    return pts_m.to_crs("EPSG:4326")

def main(method: str, out_dir: str, detour_limit: float, rejoin_limit: float):
    base_fw = Path("data/outputs/floyd_warshall")
    points_fp = base_fw / f"fw_{method}_points.geojson"
    jps_path_fp = Path("data/outputs/jps_path.geojson")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not points_fp.exists():
        raise FileNotFoundError(points_fp)
    if not jps_path_fp.exists():
        raise FileNotFoundError(jps_path_fp)

    points = gpd.read_file(points_fp)
    path_line = _load_path_line(str(jps_path_fp))

    detour_gdf = _compute_detour_features(
        points_gdf=points,
        path_line=path_line,
        detour_limit=detour_limit,
        rejoin_limit=rejoin_limit,
    )
    detour_gdf["fw_index"] = np.arange(len(detour_gdf))
    detour_gdf["detour_label"] = detour_gdf["detour_feasible"].map(lambda x: "feasible" if x else "not_feasible")

    out_fp = out_dir / "debug_detour_nodes.geojson"
    detour_gdf.to_file(out_fp, driver="GeoJSON")
    print(f"✅ Saved: {out_fp}")
    print(f"🟢 feasible: {int(detour_gdf['detour_feasible'].sum())} / {len(detour_gdf)} "
          f"🔴 not_feasible: {int((~detour_gdf['detour_feasible']).sum())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="jps", choices=["jps", "dijkstra", "astar"])
    ap.add_argument("--out_dir", type=str, default="data/outputs/aco")
    ap.add_argument("--detour_limit", type=float, default=300.0)
    ap.add_argument("--rejoin_limit", type=float, default=500.0)
    args = ap.parse_args()
    main(**vars(args))
