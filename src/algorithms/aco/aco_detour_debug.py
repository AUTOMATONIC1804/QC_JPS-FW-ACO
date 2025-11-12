# src/algorithms/aco/aco_detour_debug.py
# -*- coding: utf-8 -*-
"""
Detour feasibility debugger (LOCAL chainage corridor)
----------------------------------------------------
- Reads fw_<method>_points.geojson
- Reads data/outputs/<method>_path.geojson (role=='path')
- Per node:
    * d_path (m), s_proj (m along path)
    * Build a LOCAL subline around s_proj (± window_m)
    * detour_feasible if inside detour buffer of that subline,
      or can rejoin inside rejoin buffer of that subline
This eliminates false positives from nearby-but-far-in-chainage path segments.

Supports: jps, dijkstra, astar
Usage: python -m src.algorithms.aco.aco_detour_debug --method jps
"""

import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import split as shp_split

# ----------------------------- helpers -----------------------------

def _load_path_line(path_file: str) -> LineString:
    """Load the main path LineString from a route GeoJSON file."""
    gdf = gpd.read_file(path_file)
    if "role" in gdf.columns:
        rows = gdf[gdf["role"] == "path"]
        if not rows.empty and isinstance(rows.geometry.iloc[0], LineString):
            return rows.geometry.iloc[0]
    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            return geom
    raise RuntimeError(f"No LineString path found in {path_file}")

def _cut_line_at_distance(line: LineString, dist: float):
    """Return line cut at distance 'dist' (0<=dist<=length)."""
    if dist <= 0.0:
        return [LineString([]), LineString(line)]
    L = line.length
    if dist >= L:
        return [LineString(line), LineString([])]
    # cut by creating a point and splitting
    pt = line.interpolate(dist)
    # small buffer to ensure split hits a vertex
    res = shp_split(line, pt.buffer(1e-7))
    # order pieces by start chainage
    parts = sorted(list(res.geoms), key=lambda ls: ls.project(line.coords[0] if False else line.interpolate(0)))
    # heuristic: pick the piece that ends closest to 'dist' as left
    # simpler: accumulate lengths to decide
    acc = 0.0
    left = []
    for seg in parts:
        segL = seg.length
        if acc + segL <= dist + 1e-6:
            left.append(seg)
            acc += segL
        else:
            # split this seg internally to match exact dist
            rem = dist - acc
            if rem > 1e-9:
                pt2 = seg.interpolate(rem)
                split2 = shp_split(seg, pt2.buffer(1e-7))
                seg_left = max(split2.geoms, key=lambda x: x.length)  # the longer piece ≈ left
                seg_right = min(split2.geoms, key=lambda x: x.length)
                left.append(seg_left)
                right_first = seg_right
            else:
                right_first = seg
            right = [right_first] + [s for s in parts[parts.index(seg)+1:]]
            return [LineString([c for ls in left for c in ls.coords]) if left else LineString([]),
                    LineString([c for ls in right for c in ls.coords]) if right else LineString([])]
    # fallback
    return [LineString(line), LineString([])]

def _subline(line: LineString, start_m: float, end_m: float) -> LineString:
    """Extract subline between chainages [start_m, end_m]."""
    L = line.length
    a = max(0.0, min(start_m, L))
    b = max(0.0, min(end_m, L))
    if b <= a:
        return LineString([])
    left, right = _cut_line_at_distance(line, a)
    sub, _ = _cut_line_at_distance(right, b - a)
    return sub

# ------------------------- core computation ------------------------

def _compute_detour_features(points_gdf,
                             path_line,
                             detour_limit=300,
                             rejoin_limit=500,
                             window_m=600):
    """
    Feasibility is tested against a LOCAL corridor:
    buffer(subline(s_proj±window_m), detour/rejoin).
    """
    # Project to metric
    pts_m = points_gdf.to_crs("EPSG:3857").copy()
    path_m = gpd.GeoSeries([path_line], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    L = float(path_m.length)

    # Basic metrics
    pts_m["d_path"] = pts_m.geometry.distance(path_m)
    pts_m["s_proj"] = pts_m.geometry.apply(lambda p: float(path_m.project(p)))

    feasible, reasons = [], []

    for _, row in pts_m.iterrows():
        geom = row.geometry
        s = float(row["s_proj"])

        # Local subline
        sub = _subline(path_m, s - window_m, s + window_m)
        if sub.is_empty or sub.length == 0:
            feasible.append(False)
            reasons.append("no_local_subline")
            continue

        detour_buf = sub.buffer(detour_limit)
        rejoin_buf = sub.buffer(rejoin_limit)

        # Case 1: inside local detour corridor
        if geom.within(detour_buf):
            feasible.append(True)
            reasons.append("in_local_detour")
            continue

        # Case 2: can rejoin inside local window
        # sample along the local subline only (prevents remote segment snaps)
        ts = np.linspace(0.0, sub.length, 12)
        near_pts = [sub.interpolate(t) for t in ts]
        if min(geom.distance(p) for p in near_pts) <= rejoin_limit * 0.8 and geom.within(rejoin_buf):
            feasible.append(True)
            reasons.append("in_local_rejoin")
            continue

        feasible.append(False)
        reasons.append("outside_local_corridor")

    pts_m["detour_feasible"] = feasible
    pts_m["feasibility_reason"] = reasons
    return pts_m.to_crs("EPSG:4326")

# ------------------------------ CLI -------------------------------

def main(method: str, out_dir: str, detour_limit: float, rejoin_limit: float, window_m: float):
    """
    Main function to compute detour feasibility for nodes.
    
    Args:
        method: Algorithm method ("jps", "dijkstra", or "astar")
        out_dir: Output directory for results
        detour_limit: Maximum detour distance in meters
        rejoin_limit: Maximum rejoin distance in meters
        window_m: Half-window size for local corridor in meters
    """
    method_upper = method.upper() if method == "astar" else method.capitalize()
    
    print(f"\n{'='*60}")
    print(f"🔍 Detour Feasibility Debugger - Running with {method_upper}")
    print(f"{'='*60}\n")
    
    base_fw = Path("data/outputs/floyd_warshall")
    points_fp = base_fw / f"fw_{method}_points.geojson"
    path_fp = Path(f"data/outputs/{method}_path.geojson")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not points_fp.exists():
        raise FileNotFoundError(f"Missing points file: {points_fp}")
    if not path_fp.exists():
        raise FileNotFoundError(f"Missing path file: {path_fp}")

    print(f"📍 Algorithm: {method_upper}")
    print(f"🔍 Nodes: {points_fp}")
    print(f"🛣️ Path:  {path_fp}")

    points = gpd.read_file(points_fp)
    path_line = _load_path_line(str(path_fp))

    print(f"🧮 detour_limit={detour_limit} | rejoin_limit={rejoin_limit} | window_m={window_m}")
    detour_gdf = _compute_detour_features(
        points_gdf=points,
        path_line=path_line,
        detour_limit=detour_limit,
        rejoin_limit=rejoin_limit,
        window_m=window_m
    )

    detour_gdf["fw_index"] = np.arange(len(detour_gdf))
    detour_gdf["detour_label"] = detour_gdf["detour_feasible"].map(lambda x: "feasible" if x else "not_feasible")

    out_fp = out_dir / "debug_detour_nodes.geojson"
    detour_gdf.to_file(out_fp, driver="GeoJSON")

    n_total = len(detour_gdf)
    n_feas = int(detour_gdf["detour_feasible"].sum())
    print(f"\n✅ Saved: {out_fp}")
    print(f"🟢 feasible: {n_feas} / {n_total}  🔴 not_feasible: {n_total - n_feas}")
    print(f"\n📊 Feasibility reasons:")
    print(detour_gdf["feasibility_reason"].value_counts().to_string())
    print(f"\n{'='*60}")
    print(f"✅ Detour debug complete for {method_upper}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="jps", choices=["jps", "dijkstra", "astar"])
    ap.add_argument("--out_dir", type=str, default="data/outputs/aco")
    ap.add_argument("--detour_limit", type=float, default=300.0)
    ap.add_argument("--rejoin_limit", type=float, default=500.0)
    ap.add_argument("--window_m", type=float, default=600.0,
                    help="Half-window around s_proj to define local corridor")
    args = ap.parse_args()
    main(**vars(args))
