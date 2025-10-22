"""
fw_dijkstra_runner.py (real network nodes + 2 km buffer)
--------------------------------------------------------

Floyd–Warshall (FW) for Dijkstra route corridor.

- Buffer: 2000 m around route
- Path points: from Dijkstra GeoJSON (already spaced)
- Road nodes: real graph nodes inside buffer (~500 m filtered)
- Distances: Haversine (meters)
- CRS: EPSG:3857 for metric ops, EPSG:4326 for storage
"""

import argparse, time, warnings
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from math import radians, sin, cos, asin, sqrt

# Coordinate systems
WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# -------------------------------------------------------------------------
# Distance + FW helpers
# -------------------------------------------------------------------------

def _haversine_m(p1, p2):
    lon1, lat1 = p1; lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * asin(min(1.0, np.sqrt(a)))


def haversine_matrix(coords):
    n = len(coords)
    D = np.full((n, n), np.inf)
    if n == 0:
        return D
    np.fill_diagonal(D, 0)
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_m(coords[i], coords[j])
            D[i, j] = D[j, i] = d
    return D


def floyd_warshall_numpy(D):
    dist = D.copy()
    n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


# -------------------------------------------------------------------------
# Geo helpers
# -------------------------------------------------------------------------

def load_route_line(route_geojson: str) -> LineString:
    """Load Dijkstra route (EPSG:4326) and unify to a LineString."""
    gdf = gpd.read_file(route_geojson)
    if gdf.crs is None:
        warnings.warn("Route GeoJSON has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs(WGS84)
    gdf = gdf.to_crs(WGS84)
    geom = unary_union(gdf.geometry.values)
    if isinstance(geom, MultiLineString):
        coords = []
        for ls in geom.geoms:
            coords.extend(ls.coords)
        return LineString(coords)
    return geom


def buffer_route(line_4326: LineString, buffer_m: float) -> gpd.GeoDataFrame:
    """Create metric buffer (2 km default)."""
    line_m = gpd.GeoSeries([line_4326], crs=WGS84).to_crs(METRIC)
    buf = line_m.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=buf, crs=METRIC)


def load_graph_within_buffer(graphml_path: str, buffer_gdf: gpd.GeoDataFrame):
    """
    Load GraphML network and clip to the buffer polygon.

    Works across OSMnx versions (1.x–2.x) by dynamically locating the
    correct truncate function.
    """
    print(f"📂 Loading GraphML: {graphml_path}")
    G_full = ox.load_graphml(graphml_path)
    G_full = ox.project_graph(G_full, to_crs=METRIC)

    polygon = buffer_gdf.geometry.iloc[0]

    # --- Try multiple possible truncate functions depending on OSMnx version ---
    truncate_fn = None
    for mod_name, func_name in [
        ("osmnx.truncate", "truncate_graph_polygon"),
        ("osmnx.utils_graph", "truncate_graph_polygon"),
        ("osmnx.graph", "truncate_graph_polygon"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[func_name])
            truncate_fn = getattr(mod, func_name, None)
            if callable(truncate_fn):
                break
        except Exception:
            continue

    if truncate_fn is None:
        raise ImportError(
            "❌ Could not locate truncate_graph_polygon in any OSMnx module. "
            "Please update or reinstall OSMnx."
        )

    # --- Perform clipping safely ---
    try:
        G_clip = truncate_fn(G_full, polygon, retain_all=True)
    except TypeError:
        # Some old versions don’t accept 'retain_all'
        G_clip = truncate_fn(G_full, polygon)

    # --- Extract nodes and edges ---
    nodes_clip, edges_clip = ox.graph_to_gdfs(G_clip)
    print(f"✅ Clipped: {len(nodes_clip)} nodes, {len(edges_clip)} edges inside buffer")
    return nodes_clip.to_crs(WGS84), edges_clip.to_crs(WGS84)



# -------------------------------------------------------------------------
# Node filtering (real graph nodes ~500 m apart)
# -------------------------------------------------------------------------

def filter_nodes_by_spacing(nodes_gdf: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    """Keep roughly one node per 'spacing_m' grid cell (in EPSG:3857)."""
    if nodes_gdf.empty:
        return nodes_gdf

    nodes_m = nodes_gdf.to_crs(METRIC)
    nodes_m["gx"] = (nodes_m.geometry.x // spacing_m).astype(int)
    nodes_m["gy"] = (nodes_m.geometry.y // spacing_m).astype(int)
    thinned = nodes_m.drop_duplicates(subset=["gx", "gy"]).drop(columns=["gx", "gy"])
    print(f"✅ Filtered {len(nodes_gdf)} → {len(thinned)} nodes (~{spacing_m} m grid)")
    return thinned.to_crs(WGS84)


# -------------------------------------------------------------------------
# Main FW Dijkstra
# -------------------------------------------------------------------------

def run_fw_dijkstra(route_geojson,
                    graphml_path,
                    buffer_m=2000,
                    spacing_m=500,
                    output_dir="data/outputs/floyd_warshall") -> Dict[str, Any]:

    timings = {}
    t0 = time.perf_counter()

    # 1. Route
    t = time.perf_counter()
    route_line = load_route_line(route_geojson)
    timings["load_route_ms"] = (time.perf_counter() - t) * 1000

    # 2. Buffer
    t = time.perf_counter()
    buffer_gdf = buffer_route(route_line, buffer_m)
    timings["buffer_ms"] = (time.perf_counter() - t) * 1000

    # 3. Graph + clip to buffer
    t = time.perf_counter()
    nodes, edges = load_graph_within_buffer(graphml_path, buffer_gdf)
    timings["graph_clip_ms"] = (time.perf_counter() - t) * 1000

    # 4. Filter nodes (~500 m)
    t = time.perf_counter()
    nodes_filtered = filter_nodes_by_spacing(nodes, spacing_m)
    timings["filter_nodes_ms"] = (time.perf_counter() - t) * 1000

    # 5. Combine with Dijkstra path points
    t = time.perf_counter()
    # take only Point features if path file has them, otherwise sample
    path_gdf = gpd.read_file(route_geojson)
    if "Point" not in path_gdf.geom_type.unique():
        # if route file only contains a LineString, convert its vertices to points
        coords = list(route_line.coords)
        path_gdf = gpd.GeoDataFrame(geometry=[gpd.points_from_xy([x for x, y in coords],
                                                                 [y for x, y in coords])])
    path_points = path_gdf[path_gdf.geom_type == "Point"].to_crs(WGS84)
    merged_pts = pd.concat([path_points, nodes_filtered], ignore_index=True)
    merged_pts = gpd.GeoDataFrame(geometry=merged_pts.geometry, crs=WGS84)
    merged_pts["lon"] = merged_pts.geometry.x.round(6)
    merged_pts["lat"] = merged_pts.geometry.y.round(6)
    merged_pts = merged_pts.drop_duplicates(subset=["lon", "lat"]).drop(columns=["lon", "lat"])
    timings["merge_points_ms"] = (time.perf_counter() - t) * 1000

    # 6. Distance + FW
    t = time.perf_counter()
    coords = [(p.x, p.y) for p in merged_pts.geometry]
    D = haversine_matrix(coords)
    timings["build_matrix_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    FW = floyd_warshall_numpy(D)
    timings["floyd_warshall_ms"] = (time.perf_counter() - t) * 1000

    # 7. Save outputs
    t = time.perf_counter()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_dijkstra_buffer.geojson", driver="GeoJSON")
    edges.to_file(out / "fw_dijkstra_roads.geojson", driver="GeoJSON")
    nodes_filtered.to_file(out / "fw_dijkstra_nodes.geojson", driver="GeoJSON")
    path_points.to_file(out / "fw_dijkstra_path_points.geojson", driver="GeoJSON")
    merged_pts.to_file(out / "fw_dijkstra_points_merged.geojson", driver="GeoJSON")
    np.save(out / "fw_dijkstra_D.npy", D)
    np.save(out / "fw_dijkstra_FW.npy", FW)
    timings["save_outputs_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "params": {"buffer_m": buffer_m, "spacing_m": spacing_m},
        "counts": {"n_points": int(len(merged_pts)), "matrix_shape": list(D.shape)},
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "outputs": str(out)
    }


# -------------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/dijkstra_path.geojson")
    ap.add_argument("--graphml_path", default=r"D:\Quezon_City\data\processed\qc_roads_major.graphml")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=500)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    args = ap.parse_args()

    info = run_fw_dijkstra(
        route_geojson=args.route_geojson,
        graphml_path=args.graphml_path,
        buffer_m=args.buffer_m,
        spacing_m=args.spacing_m,
        output_dir=args.output_dir
    )

    print("\n=== 🚆 Dijkstra → FW (real nodes, 2 km buffer, 500 m spacing) ===")
    print("Counts:", info["counts"])
    print("Timings (ms):", info["timings_ms"])
    print("Outputs →", info["outputs"])
    print("=== ✅ Done ===")
