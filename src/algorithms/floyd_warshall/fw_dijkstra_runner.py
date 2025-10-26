"""
fw_dijkstra_runner.py (real network nodes only)
--------------------------------------------------------

Floyd–Warshall (FW) for Dijkstra corridor.

- Buffer: around Dijkstra route
- Uses: only real road graph nodes within buffer
- Distances: Haversine (meters)
- CRS: EPSG:3857 for metric ops, EPSG:4326 for storage
"""

import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from math import radians, sin, cos, asin

# Coordinate systems
WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# ---------------------------------------------------
# Distance + FW
# ---------------------------------------------------
def _haversine_m(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
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


# ---------------------------------------------------
# Geo helpers
# ---------------------------------------------------
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
        geom = LineString(coords)
    print(f"✅ Loaded Dijkstra route (length ≈ {LineString(geom).length:.0f})")
    return geom


def buffer_route(line_4326: LineString, buffer_m: float) -> gpd.GeoDataFrame:
    """Create metric buffer (default 2000 m)."""
    line_m = gpd.GeoSeries([line_4326], crs=WGS84).to_crs(METRIC)
    buf = line_m.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=buf, crs=METRIC)


def load_graph_within_buffer(graphml_path: str, buffer_gdf: gpd.GeoDataFrame):
    """Load GraphML and clip to buffer polygon."""
    print(f"📂 Loading GraphML: {graphml_path}")
    G_full = ox.load_graphml(graphml_path)
    G_full = ox.project_graph(G_full, to_crs=METRIC)

    polygon = buffer_gdf.geometry.iloc[0]

    # Try to find compatible truncate function
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
        raise ImportError("❌ Could not locate truncate_graph_polygon in OSMnx.")

    try:
        G_clip = truncate_fn(G_full, polygon, retain_all=True)
    except TypeError:
        G_clip = truncate_fn(G_full, polygon)

    nodes_clip, edges_clip = ox.graph_to_gdfs(G_clip)
    print(f"✅ Clipped: {len(nodes_clip)} nodes, {len(edges_clip)} edges inside buffer")
    return nodes_clip.to_crs(WGS84), edges_clip.to_crs(WGS84)


# ---------------------------------------------------
# Node filtering (distance-based thinning)
# ---------------------------------------------------
def filter_nodes_by_spacing(nodes_gdf: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    """Keep nodes spaced at least `spacing_m` meters apart."""
    if nodes_gdf.empty:
        return nodes_gdf

    nodes_m = nodes_gdf.to_crs(METRIC).copy()
    nodes_m["x"] = nodes_m.geometry.x
    nodes_m["y"] = nodes_m.geometry.y
    nodes_m = nodes_m.sort_values(by=["x", "y"]).reset_index(drop=True)
    nodes_m = nodes_m.drop(columns=["x", "y"])

    kept_indices = []
    kept_points = []

    for idx, row in nodes_m.iterrows():
        geom = row.geometry
        if all(geom.distance(p) >= spacing_m for p in kept_points):
            kept_indices.append(idx)
            kept_points.append(geom)

    thinned = nodes_m.loc[kept_indices].copy()
    print(f"🧩 Filtered {len(nodes_gdf)} → {len(thinned)} nodes (~{spacing_m} m apart)")
    return thinned.to_crs(WGS84)


# ---------------------------------------------------
# Runner
# ---------------------------------------------------
def run_fw_dijkstra(
    route_geojson="data/outputs/dijkstra_path.geojson",
    graphml_path=r"D:\Quezon_City\data\processed\qc_roads_major.graphml",
    buffer_m=2000,
    spacing_m=500,
    output_dir="data/outputs/floyd_warshall"
):
    print("🚆 FW (Dijkstra + Real Road Network v5.3)")
    print(f"Route: {route_geojson}")
    print(f"Graph: {graphml_path}")
    print(f"Buffer: ±{buffer_m} m | Spacing: {spacing_m} m")

    t0 = time.perf_counter()

    # 1) Load Dijkstra route
    route_line = load_route_line(route_geojson)

    # 2) Buffer around route
    buffer_gdf = buffer_route(route_line, buffer_m)

    # 3) Clip road network
    nodes, edges = load_graph_within_buffer(graphml_path, buffer_gdf)

    # 4) Filter nodes
    nodes_filtered = filter_nodes_by_spacing(nodes, spacing_m)
    if len(nodes_filtered) == 0:
        print("❌ No nodes available after filtering.")
        return

    # 5) Build distance + FW
    coords = [(p.x, p.y) for p in nodes_filtered.geometry]
    D = haversine_matrix(coords)
    FW = floyd_warshall_numpy(D)

    # 6) Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_dijkstra_buffer.geojson", driver="GeoJSON")
    edges.to_file(out / "fw_dijkstra_roads.geojson", driver="GeoJSON")
    nodes_filtered.to_file(out / "fw_dijkstra_nodes.geojson", driver="GeoJSON")
    np.save(out / "fw_dijkstra_D.npy", D)
    np.save(out / "fw_dijkstra_FW.npy", FW)

    # Summary
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print("\n✅ Summary (Dijkstra + Real Nodes)")
    print(f"🧩 Nodes: {len(nodes_filtered)} | 🛣️ Edges: {len(edges)}")
    print(f"📐 Matrix {D.shape} | ⏱ {elapsed_ms:.2f} ms | 📂 {out}")
    print("===================================")


# ---------------------------------------------------
# Entry
# ---------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/dijkstra_path.geojson")
    ap.add_argument("--graphml_path", default=r"D:\Quezon_City\data\processed\qc_roads_major.graphml")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=500)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    args = ap.parse_args()

    run_fw_dijkstra(**vars(args))
