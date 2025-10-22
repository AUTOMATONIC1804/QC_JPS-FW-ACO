"""
fw_dijkstra_runner.py (accurate spacing + 2km buffer)
------------------------------------------------------
Floyd–Warshall (FW) for Dijkstra route corridor.

- Buffer: 2000 m around route
- Path points: every 500 m along route
- Buffer roads: every 500 m along roads inside buffer (excluding path duplicates)
- All distances computed in EPSG:3857 for accuracy
"""

import argparse, time, warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from math import radians, sin, cos, asin, sqrt

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# -------------------------------------------------------------------------
# Distance + FW helpers
# -------------------------------------------------------------------------

def _haversine_m(p1, p2):
    lon1, lat1 = p1; lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2*6371008.8*asin(min(1.0, np.sqrt(a)))


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
    dist = D.copy(); n = dist.shape[0]
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
    return gpd.GeoDataFrame(geometry=buf, crs=METRIC).to_crs(WGS84)


def graph_edges_to_gdf(graphml_path: str) -> gpd.GeoDataFrame:
    """Convert GraphML to GeoDataFrame (EPSG:4326)."""
    G = ox.load_graphml(graphml_path)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    return edges.to_crs(WGS84)


def clip_edges_with_buffer(edges: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame):
    """Clip road edges within the route buffer."""
    clipped = gpd.overlay(edges, buffer_gdf, how="intersection")
    return clipped.reset_index(drop=True)


# -------------------------------------------------------------------------
# Point sampling (accurate 500 m)
# -------------------------------------------------------------------------

def sample_points_exact_3857(geom, spacing_m: float) -> List:
    """Return points spaced at exact 500 m intervals (in EPSG:3857)."""
    if geom is None:
        return []
    lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
    pts = []
    for ls in lines:
        length = ls.length
        dists = np.arange(0, length + spacing_m, spacing_m)
        for d in dists:
            pts.append(ls.interpolate(min(d, length)))
    return pts


def sample_points_along_route(line_4326: LineString, spacing_m: float) -> gpd.GeoDataFrame:
    """Sample points every 500 m along the Dijkstra path."""
    line_m = gpd.GeoSeries([line_4326], crs=WGS84).to_crs(METRIC).iloc[0]
    pts = sample_points_exact_3857(line_m, spacing_m)
    gdf = gpd.GeoDataFrame(geometry=pts, crs=METRIC).to_crs(WGS84)
    return gdf


def sample_points_along_edges(edges_4326: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    """Sample points along all roads in buffer (EPSG:3857 accurate spacing)."""
    edges_m = edges_4326.to_crs(METRIC)
    pts = []
    for geom in edges_m.geometry:
        pts.extend(sample_points_exact_3857(geom, spacing_m))
    gdf = gpd.GeoDataFrame(geometry=pts, crs=METRIC).to_crs(WGS84)
    return gdf


# -------------------------------------------------------------------------
# Main FW Dijkstra
# -------------------------------------------------------------------------

def run_fw_dijkstra(route_geojson, graphml_path,
                    buffer_m=2000, spacing_m=500,
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

    # 3. Graph
    t = time.perf_counter()
    edges = graph_edges_to_gdf(graphml_path)
    timings["load_graph_ms"] = (time.perf_counter() - t) * 1000

    # 4. Clip to buffer
    t = time.perf_counter()
    roads = clip_edges_with_buffer(edges, buffer_gdf)
    timings["clip_edges_ms"] = (time.perf_counter() - t) * 1000

    # 5. Sample path + buffer roads
    t = time.perf_counter()
    path_pts = sample_points_along_route(route_line, spacing_m)
    road_pts = sample_points_along_edges(roads, spacing_m)
    # Merge + deduplicate
    pts = gpd.GeoDataFrame(pd.concat([path_pts, road_pts], ignore_index=True), crs=WGS84)
    pts["lon"] = pts.geometry.x.round(6)
    pts["lat"] = pts.geometry.y.round(6)
    pts = pts.drop_duplicates(subset=["lon", "lat"]).drop(columns=["lon", "lat"])
    timings["sample_points_ms"] = (time.perf_counter() - t) * 1000

    # 6. Build matrices
    t = time.perf_counter()
    coords = [(p.x, p.y) for p in pts.geometry]
    D = haversine_matrix(coords)
    timings["build_matrix_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    FW = floyd_warshall_numpy(D)
    timings["floyd_warshall_ms"] = (time.perf_counter() - t) * 1000

    # 7. Save
    t = time.perf_counter()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_file(out / "fw_dijkstra_buffer.geojson", driver="GeoJSON")
    roads.to_file(out / "fw_dijkstra_roads.geojson", driver="GeoJSON")
    path_pts.to_file(out / "fw_dijkstra_path_points.geojson", driver="GeoJSON")
    road_pts.to_file(out / "fw_dijkstra_buffer_points.geojson", driver="GeoJSON")
    pts.to_file(out / "fw_dijkstra_points_merged.geojson", driver="GeoJSON")
    np.save(out / "fw_dijkstra_D.npy", D)
    np.save(out / "fw_dijkstra_FW.npy", FW)
    timings["save_outputs_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "params": {"buffer_m": buffer_m, "spacing_m": spacing_m},
        "counts": {"n_points": int(len(pts)), "matrix_shape": list(D.shape)},
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "outputs": str(out)
    }


# -------------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/dijkstra_path.geojson")
    ap.add_argument("--graphml_path", default="data/processed/qc_roads_major.graphml")
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

    print("=== 🚆 Dijkstra → FW (2 km buffer, 500 m spacing) ===")
    print("Counts:", info["counts"])
    print("Timings (ms):", info["timings_ms"])
    print("Outputs →", info["outputs"])
    print("=== ✅ Done ===")
