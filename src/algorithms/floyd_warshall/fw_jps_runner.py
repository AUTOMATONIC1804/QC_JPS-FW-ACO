"""
fw_jps_runner.py
-----------------------------
Floyd–Warshall pipeline for a JPS-generated route.

Assumptions
===========
- The JPS route is saved to GeoJSON at data/outputs/jps_path.geojson.
- The route is expected in EPSG:3857 (metric grid domain).
- Passable roads are provided by a local GeoPackage (default: data/processed/qc_roads_major.gpkg).

Pipeline
========
1) Load route (EPSG:3857 expected) → reproject to 3857 if needed.
2) Build buffer polygon around the route (default 5000 m).
3) Clip local roads (GPKG) to the buffer; filter to drivable where possible.
4) Sample points along clipped roads every spacing_m (default 500 m), with dedup.
5) Build pairwise distance matrix using Haversine (meters) from points (reprojected to WGS84).
6) Run vectorized Floyd–Warshall to get APSP.
7) Save outputs: buffer/roads/points GeoJSON and D/FW .npy files.

Run
===
python -m algorithms.floyd_warshall.fw_jps_runner \
  --edges_gpkg data/processed/qc_roads_major.gpkg \
  --buffer_m 5000 --spacing_m 500
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import warnings

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from math import radians, sin, cos, asin, sqrt


WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# --------------------------------------------------------------------------------------
# Config & helpers
# --------------------------------------------------------------------------------------

@dataclass
class FWConfig:
    route_geojson: str = "data/outputs/jps_path.geojson"
    edges_gpkg: str = "data/processed/qc_rods_major.gpkg"  # <-- fix below in argparse default
    edges_layer: Optional[str] = None
    buffer_m: float = 5000.0
    spacing_m: float = 500.0
    expected_route_crs: str = METRIC  # JPS expected CRS


def _ensure_crs(gdf: gpd.GeoDataFrame, expected: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(expected)
    elif str(gdf.crs) != expected:
        gdf = gdf.to_crs(expected)
    return gdf


def _haversine_m(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(min(1.0, np.sqrt(a)))
    return 6371008.8 * c  # meters


def haversine_matrix(coords_lonlat: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords_lonlat)
    D = np.full((n, n), np.inf, dtype=float)
    if n == 0:
        return D
    np.fill_diagonal(D, 0.0)
    for i in range(n):
        xi = coords_lonlat[i]
        for j in range(i + 1, n):
            d = _haversine_m(xi, coords_lonlat[j])
            D[i, j] = D[j, i] = d
    return D


def floyd_warshall_numpy(D: np.ndarray) -> np.ndarray:
    dist = D.copy()
    n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


# --------------------------------------------------------------------------------------
# Core steps
# --------------------------------------------------------------------------------------

def load_route_line(route_geojson: str, expected_route_crs: str) -> LineString:
    gdf = gpd.read_file(route_geojson)
    gdf = _ensure_crs(gdf, expected_route_crs)
    gdf = gdf.to_crs(METRIC)
    u = unary_union(gdf.geometry.values)
    if isinstance(u, LineString):
        return u
    if isinstance(u, MultiLineString):
        coords = []
        for ls in u.geoms:
            coords.extend(ls.coords)
        return LineString(coords)
    raise ValueError(f"Unsupported route geometry: {u.geom_type}")


def buffer_around_line(line_3857: LineString, buffer_m: float) -> gpd.GeoDataFrame:
    buf = line_3857.buffer(buffer_m)
    return gpd.GeoDataFrame({"id": [0]}, geometry=[buf], crs=METRIC)


def _filter_drivable(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "highway" not in edges.columns:
        return edges
    drivable = {
        "motorway","trunk","primary","secondary","tertiary",
        "unclassified","residential","service"
    }
    def ok(x):
        if x is None:
            return True
        if isinstance(x, str):
            return x in drivable
        try:
            return any(v in drivable for v in x)
        except Exception:
            return True
    return edges[edges["highway"].apply(ok)]


def get_roads_within_buffer_gpkg(buffer_gdf: gpd.GeoDataFrame, edges_gpkg: str, edges_layer: Optional[str]) -> gpd.GeoDataFrame:
    edges = gpd.read_file(edges_gpkg, layer=edges_layer) if edges_layer else gpd.read_file(edges_gpkg)
    if edges.crs is None:
        warnings.warn("Edges GPKG has no CRS; assuming WGS84.")
        edges = edges.set_crs(WGS84)
    edges = edges.to_crs(METRIC)
    clipped = gpd.overlay(edges, buffer_gdf, how="intersection")
    return _filter_drivable(clipped).reset_index(drop=True)


def sample_points_along_roads(edges_3857: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    rows = []
    for idx, geom in edges_3857.geometry.items():
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for ls in lines:
            length = ls.length
            n = max(1, int(length // spacing_m))
            for i in range(n + 1):
                d = min(i * spacing_m, length)
                pt = ls.interpolate(d)
                rows.append({"edge_id": idx, "geometry": pt})
    pts = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC)
    if len(pts) == 0:
        return pts
    pts["X"] = pts.geometry.x.round(0)
    pts["Y"] = pts.geometry.y.round(0)
    pts = pts.drop_duplicates(subset=["X","Y"]).drop(columns=["X","Y"]).reset_index(drop=True)
    return pts


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def run(cfg: FWConfig, output_dir: str = "data/outputs/floyd_warshall", prefix: str = "fw_jps") -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    line_3857 = load_route_line(cfg.route_geojson, cfg.expected_route_crs)
    timings["load_route_and_project_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    buffer_gdf = buffer_around_line(line_3857, cfg.buffer_m)
    timings["buffer_ms"] = (time.perf_counter() - t) * 1000

    if cfg.edges_gpkg is None:
        raise ValueError("edges_gpkg is required for JPS FW pipeline.")
