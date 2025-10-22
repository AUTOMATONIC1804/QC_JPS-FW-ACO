"""
fw_core.py
-----------
Shared Floyd–Warshall pipeline for building an APSP matrix around a route corridor.

Key features
============
- Accepts a route GeoJSON produced by JPS / A* / Dijkstra.
- Handles CRS correctly per algorithm:
    * Dijkstra route files are in EPSG:4326 (lon/lat).
    * JPS and A* route files are in EPSG:3857 (meters).
- Projects the route to EPSG:3857, buffers it (default 5 km), then collects all passable
  road geometries within that buffer using either a local GeoPackage (GPKG) or OSMnx.
- Samples points along every road segment inside the buffer (default every 500 m).
- Builds a dense pairwise distance matrix (meters) using Haversine on WGS84.
- Runs vectorized Floyd–Warshall to compute APSP.
- Saves buffer/roads/points as GeoJSON and matrices as .npy.
- Returns detailed timing (ms) for each stage + counts.

Outputs
=======
In <output_dir> with <prefix>:
- {prefix}_buffer.geojson
- {prefix}_roads.geojson
- {prefix}_points.geojson
- {prefix}_D.npy         (pairwise distances, meters)
- {prefix}_FW.npy        (APSP distances, meters)

Dependencies
============
- numpy, geopandas, shapely, pyproj (implicit), optionally osmnx for live road data.

Notes
=====
- "Passable roads" filter is best-effort: if a 'highway' column exists, we keep common
  drivable classes; otherwise we keep edges as-is.
- For large buffers / dense networks, the point count (and thus matrix size) can explode.
  Tune `buffer_m` and `spacing_m` accordingly.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import time
import warnings
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from math import radians, sin, cos, asin, sqrt

try:
    import osmnx as ox
except Exception:
    ox = None

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class FWConfig:
    """
    Parameters controlling the FW pipeline.
    - buffer_m:   Buffer radius (meters) around the route corridor (default 5000 m).
    - spacing_m:  Sampling step along roads (meters) (default 500 m).
    - use_osmnx:  If True, download passable roads via OSMnx inside the buffer polygon.
    - edges_gpkg: Optional path to a local GeoPackage of roads to clip inside the buffer.
    - edges_layer:Optional layer name for the GeoPackage.
    - expected_route_crs: The CRS the input route file is expected to be in ("EPSG:4326" or "EPSG:3857").
                         If None, we assume the file is correctly tagged and use it.
    """
    buffer_m: float = 5000.0
    spacing_m: float = 500.0
    use_osmnx: bool = False
    edges_gpkg: Optional[str] = None
    edges_layer: Optional[str] = None
    expected_route_crs: Optional[str] = None   # "EPSG:4326" for Dijkstra, "EPSG:3857" for A*/JPS


# --------------------------------------------------------------------------------------
# Math utilities
# --------------------------------------------------------------------------------------

def _haversine_m(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Haversine distance in meters between two points in lon/lat (WGS84).
    """
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(min(1.0, sqrt(a)))
    R = 6371008.8  # mean Earth radius
    return R * c


def haversine_matrix(coords_lonlat: List[Tuple[float, float]]) -> np.ndarray:
    """
    Build a symmetric pairwise distance matrix (meters) for lon/lat coordinates.
    """
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
    """
    Vectorized Floyd–Warshall. Returns APSP matrix (meters).
    Complexity is O(n^3) so be mindful of n (number of sampled points).
    """
    dist = D.copy()
    n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


# --------------------------------------------------------------------------------------
# Geometry & IO utilities
# --------------------------------------------------------------------------------------

def _ensure_crs(gdf: gpd.GeoDataFrame, expected: Optional[str]) -> gpd.GeoDataFrame:
    """
    Ensure the GeoDataFrame has a CRS:
    - If `expected` is provided and gdf has no CRS, assign expected.
    - If `expected` is provided and gdf has a different CRS, reproject to expected first.
    - If `expected` is None, keep gdf's CRS (but must exist).
    """
    if gdf.crs is None:
        if expected is None:
            raise ValueError("Route has no CRS. Provide FWConfig.expected_route_crs.")
        gdf = gdf.set_crs(expected)
        return gdf

    if expected and str(gdf.crs) != expected:
        # If the user asserts a specific CRS, reproject to that first to be precise.
        gdf = gdf.to_crs(expected)
    return gdf


def load_route_line(route_geojson: str, expected_route_crs: Optional[str]) -> LineString:
    """
    Load a route GeoJSON (LineString or MultiLineString), validate/apply its CRS,
    then reproject to EPSG:3857 and return a single LineString.

    For MultiLineString, segments are concatenated in read order for sampling convenience.
    """
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
    """
    Build a buffer polygon around a metric LineString and return as GeoDataFrame (EPSG:3857).
    """
    buf = line_3857.buffer(buffer_m)
    return gpd.GeoDataFrame({"id": [0]}, geometry=[buf], crs=METRIC)


def _filter_drivable(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter edges to common drivable types if a 'highway' column exists. Otherwise, keep as-is.
    """
    if "highway" not in edges.columns:
        return edges

    drivable = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service"
    }

    def ok(x):
        if x is None:
            return True  # keep unknowns
        if isinstance(x, str):
            return (x in drivable)
        try:
            return any(v in drivable for v in x)
        except Exception:
            return True

    return edges[edges["highway"].apply(ok)]


def get_roads_within_buffer(line_3857: LineString,
                            cfg: FWConfig,
                            buffer_gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """
    Retrieve passable roads within the buffer polygon. Two sources:
    - Local GeoPackage (edges_gpkg[/edges_layer])
    - OSMnx (use_osmnx=True)

    Returns LineString/MultiLineString GeoDataFrame in EPSG:3857.
    """
    if buffer_gdf is None:
        buffer_gdf = buffer_around_line(line_3857, cfg.buffer_m)

    # Option 1: local GPKG
    if cfg.edges_gpkg:
        edges = gpd.read_file(cfg.edges_gpkg, layer=cfg.edges_layer) if cfg.edges_layer else gpd.read_file(cfg.edges_gpkg)
        if edges.crs is None:
            warnings.warn("Edges source has no CRS; assuming EPSG:4326.")
            edges = edges.set_crs(WGS84)
        edges = edges.to_crs(METRIC)
        edges_clip = gpd.overlay(edges, buffer_gdf, how="intersection")
        edges_clip = _filter_drivable(edges_clip)
        return edges_clip.reset_index(drop=True)

    # Option 2: OSMnx
    if cfg.use_osmnx:
        if ox is None:
            raise ImportError("osmnx not installed but use_osmnx=True")
        # Build polygon in WGS84 for OSMnx retrieval
        poly_wgs84 = buffer_gdf.to_crs(WGS84).geometry.iloc[0]
        G = ox.graph_from_polygon(poly_wgs84, network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        edges = edges.to_crs(METRIC)
        edges_clip = gpd.overlay(edges, buffer_gdf, how="intersection")
        return edges_clip.reset_index(drop=True)

    raise ValueError("Provide edges_gpkg or set use_osmnx=True to fetch roads.")


def sample_points_along_roads(edges_3857: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    """
    Sample points at exact 'spacing_m' intervals along each road geometry (EPSG:3857).
    This ensures true 500 m (or custom) spacing, not vertex-based density.

    Deduplicates by snapping to a 1 m grid.
    """
    rows = []
    for idx, geom in edges_3857.geometry.items():
        if geom is None:
            continue

        # Handle MultiLineString and LineString uniformly
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
        for ls in lines:
            length = ls.length
            if length < spacing_m:
                # include only start and end
                pts = [ls.interpolate(0), ls.interpolate(length)]
            else:
                dists = np.arange(0, length + spacing_m, spacing_m)
                pts = [ls.interpolate(d) for d in dists]

            for p in pts:
                rows.append({"edge_id": idx, "geometry": p})

    pts = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")

    if len(pts) == 0:
        return pts

    # Snap to 1 m grid (avoid duplicates)
    pts["X"] = (pts.geometry.x / 1).round(0)
    pts["Y"] = (pts.geometry.y / 1).round(0)
    pts = pts.drop_duplicates(subset=["X", "Y"]).drop(columns=["X", "Y"]).reset_index(drop=True)
    return pts


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def run_fw_pipeline(
    route_geojson: str,
    output_dir: str,
    prefix: str,
    cfg: FWConfig
) -> Dict[str, Any]:
    """
    Execute the full FW pipeline with detailed timing.

    Returns
    -------
    dict with:
        - counts: n_points, matrix shape
        - timings_ms: per-stage timings in ms
        - outputs: output directory path
        - params: effective params used
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 1) Load route & project to metric
    t = time.perf_counter()
    line_3857 = load_route_line(route_geojson, cfg.expected_route_crs)
    timings["load_route_and_project_ms"] = (time.perf_counter() - t) * 1000

    # 2) Buffer
    t = time.perf_counter()
    buffer_gdf = buffer_around_line(line_3857, cfg.buffer_m)
    timings["buffer_ms"] = (time.perf_counter() - t) * 1000

    # 3) Roads within buffer
    t = time.perf_counter()
    roads_3857 = get_roads_within_buffer(line_3857, cfg, buffer_gdf=buffer_gdf)
    timings["fetch_roads_ms"] = (time.perf_counter() - t) * 1000

    # 4) Sample points
    t = time.perf_counter()
    points_3857 = sample_points_along_roads(roads_3857, cfg.spacing_m)
    timings["sample_points_ms"] = (time.perf_counter() - t) * 1000

    # 5) Build distance matrix
    t = time.perf_counter()
    pts_wgs84 = points_3857.to_crs(WGS84) if len(points_3857) > 0 else points_3857
    coords = [(geom.x, geom.y) for geom in pts_wgs84.geometry] if len(points_3857) > 0 else []
    D = haversine_matrix(coords)
    timings["build_matrix_ms"] = (time.perf_counter() - t) * 1000

    # 6) Floyd–Warshall
    t = time.perf_counter()
    FW = floyd_warshall_numpy(D) if D.size > 0 else D
    timings["floyd_warshall_ms"] = (time.perf_counter() - t) * 1000

    # 7) Save outputs
    t = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / f"{prefix}_buffer.geojson", driver="GeoJSON")
    roads_3857.to_crs(WGS84).to_file(out / f"{prefix}_roads.geojson", driver="GeoJSON")
    if len(points_3857) > 0:
        points_3857.to_crs(WGS84).to_file(out / f"{prefix}_points.geojson", driver="GeoJSON")
    else:
        # Write empty GDF if no points to avoid confusion
        gpd.GeoDataFrame(geometry=[], crs=WGS84).to_file(out / f"{prefix}_points.geojson", driver="GeoJSON")
    np.save(out / f"{prefix}_D.npy", D)
    np.save(out / f"{prefix}_FW.npy", FW)
    timings["save_outputs_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "counts": {
            "n_points": int(len(points_3857)),
            "matrix_shape": list(D.shape)
        },
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "outputs": str(out),
        "params": {
            "buffer_m": cfg.buffer_m,
            "spacing_m": cfg.spacing_m,
            "use_osmnx": cfg.use_osmnx,
            "edges_gpkg": cfg.edges_gpkg,
            "edges_layer": cfg.edges_layer,
            "expected_route_crs": cfg.expected_route_crs,
            "route_file": route_geojson,
            "prefix": prefix,
        },
    }
