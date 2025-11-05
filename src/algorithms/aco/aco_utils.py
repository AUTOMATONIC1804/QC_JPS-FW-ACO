"""
aco_utils.py
-----------------------------------------
Shared utility functions for ACO-based
station optimization across all algorithm variants.

Handles:
- Loading Floyd–Warshall nodes (points)
- Loading FW road geometries (passable network)
- Loading FW distance matrix (.npy)
- Loading start/end route points from path GeoJSON
- Standardized Haversine formula (same as FW core)
"""

import geopandas as gpd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Point
import os


# ---------------------------------------------------
# Haversine Distance (Original Formula for Consistency)
# ---------------------------------------------------
def haversine_m(p1, p2):
    """
    Haversine distance in meters between two points in lon/lat (WGS84).
    Matches the version used in fw_core.py for uniformity.
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


# ---------------------------------------------------
# FW Loader
# ---------------------------------------------------
def load_fw_data(points_file, dist_file, roads_file=None):
    """
    Loads:
    - Sampled points (GeoDataFrame)
    - Distance matrix (numpy)
    - Optional road geometries for reference

    Returns:
    --------
    nodes : list of (lon, lat)
    D : np.ndarray
    poi_scores : np.ndarray (placeholder, uniform 1.0 for now)
    roads_gdf : GeoDataFrame or None
    """
    print("📂 Loading FW dataset...")
    points_gdf = gpd.read_file(points_file)
    points_gdf = points_gdf.to_crs("EPSG:4326")

    # Node coordinates
    nodes = [(geom.x, geom.y) for geom in points_gdf.geometry]

    # Load distance matrix
    D = np.load(dist_file)

    print(f"✅ Loaded {len(nodes)} nodes and {D.shape} distance matrix from {os.path.basename(points_file)}")

    # Optional roads
    if roads_file:
        try:
            roads_gdf = gpd.read_file(roads_file)
            print(f"🛣️ Loaded {len(roads_gdf)} road geometries from {os.path.basename(roads_file)}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load roads file ({roads_file}): {e}")
            roads_gdf = None
    else:
        roads_gdf = None

    # Placeholder POI scores (until POI weighting integration)
    poi_scores = np.ones(len(nodes))

    return nodes, D, poi_scores, roads_gdf


# ---------------------------------------------------
# Fixed Endpoints Loader
# ---------------------------------------------------
def load_fixed_endpoints(route_geojson, nodes, snap_tol_m=200):
    """
    Reads a route file (e.g. jps_path.geojson),
    extracts start and end coordinates,
    and finds the nearest sampled nodes by distance.

    Returns:
    --------
    start_idx, end_idx : int
    """

    print(f"📍 Loading route: {os.path.basename(route_geojson)}")
    gdf = gpd.read_file(route_geojson)
    gdf = gdf.to_crs("EPSG:4326")

    # Get first and last coordinate of the route
    geom = gdf.geometry.iloc[0]
    if geom.geom_type == "MultiLineString":
        coords = [pt for ls in geom.geoms for pt in ls.coords]
    else:
        coords = list(geom.coords)

    start_pt = coords[0]
    end_pt = coords[-1]

    # Snap to nearest FW nodes
    d_start = [haversine_m(start_pt, n) for n in nodes]
    d_end = [haversine_m(end_pt, n) for n in nodes]
    start_idx = int(np.argmin(d_start))
    end_idx = int(np.argmin(d_end))

    print(f"🎯 Start snapped to node {start_idx} ({d_start[start_idx]:.1f} m away)")
    print(f"🏁 End snapped to node {end_idx} ({d_end[end_idx]:.1f} m away)")

    return start_idx, end_idx
