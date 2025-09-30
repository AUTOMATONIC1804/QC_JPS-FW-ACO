# src/utils/rasterize.py
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def graph_to_grid(edges_gdf: gpd.GeoDataFrame, resolution=10, buffer_m=3):
    """
    Convert a GeoDataFrame of road edges into a binary grid.

    Parameters
    ----------
    edges_gdf : gpd.GeoDataFrame
        GeoDataFrame of road edges (must have LineString geometries).
    resolution : float
        Cell size in meters.
    buffer_m : float
        Buffer (road thickness) in meters.

    Returns
    -------
    grid : np.ndarray
        2D numpy array: 1 = road, 0 = obstacle
    meta : dict
        Metadata with origin (xmin, ymax), resolution, CRS
    """

    if edges_gdf.empty:
        raise ValueError("edges_gdf is empty. No roads to rasterize.")
    if edges_gdf.crs is None:
        raise ValueError("edges_gdf has no CRS. Please set one before rasterizing.")

    # 1. Buffer roads
    roads_buffered = edges_gdf.copy()
    roads_buffered["geometry"] = roads_buffered.geometry.buffer(buffer_m)

    # 2. Get bounds and grid size
    xmin, ymin, xmax, ymax = roads_buffered.total_bounds
    width = int((xmax - xmin) / resolution) + 1
    height = int((ymax - ymin) / resolution) + 1

    # 3. Initialize grid (0 = obstacle everywhere)
    grid = np.zeros((height, width), dtype=np.uint8)

    # 4. Rasterize: mark road cells as 1
    for geom in roads_buffered.geometry:
        if geom is None:
            continue
        minx, miny, maxx, maxy = geom.bounds
        x_start = int((minx - xmin) / resolution)
        x_end = int((maxx - xmin) / resolution)
        y_start = int((ymax - maxy) / resolution)
        y_end = int((ymax - miny) / resolution)

        for i in range(y_start, y_end + 1):
            for j in range(x_start, x_end + 1):
                cx = xmin + j * resolution + resolution / 2
                cy = ymax - i * resolution - resolution / 2
                if geom.contains(Point(cx, cy)):
                    grid[i, j] = 1  # road

    meta = {
        "origin": (xmin, ymax),
        "resolution": resolution,
        "crs": edges_gdf.crs
    }

    return grid, meta
