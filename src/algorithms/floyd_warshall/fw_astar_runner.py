"""
fw_astar_runner.py
-----------------------------
Floyd–Warshall (FW) for A*-generated route (GeoTIFF-based).

- A* route GeoJSON: path line (EPSG:3857)
- GeoTIFF grid: passable = 1 (roads)
- Builds buffer around path
- Extracts traversable road *lines* within buffer (not points)
- Runs FW using sampled points from those lines
"""

import argparse
import time
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, LineString, MultiLineString, mapping
from shapely.ops import unary_union
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# ---------------------------------------------------
# Distance + FW
# ---------------------------------------------------

def _haversine_m(p1, p2):
    lon1, lat1 = p1; lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2 - lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2 - lon1)/2)**2
    return 2 * 6371008.8 * asin(min(1.0, np.sqrt(a)))

def haversine_matrix(coords):
    n = len(coords)
    D = np.full((n, n), np.inf)
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
# Geo + raster helpers
# ---------------------------------------------------

def load_astar_route(route_geojson):
    gdf = gpd.read_file(route_geojson)
    if gdf.crs is None:
        warnings.warn("A* route missing CRS; assuming EPSG:3857.")
        gdf = gdf.set_crs(METRIC)
    else:
        gdf = gdf.to_crs(METRIC)

    geom = unary_union(gdf.geometry.values)
    if isinstance(geom, MultiLineString):
        coords = []
        for ls in geom.geoms:
            coords.extend(ls.coords)
        return LineString(coords)
    return geom


def buffer_around_line(line_3857, buffer_m=1000):
    buf = line_3857.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=[buf], crs=METRIC)


def extract_roads_from_raster(tif_path, buffer_geom):
    """
    Extract traversable road *lines* (value=1) within buffer from GeoTIFF.
    Uses rasterio.features.shapes() to vectorize connected cells.
    """
    with rasterio.open(tif_path) as src:
        buf_proj = buffer_geom.to_crs(src.crs)
        out_image, out_transform = mask(src, buf_proj.geometry, crop=True)
        mask_data = out_image[0] == 1

        # Vectorize connected components (road-like regions)
        results = list(shapes(out_image[0], mask=mask_data, transform=out_transform))
        roads = []
        for geom, val in results:
            if val == 1:
                geom_shape = shape(geom)
                # Extract boundaries as line geometry
                if geom_shape.geom_type == "Polygon":
                    line = LineString(geom_shape.exterior.coords)
                    roads.append({"geometry": line})
                elif geom_shape.geom_type in ["LineString", "MultiLineString"]:
                    roads.append({"geometry": geom_shape})

        roads_gdf = gpd.GeoDataFrame(roads, geometry="geometry", crs=src.crs)
        if len(roads_gdf) == 0:
            print("⚠️ No traversable road lines found in buffer.")
            return roads_gdf

        roads_gdf = roads_gdf.to_crs(METRIC)
        print(f"✅ Extracted {len(roads_gdf)} road line features from raster buffer.")
        return roads_gdf


def sample_points_along_lines(roads_gdf, spacing_m=500):
    """Sample points at regular spacing along all road LineStrings."""
    pts = []
    for geom in roads_gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            length = geom.length
            dists = np.arange(0, length + spacing_m, spacing_m)
            for d in dists:
                pts.append(geom.interpolate(d))
        elif geom.geom_type == "MultiLineString":
            for ls in geom.geoms:
                length = ls.length
                dists = np.arange(0, length + spacing_m, spacing_m)
                for d in dists:
                    pts.append(ls.interpolate(d))
    pts_gdf = gpd.GeoDataFrame(geometry=pts, crs=roads_gdf.crs)
    return pts_gdf


# ---------------------------------------------------
# Runner
# ---------------------------------------------------

def run_fw_astar_raster(
    route_geojson="data/outputs/astar_path.geojson",
    tif_path="data/processed/qc_grid_clean.tif",
    buffer_m=1000,
    spacing_m=500,
    output_dir="data/outputs/floyd_warshall"
):
    print("🚆 Running FW (A* with real road lines from GeoTIFF)")
    print(f"Route: {route_geojson}")
    print(f"Raster: {tif_path}")
    print(f"Buffer: ±{buffer_m} m | Spacing: {spacing_m} m")

    t0 = time.perf_counter()

    # 1. Load route + buffer
    route_line = load_astar_route(route_geojson)
    buffer_gdf = buffer_around_line(route_line, buffer_m)

    # 2. Extract raster roads (LineStrings)
    roads_gdf = extract_roads_from_raster(tif_path, buffer_gdf)

    # 3. Sample points along road lines
    pts_gdf = sample_points_along_lines(roads_gdf, spacing_m)

    if len(pts_gdf) == 0:
        print("❌ No points found for FW computation.")
        return

    # 4. Build distance + FW
    pts_wgs = pts_gdf.to_crs(WGS84)
    coords = [(p.x, p.y) for p in pts_wgs.geometry]
    D = haversine_matrix(coords)
    FW = floyd_warshall_numpy(D)

    # 5. Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_astar_buffer.geojson", driver="GeoJSON")
    roads_gdf.to_crs(WGS84).to_file(out / "fw_astar_roads.geojson", driver="GeoJSON")
    pts_wgs.to_file(out / "fw_astar_points.geojson", driver="GeoJSON")
    np.save(out / "fw_astar_D.npy", D)
    np.save(out / "fw_astar_FW.npy", FW)

    print("\n=== ✅ FW A* Raster Summary ===")
    print(f"🟩 Road lines: {len(roads_gdf)} | Points sampled: {len(pts_gdf)}")
    print(f"📐 Matrix: {D.shape}")
    print(f"🕒 Runtime: {time.perf_counter() - t0:.2f} s")
    print(f"📂 Output: {out}")
    print("===================================")


# ---------------------------------------------------
# Entry
# ---------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/astar_path.geojson")
    ap.add_argument("--tif_path", default="data/processed/qc_grid_clean.tif")
    ap.add_argument("--buffer_m", type=float, default=1000)
    ap.add_argument("--spacing_m", type=float, default=500)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    args = ap.parse_args()

    run_fw_astar_raster(
        route_geojson=args.route_geojson,
        tif_path=args.tif_path,
        buffer_m=args.buffer_m,
        spacing_m=args.spacing_m,
        output_dir=args.output_dir
    )
