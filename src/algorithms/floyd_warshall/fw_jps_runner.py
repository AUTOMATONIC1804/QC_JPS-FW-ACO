"""
fw_jps_runner.py
-------------------------------------------
Uses JPS path + vector-based roads (GeoJSON) to generate
Floyd–Warshall-ready points and matrices.
"""

import argparse, time, numpy as np, geopandas as gpd, pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from pathlib import Path
from math import radians, sin, cos, asin
from scipy.spatial import cKDTree
from contextlib import contextmanager
import sys
from io import StringIO

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

def floyd_warshall_numpy(D):
    dist = D.copy(); n = dist.shape[0]
    for k in range(n): dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist

# ---------------------------------------------------
# Geo helpers
# ---------------------------------------------------
def load_jps_route(route_geojson):
    gdf = gpd.read_file(route_geojson)
    gdf = gdf.set_crs(METRIC) if gdf.crs is None else gdf.to_crs(METRIC)
    geom = unary_union(gdf.geometry.values)
    if isinstance(geom, MultiLineString):
        coords = []
        for ls in geom.geoms: coords.extend(ls.coords)
        geom = LineString(coords)
    print(f"✅ Loaded JPS route (length ≈ {geom.length:.0f} m)")
    return geom

def buffer_around_line(line, buffer_m=1000):
    return gpd.GeoDataFrame(geometry=[line.buffer(buffer_m)], crs=METRIC)

# ---------------------------------------------------
# Road extraction (with GDAL silencer)
# ---------------------------------------------------
@contextmanager
def silence_gdal():
    """Temporarily suppress GDAL/Fiona stderr output."""
    saved_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = saved_stderr


def extract_vector_roads(roads_vector, buffer_geom):
    """Extract edges (roads) within the JPS buffer, suppressing GDAL logs."""
    print(f"📂 Loading vector roads: {roads_vector}")

    try:
        with silence_gdal():
            roads = gpd.read_file(roads_vector)
    except Exception as e:
        print(f"❌ Could not load road file: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC)

    # Only keep geometry column (avoid unsupported field types)
    if "geometry" not in roads.columns:
        print("❌ Invalid road data — no geometry column found.")
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC)
    roads = roads[["geometry"]].copy()

    # Normalize CRS
    if roads.crs is None:
        roads = roads.set_crs(WGS84)
    roads = roads.to_crs(METRIC)

    # Clip to buffer
    buf = buffer_geom.to_crs(METRIC)
    clipped = gpd.overlay(roads, buf, how="intersection")
    clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    clipped = clipped[clipped.geometry.length > 10]

    print(f"✅ Extracted {len(clipped)} road segments inside buffer.")
    return clipped

# ---------------------------------------------------
# Sampling + Deduplication
# ---------------------------------------------------
def sample_points_along(line, spacing):
    L = line.length
    if L == 0:
        return gpd.GeoDataFrame(geometry=[Point(line.coords[0])], crs=METRIC)
    dists = np.arange(0, L + 1e-6, spacing)
    pts = [line.interpolate(d) for d in dists]
    if pts[-1].distance(Point(line.coords[-1])) > 1:
        pts.append(Point(line.coords[-1]))
    return gpd.GeoDataFrame(geometry=pts, crs=METRIC)

def merge_nearby(points, rad):
    if len(points) == 0: return points
    coords = np.array([[p.x, p.y] for p in points.geometry])
    tree = cKDTree(coords)
    groups = tree.query_ball_tree(tree, rad)
    seen, keep = set(), []
    for i, g in enumerate(groups):
        if i in seen: continue
        seen.update(g); keep.append(i)
    cleaned = points.iloc[keep].copy()
    print(f"🧹 Dedup {len(points)} → {len(cleaned)} pts (r={rad} m)")
    return cleaned.reset_index(drop=True)

# ---------------------------------------------------
# Runner
# ---------------------------------------------------
def run_fw_vector(
    route_geojson="data/outputs/jps_path.geojson",
    roads_vector="data/processed/qc_roads_major_edges.geojson",
    buffer_m=2000,
    spacing_m=600,
    merge_radius_m=450,
    output_dir="data/outputs/floyd_warshall"
):
    print("🚆 FW (JPS + Vector Roads v5.3)")
    t0 = time.perf_counter()

    route = load_jps_route(route_geojson)
    buffer_gdf = buffer_around_line(route, buffer_m)
    roads_gdf = extract_vector_roads(roads_vector, buffer_gdf)

    # Ensure JPS path is always included
    jps_gdf = gpd.GeoDataFrame(geometry=[route], crs=METRIC)
    roads_gdf = pd.concat([roads_gdf, jps_gdf], ignore_index=True)
    print(f"🛤️ Added JPS route (total lines: {len(roads_gdf)})")

    # Sample
    jps_pts = sample_points_along(route, spacing_m)
    print(f"🟨 JPS pts: {len(jps_pts)}")
    road_pts_list = [sample_points_along(r, spacing_m) for r in roads_gdf.geometry if not r.is_empty]
    road_pts_gdf = gpd.GeoDataFrame(pd.concat(road_pts_list, ignore_index=True), crs=METRIC)
    print(f"🛣️ Road pts: {len(road_pts_gdf)}")

    # Merge
    all_pts = gpd.GeoDataFrame(pd.concat([jps_pts, road_pts_gdf], ignore_index=True)).set_crs(METRIC)
    merged = merge_nearby(all_pts, merge_radius_m)
    if len(merged) == 0:
        print("❌ No points generated."); return

    # Floyd–Warshall
    pts_wgs = merged.to_crs(WGS84)
    coords = [(p.x, p.y) for p in pts_wgs.geometry]
    n = len(coords)
    D = np.full((n, n), np.inf); np.fill_diagonal(D, 0)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _haversine_m(coords[i], coords[j])
    FW = floyd_warshall_numpy(D)

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_jps_buffer.geojson", driver="GeoJSON")
    roads_gdf.to_crs(WGS84).to_file(out / "fw_jps_roads.geojson", driver="GeoJSON")
    pts_wgs.to_file(out / "fw_jps_points.geojson", driver="GeoJSON")
    np.save(out / "fw_jps_D.npy", D)
    np.save(out / "fw_jps_FW.npy", FW)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    print("\n✅ Summary")
    print(f"🟨 JPS pts: {len(jps_pts)} | 🛣️ Road pts: {len(road_pts_gdf)} | 🧹 Unique: {len(merged)}")
    print(f"📐 Matrix {D.shape} | ⏱ {elapsed_ms:.2f} ms | 📂 {out}")
    print("===================================")

# ---------------------------------------------------
# Entry
# ---------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/jps_path.geojson")
    ap.add_argument("--roads_vector", default="data/processed/qc_roads_major_edges.geojson")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=600)
    ap.add_argument("--merge_radius_m", type=float, default=450)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    args = ap.parse_args()
    run_fw_vector(**vars(args))
