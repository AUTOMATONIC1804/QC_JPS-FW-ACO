"""
fw_astar_runner.py
-------------------------------------------
Uses A* path + vector-based roads (GeoJSON) to generate
Floyd–Warshall-ready points and matrices.

Parallels fw_jps_runner_fixed.py:
- Loads A* route (EPSG:3857 expected; infers if missing)
- Builds buffer around route
- Extracts vector road LineStrings inside buffer
- Samples points along both the route and roads
- Deduplicates with KDTree radius
- Builds Haversine distance matrix + Floyd–Warshall
- Writes buffer/roads/points + .npy matrices
"""

import argparse, time, warnings
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from pathlib import Path
from math import radians, sin, cos, asin
from scipy.spatial import cKDTree

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
        for ls in geom.geoms: coords.extend(ls.coords)
        geom = LineString(coords)
    print(f"✅ Loaded A* route (length ≈ {geom.length:.0f} m)")
    return geom

def buffer_around_line(line, buffer_m=1000):
    return gpd.GeoDataFrame(geometry=[line.buffer(buffer_m)], crs=METRIC)

# ---------------------------------------------------
# Road extraction (vector)
# ---------------------------------------------------
def extract_vector_roads(roads_vector, buffer_geom):
    """Extract road LineStrings within the route buffer (quiet version, no GDAL logs)."""
    import sys
    from contextlib import contextmanager
    import geopandas as gpd

    print(f"📂 Loading vector roads: {roads_vector}")

    @contextmanager
    def silence_gdal():
        from io import StringIO
        saved_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            yield
        finally:
            sys.stderr = saved_stderr

    try:
        with silence_gdal():
            roads = gpd.read_file(roads_vector)
    except Exception as e:
        print(f"❌ Could not load road file: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC)

    if "geometry" not in roads.columns:
        print("❌ Invalid road data — no geometry column found.")
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC)
    roads = roads[["geometry"]].copy()

    if roads.crs is None:
        roads = roads.set_crs(WGS84)
    roads = roads.to_crs(METRIC)

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
    if pts and pts[-1].distance(Point(line.coords[-1])) > 1:
        pts.append(Point(line.coords[-1]))
    return gpd.GeoDataFrame(geometry=pts, crs=METRIC)

def sample_points_on_lines(gdf_lines, spacing):
    all_pts = []
    for geom in gdf_lines.geometry:
        if geom is None or geom.is_empty: continue
        if geom.geom_type == "LineString":
            all_pts.append(sample_points_along(geom, spacing))
        elif geom.geom_type == "MultiLineString":
            for ls in geom.geoms:
                all_pts.append(sample_points_along(ls, spacing))
    if not all_pts:
        return gpd.GeoDataFrame(geometry=[], crs=METRIC)
    return gpd.GeoDataFrame(pd.concat(all_pts, ignore_index=True), crs=METRIC)

def merge_nearby(points_gdf, rad_m):
    if len(points_gdf) == 0: return points_gdf
    coords = np.array([[p.x, p.y] for p in points_gdf.geometry])
    tree = cKDTree(coords)
    groups = tree.query_ball_tree(tree, rad_m)
    seen, keep = set(), []
    for i, g in enumerate(groups):
        if i in seen: continue
        seen.update(g); keep.append(i)
    cleaned = points_gdf.iloc[keep].copy()
    print(f"🧹 Dedup {len(points_gdf)} → {len(cleaned)} pts (r={rad_m} m)")
    return cleaned.reset_index(drop=True)

# ---------------------------------------------------
# Runner
# ---------------------------------------------------
def run_fw_astar_vector(
    route_geojson="data/outputs/astar_path.geojson",
    roads_vector="data/processed/qc_roads_major_edges.geojson",
    buffer_m=2000,
    spacing_m=600,
    merge_radius_m=450,
    output_dir="data/outputs/floyd_warshall"
):
    """
    Vector-based FW pipeline for A* path, mirroring fw_jps_runner_fixed.
    """
    print("🚆 FW (A* + Vector Roads v5.3)")
    print(f"Route: {route_geojson}")
    print(f"Roads: {roads_vector}")
    print(f"Buffer: ±{buffer_m} m | Spacing: {spacing_m} m | Merge r: {merge_radius_m} m")

    t0 = time.perf_counter()

    # 1) Load A* route + buffer
    route = load_astar_route(route_geojson)
    buffer_gdf = buffer_around_line(route, buffer_m)

    # 2) Extract vector roads within buffer
    roads_gdf = extract_vector_roads(roads_vector, buffer_gdf)

    # 3) Ensure the A* route is included as a line
    astar_gdf = gpd.GeoDataFrame(geometry=[route], crs=METRIC)
    roads_plus = pd.concat([roads_gdf, astar_gdf], ignore_index=True)
    roads_plus = gpd.GeoDataFrame(roads_plus, geometry="geometry", crs=METRIC)
    print(f"🛤️ Added A* route (total lines: {len(roads_plus)})")

    # 4) Sample points (A* line and roads)
    astar_pts = sample_points_along(route, spacing_m)
    road_pts = sample_points_on_lines(roads_plus, spacing_m)
    print(f"🟨 A* pts: {len(astar_pts)} | 🛣️ Road pts: {len(road_pts)}")

    # 5) Merge & deduplicate
    all_pts = gpd.GeoDataFrame(pd.concat([astar_pts, road_pts], ignore_index=True), crs=METRIC)
    merged = merge_nearby(all_pts, merge_radius_m)
    if len(merged) == 0:
        print("❌ No points generated."); return

    # 6) Distance + FW
    pts_wgs = merged.to_crs(WGS84)
    coords = [(p.x, p.y) for p in pts_wgs.geometry]
    n = len(coords)
    D = np.full((n, n), np.inf); np.fill_diagonal(D, 0)
    for i in range(n):
        for j in range(i + 1, n):
            dij = _haversine_m(coords[i], coords[j])
            D[i, j] = D[j, i] = dij
    FW = floyd_warshall_numpy(D)

    # 7) Write outputs
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_astar_buffer.geojson", driver="GeoJSON")
    roads_plus.to_crs(WGS84).to_file(out / "fw_astar_roads.geojson", driver="GeoJSON")
    pts_wgs.to_file(out / "fw_astar_points.geojson", driver="GeoJSON")
    np.save(out / "fw_astar_D.npy", D)
    np.save(out / "fw_astar_FW.npy", FW)

    # 8) Summary
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print("\n✅ Summary (A* + Vector)")
    print(f"🟨 A* pts: {len(astar_pts)} | 🛣️ Road pts: {len(road_pts)} | 🧹 Unique: {len(merged)}")
    print(f"📐 Matrix {D.shape} | ⏱ {elapsed_ms:.2f} ms | 📂 {out}")
    print("===================================")

# ---------------------------------------------------
# Entry
# ---------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/astar_path.geojson")
    ap.add_argument("--roads_vector", default="data/processed/qc_roads_major_edges.geojson")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=600)
    ap.add_argument("--merge_radius_m", type=float, default=450)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    args = ap.parse_args()

    run_fw_astar_vector(**vars(args))
