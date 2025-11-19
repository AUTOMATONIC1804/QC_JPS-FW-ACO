"""
fw_jps_runner.py
-------------------------------------------
Uses JPS path + vector-based roads (GeoJSON) to generate
Floyd–Warshall-ready points and matrices — now WITHOUT Haversine.
Distances are computed by running JPS on a raster grid (tif_path)
for every sampled-point pair.

Pipeline kept intact:
  route -> buffer -> clip roads -> sample -> dedupe -> SAVE (buffer/roads/points)
Then:
  points (EPSG:3857) -> pixel coords -> JPS per pair -> D (meters)
Optionally:
  run FW on D (off by default)
"""

import argparse, time, numpy as np, geopandas as gpd, pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from pathlib import Path
from scipy.spatial import cKDTree
from contextlib import contextmanager
import sys
from io import StringIO
from itertools import combinations
from math import hypot
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"

# ===================================================
# (Optional) FW over JPS distances
# ===================================================
def floyd_warshall_numpy(D: np.ndarray) -> np.ndarray:
    dist = D.copy(); n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist

# ===================================================
# Geo helpers
# ===================================================
def load_jps_route(route_geojson):
    gdf = gpd.read_file(route_geojson)
    gdf = gdf.set_crs(METRIC) if gdf.crs is None else gdf.to_crs(METRIC)
    geom = unary_union(gdf.geometry.values)
    if isinstance(geom, MultiLineString):
        coords = []
        for ls in geom.geoms:
            coords.extend(ls.coords)
        geom = LineString(coords)
    print(f"Loaded JPS route (length ≈ {geom.length:.0f} m)")
    return geom

def buffer_around_line(line, buffer_m=1000):
    return gpd.GeoDataFrame(geometry=[line.buffer(buffer_m)], crs=METRIC)

# ---------------------------------------------------
# Road extraction (quiet GDAL)
# ---------------------------------------------------
@contextmanager
def silence_gdal():
    saved_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = saved_stderr

def extract_vector_roads(roads_vector, buffer_geom):
    print(f"Loading vector roads: {roads_vector}")
    try:
        with silence_gdal():
            roads = gpd.read_file(roads_vector)
    except Exception as e:
        print(f"Could not load road file: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC)
    if "geometry" not in roads.columns:
        print("Invalid road data — no geometry column found.")
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC)

    roads = roads[["geometry"]].copy()
    if roads.crs is None:
        roads = roads.set_crs(WGS84)
    roads = roads.to_crs(METRIC)

    buf = buffer_geom.to_crs(METRIC)
    clipped = gpd.overlay(roads, buf, how="intersection")
    clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    clipped = clipped[clipped.geometry.length > 10]
    print(f"Extracted {len(clipped)} road segments inside buffer.")
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

def merge_nearby(points, rad):
    if len(points) == 0: return points
    coords = np.array([[p.x, p.y] for p in points.geometry])
    tree = cKDTree(coords)
    groups = tree.query_ball_tree(tree, rad)
    seen, keep = set(), []
    for i, g in enumerate(groups):
        if i in seen: continue
        seen.update(g)
        keep.append(i)
    cleaned = points.iloc[keep].copy()
    print(f"Dedup {len(points)} → {len(cleaned)} pts (r={rad} m)")
    return cleaned.reset_index(drop=True)

def dedup_tiny(points_gdf, tol_m=1.0):
    if len(points_gdf) == 0: return points_gdf
    coords = np.array([[p.x, p.y] for p in points_gdf.geometry])
    tree = cKDTree(coords)
    groups = tree.query_ball_tree(tree, tol_m)
    seen, keep = set(), []
    for i, g in enumerate(groups):
        if i in seen: continue
        seen.update(g)
        keep.append(i)
    return points_gdf.iloc[keep].reset_index(drop=True)

# ===================================================
# Raster adapter
# ===================================================
class GridAdapter:
    def __init__(self, tif_path: str):
        self.ds = rasterio.open(tif_path)
        self.transform = self.ds.transform
        self.crs = self.ds.crs.to_string() if self.ds.crs else WGS84
        self.array = self.ds.read(1)
        self.nodata = self.ds.nodata
        self.cell_w = hypot(self.transform.a, self.transform.b)
        self.cell_h = hypot(self.transform.d, self.transform.e)
        self.cell_m = (abs(self.cell_w) + abs(self.cell_h)) / 2.0
        if self.crs != METRIC:
            self.to_raster_crs = Transformer.from_crs(METRIC, self.crs, always_xy=True)
        else:
            self.to_raster_crs = None

    def metric_to_rowcol(self, x_m, y_m):
        if self.to_raster_crs:
            x, y = self.to_raster_crs.transform(x_m, y_m)
        else:
            x, y = x_m, y_m
        r, c = rowcol(self.transform, x, y)
        return int(r), int(c)

    def is_in_bounds(self, r, c):
        return 0 <= r < self.array.shape[0] and 0 <= c < self.array.shape[1]

    def is_passable(self, r, c):
        if not self.is_in_bounds(r, c): return False
        v = self.array[r, c]
        if self.nodata is not None and v == self.nodata: return False
        return v > 0

# ===================================================
# JPS distance
# ===================================================
def jps_distance_pixels(grid: GridAdapter, start_rc, goal_rc):
    import numpy as np
    try:
        from src.algorithms.jps.jps_grid import Grid
        from src.algorithms.jps.jps_main import jump_point_search
        g = Grid(grid.array)
        path = jump_point_search(g, start_rc, goal_rc)
        if not path or len(path) < 2:
            return np.inf
        pix_len = 0.0
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            dr, dc = abs(r1 - r0), abs(c1 - c0)
            pix_len += np.sqrt(2) * min(dr, dc) + abs(dr - dc)
        return pix_len
    except Exception as e:
        print(f"JPS failed between {start_rc} → {goal_rc}: {e}")
        return np.inf

def jps_distance_meters(grid, start_xy_m, goal_xy_m):
    r0, c0 = grid.metric_to_rowcol(start_xy_m[0], start_xy_m[1])
    r1, c1 = grid.metric_to_rowcol(goal_xy_m[0], goal_xy_m[1])
    if not (grid.is_passable(r0, c0) and grid.is_passable(r1, c1)):
        return np.inf
    pix_len = jps_distance_pixels(grid, (r0, c0), (r1, c1))
    if not np.isfinite(pix_len):
        return np.inf
    return pix_len * grid.cell_m

# ===================================================
# Runner
# ===================================================
def run_fw_vector(
    route_geojson="data/outputs/jps_path.geojson",
    roads_vector="data/processed/qc_roads_major_edges.geojson",
    buffer_m=2000,
    spacing_m=500,
    merge_radius_m=495,
    output_dir="data/outputs/floyd_warshall",
    tif_path="data/processed/qc_grid_clean.tif",
    run_fw=False,
    max_pairs=None
):
    print("FW (JPS-on-Grid + Vector Roads)")
    t0 = time.perf_counter()

    route = load_jps_route(route_geojson)
    buffer_gdf = buffer_around_line(route, buffer_m)
    roads_gdf = extract_vector_roads(roads_vector, buffer_gdf)

    # Include JPS route geometry in roads
    jps_gdf = gpd.GeoDataFrame(geometry=[route], crs=METRIC)
    roads_gdf = pd.concat([roads_gdf, jps_gdf], ignore_index=True)

    # Sample route and roads
    jps_pts = sample_points_along(route, spacing_m)
    raw_pts = gpd.GeoDataFrame(geometry=[Point(xy) for xy in route.coords], crs=METRIC)
    jps_combined = pd.concat([jps_pts, raw_pts], ignore_index=True)
    road_pts_list = [sample_points_along(r, spacing_m) for r in roads_gdf.geometry if not r.is_empty]
    road_pts_gdf = gpd.GeoDataFrame(pd.concat(road_pts_list, ignore_index=True), crs=METRIC)

    all_pts = gpd.GeoDataFrame(pd.concat([jps_combined, road_pts_gdf], ignore_index=True)).set_crs(METRIC)
    merged = merge_nearby(all_pts, merge_radius_m)
    if len(merged) == 0:
        print("No points generated."); return

    # ===================== NEW: endpoint guard (with labeling + lat/lon) =====================
    clearance_m = 100.0
    start_pt = Point(route.coords[0])
    end_pt = Point(route.coords[-1])

    # Distances to start/end
    d_start = merged.geometry.distance(start_pt)
    d_end = merged.geometry.distance(end_pt)

    # Remove redundant points near start/end
    remove_mask = (d_start < clearance_m) | (d_end < clearance_m)
    pruned = merged.loc[~remove_mask].copy()

    # Add start & end points
    endpoints_gdf = gpd.GeoDataFrame(
        {"geometry": [start_pt, end_pt], "role": ["start", "end"]},
        crs=METRIC
    )

    # Merge and assign roles
    filtered = gpd.GeoDataFrame(
        pd.concat([pruned, endpoints_gdf], ignore_index=True),
        geometry="geometry",
        crs=METRIC
    )
    filtered["role"] = filtered.get("role", "intermediate").fillna("intermediate")

    # Deduplicate
    filtered = dedup_tiny(filtered, tol_m=1.0)

    # --- Add lat/lon columns (WGS84 order: lat, lon) ---
    filtered_wgs = filtered.to_crs(WGS84).copy()
    filtered_wgs["lat"] = filtered_wgs.geometry.y.round(6)
    filtered_wgs["lon"] = filtered_wgs.geometry.x.round(6)

    print(f"Endpoint rule: from {len(merged)} → {len(filtered)} (kept exact start & end)")
    print(f"Start/end points tagged and lat/lon added for export.")
    # =====================================================================

    # ===============================================================
    # Save all outputs
    # ===============================================================
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    buffer_gdf.to_crs(WGS84).to_file(out / "fw_jps_buffer.geojson", driver="GeoJSON")
    roads_gdf.to_crs(WGS84).to_file(out / "fw_jps_roads.geojson", driver="GeoJSON")

    # Save enriched points with roles + lat/lon
    filtered_wgs.to_file(out / "fw_jps_points.geojson", driver="GeoJSON")
    print("Saved buffer/roads/points GeoJSON (with lat/lon + role).")

    # ===============================================================
    # Build JPS-based distance matrix
    # ===============================================================
    print(f"Loading raster grid: {tif_path}")
    grid = GridAdapter(tif_path)

    coords_xy_m = np.column_stack([filtered.geometry.x.values, filtered.geometry.y.values])
    n = coords_xy_m.shape[0]
    D = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)

    idx_pairs = list(combinations(range(n), 2))
    if max_pairs is not None:
        idx_pairs = idx_pairs[:max_pairs]
        print(f"Capping pairs to {len(idx_pairs)} for test runs.")

    t_pairs = time.perf_counter()
    last_print = t_pairs
    for k, (i, j) in enumerate(idx_pairs, 1):
        si, sj = coords_xy_m[i], coords_xy_m[j]
        d_m = jps_distance_meters(grid, (si[0], si[1]), (sj[0], sj[1]))
        D[i, j] = D[j, i] = d_m

        now = time.perf_counter()
        if now - last_print > 2.5:
            print(f"JPS pairs {k}/{len(idx_pairs)} ({100*k/len(idx_pairs):.1f}%)")
            last_print = now

    print(f"JPS distances done for {len(idx_pairs)} pairs.")
    FW = floyd_warshall_numpy(D) if run_fw else D.copy()
    np.save(out / "fw_jps_D.npy", D)
    np.save(out / "fw_jps_FW.npy", FW)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print("\nSummary")
    print(f"Total: {len(merged)} | After endpoint rule: {len(filtered)} (kept start+end)")
    print(f"Matrix {D.shape} | {elapsed_ms:.2f} ms | {out}")
    print("===================================")


# ===================================================
# Entry
# ===================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/jps_path.geojson")
    ap.add_argument("--roads_vector", default="data/processed/qc_roads_major_edges.geojson")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=500)
    ap.add_argument("--merge_radius_m", type=float, default=500)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    ap.add_argument("--tif_path", default="data/processed/qc_grid_clean.tif")
    ap.add_argument("--run_fw", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()
    run_fw_vector(**vars(args))
