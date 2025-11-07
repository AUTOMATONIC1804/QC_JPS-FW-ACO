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
from rasterio.transform import rowcol, xy
from pyproj import Transformer

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"

# ===================================================
# (Optional) FW over JPS distances
# ===================================================
def floyd_warshall_numpy(D: np.ndarray) -> np.ndarray:
    dist = D.copy(); n = dist.shape[0]
    for k in range(n): dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
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
        for ls in geom.geoms: coords.extend(ls.coords)
        geom = LineString(coords)
    print(f"✅ Loaded JPS route (length ≈ {geom.length:.0f} m)")
    return geom

def buffer_around_line(line, buffer_m=1000):
    return gpd.GeoDataFrame(geometry=[line.buffer(buffer_m)], crs=METRIC)

# ===================================================
# Road extraction (with GDAL silencer)
# ===================================================
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

# ===================================================
# Sampling + Deduplication
# ===================================================
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

# ===================================================
# Raster ⇄ metric adapters
# ===================================================
class GridAdapter:
    """
    Bridges metric points (EPSG:3857) to raster pixels.
    Assumes passable cells are >0 (or nodata=0). You can tweak is_passable().
    """
    def __init__(self, tif_path: str):
        self.ds = rasterio.open(tif_path)
        self.transform = self.ds.transform
        self.crs = self.ds.crs.to_string() if self.ds.crs else WGS84
        self.array = self.ds.read(1)
        self.nodata = self.ds.nodata
        # Approx cell size in meters (works if raster is in a metric CRS; for geographic, we’ll project points)
        # We compute from transform scale:
        self.cell_w = hypot(self.transform.a, self.transform.b)
        self.cell_h = hypot(self.transform.d, self.transform.e)
        self.cell_m = (abs(self.cell_w) + abs(self.cell_h)) / 2.0
        # Metric <-> raster CRS transformers
        if self.crs != METRIC:
            self.to_raster_crs = Transformer.from_crs(METRIC, self.crs, always_xy=True)
        else:
            self.to_raster_crs = None

    def metric_to_rowcol(self, x_m: float, y_m: float):
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
        # Treat 0 as blocked, >0 as passable by default:
        return v > 0

# ===================================================
# JPS distance adapter (plug YOUR function here)
# ===================================================
def jps_distance_pixels(grid: GridAdapter, start_rc, goal_rc) -> float:
    """
    Compute the path length in pixels between two raster cells using your JPS implementation.
    Integrates with src/algorithms/jps_runner.py.
    """
    import numpy as np

    try:
        from src.algorithms.jps.jps_grid import Grid
        from src.algorithms.jps.jps_main import jump_point_search

        # Wrap grid
        g = Grid(grid.array)

        # Run your JPS (no heuristic arg needed — handled internally)
        path = jump_point_search(g, start_rc, goal_rc)

        # If no path found
        if path is None or len(path) < 2:
            return np.inf

        # Compute pixel length (1 for straight, √2 for diagonal)
        pix_len = 0.0
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            dr, dc = abs(r1 - r0), abs(c1 - c0)
            pix_len += np.sqrt(2.0) if (dr == 1 and dc == 1) else 1.0

        return pix_len

    except Exception as e:
        print(f"⚠️ JPS failed between {start_rc} → {goal_rc}: {e}")
        return np.inf



def jps_distance_meters(grid: GridAdapter, start_xy_m, goal_xy_m) -> float:
    """
    Convert metric coords -> pixels, validate passability, call JPS, convert pixels -> meters.
    """
    r0, c0 = grid.metric_to_rowcol(start_xy_m[0], start_xy_m[1])
    r1, c1 = grid.metric_to_rowcol(goal_xy_m[0], goal_xy_m[1])

    if not (grid.is_passable(r0, c0) and grid.is_passable(r1, c1)):
        return np.inf

    pix_len = jps_distance_pixels(grid, (r0, c0), (r1, c1))
    if not np.isfinite(pix_len):
        return np.inf

    # Convert pixel length to meters using cell metric size.
    return pix_len * grid.cell_m

# ===================================================
# Runner
# ===================================================
def run_fw_vector(
    route_geojson="data/outputs/jps_path.geojson",
    roads_vector="data/processed/qc_roads_major_edges.geojson",
    buffer_m=2000,
    spacing_m=600,
    merge_radius_m=450,
    output_dir="data/outputs/floyd_warshall",
    tif_path="data/processed/qc_grid_clean.tif",
    run_fw=False,
    max_pairs=None
):
    """
    max_pairs: for quick tests, cap number of unique pairs considered (upper triangle).
    run_fw: if True, run FW over the JPS distance matrix (usually unnecessary).
    """
    print("🚆 FW (JPS-on-Grid + Vector Roads) — Haversine removed")
    t0 = time.perf_counter()

    # 1) Load + buffer + roads
    route = load_jps_route(route_geojson)
    buffer_gdf = buffer_around_line(route, buffer_m)
    roads_gdf = extract_vector_roads(roads_vector, buffer_gdf)

    # Ensure JPS path is always included
    jps_gdf = gpd.GeoDataFrame(geometry=[route], crs=METRIC)
    roads_gdf = pd.concat([roads_gdf, jps_gdf], ignore_index=True)
    print(f"🛤️ Added JPS route (total lines: {len(roads_gdf)})")

    # 2) Sample points
    jps_pts = sample_points_along(route, spacing_m)
    print(f"🟨 JPS pts: {len(jps_pts)}")
    road_pts_list = [sample_points_along(r, spacing_m) for r in roads_gdf.geometry if not r.is_empty]
    road_pts_gdf = gpd.GeoDataFrame(pd.concat(road_pts_list, ignore_index=True), crs=METRIC)
    print(f"🛣️ Road pts: {len(road_pts_gdf)}")

    # 3) Merge/dedup
    all_pts = gpd.GeoDataFrame(pd.concat([jps_pts, road_pts_gdf], ignore_index=True)).set_crs(METRIC)
    merged = merge_nearby(all_pts, merge_radius_m)
    if len(merged) == 0:
        print("❌ No points generated."); return

    # 4) Save *spatial* outputs first (as requested)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_jps_buffer.geojson", driver="GeoJSON")
    roads_gdf.to_crs(WGS84).to_file(out / "fw_jps_roads.geojson", driver="GeoJSON")
    merged.to_crs(WGS84).to_file(out / "fw_jps_points.geojson", driver="GeoJSON")
    print("💾 Saved buffer/roads/points GeoJSON.")

    # 5) Build JPS-based distance matrix over raster grid
    print(f"🗺️ Loading raster grid: {tif_path}")
    grid = GridAdapter(tif_path)

    # Prepare coordinates in meters (EPSG:3857)
    coords_xy_m = np.column_stack([merged.geometry.x.values, merged.geometry.y.values])
    n = coords_xy_m.shape[0]
    D = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)

    # Upper-tri pairs
    idx_pairs = list(combinations(range(n), 2))
    if max_pairs is not None:
        idx_pairs = idx_pairs[:max_pairs]
        print(f"🔬 Capping pairs to {len(idx_pairs)} for test runs.")

    t_pairs = time.perf_counter()
    last_print = t_pairs
    for k, (i, j) in enumerate(idx_pairs, 1):
        si = coords_xy_m[i]; sj = coords_xy_m[j]
        d_m = jps_distance_meters(grid, (si[0], si[1]), (sj[0], sj[1]))
        D[i, j] = D[j, i] = d_m

        # lightweight progress
        now = time.perf_counter()
        if now - last_print > 2.5:
            pct = 100.0 * k / len(idx_pairs)
            print(f"  ⏳ JPS pairs {k}/{len(idx_pairs)} ({pct:.1f}%)")
            last_print = now

    print(f"✅ JPS distances done for {len(idx_pairs)} pairs in {(time.perf_counter()-t_pairs):.2f}s")

    # 6) (Optional) FW on top of JPS distances
    if run_fw:
        print("♻️ Running Floyd–Warshall over JPS distances (usually unnecessary)…")
        FW = floyd_warshall_numpy(D)
    else:
        FW = D.copy()  # already shortest by construction

    # 7) Save matrices
    np.save(out / "fw_jps_D.npy", D)
    np.save(out / "fw_jps_FW.npy", FW)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print("\n✅ Summary")
    print(f"🟨 JPS pts: {len(jps_pts)} | 🛣️ Road pts: {len(road_pts_gdf)} | 🧹 Unique: {len(merged)}")
    print(f"📐 Matrix {D.shape} | ⏱ {elapsed_ms:.2f} ms | 📂 {out}")
    print("===================================")

# ===================================================
# Entry
# ===================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/jps_path.geojson")
    ap.add_argument("--roads_vector", default="data/processed/qc_roads_major_edges.geojson")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=600)
    ap.add_argument("--merge_radius_m", type=float, default=450)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    ap.add_argument("--tif_path", default="data/processed/qc_grid_clean.tif")
    ap.add_argument("--run_fw", action="store_true", help="Optionally run Floyd–Warshall over JPS distances")
    ap.add_argument("--max_pairs", type=int, default=None, help="Cap pairs for quick tests")
    args = ap.parse_args()
    run_fw_vector(**vars(args))
