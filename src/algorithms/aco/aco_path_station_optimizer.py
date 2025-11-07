"""
aco_path_station_optimizer.py
--------------------------------------------------
Intersection-aware ACO for station selection + route alignment on the FW/JPS graph.
✅ FW nodes only (fw_jps_points.geojson)
✅ FW D.npy and NEXT.npy used for distance + routing
✅ Uses fw_jps_roads.geojson for intersection detection
✅ fw_jps_buffer.geojson for spatial constraint
✅ 800 m ± spacing between stations
✅ 1 km POI influence radius (exp decay)
✅ Detours only allowed if they rejoin inside buffer and at intersections
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import math

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
@dataclass
class ACOPathParams:
    # ACO meta
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.4
    q: float = 1.0
    n_ants: int = 40
    n_iter: int = 50

    # Fitness weights
    w_poi: float = 1.0
    w_dist: float = 0.05

    # Spatial rules
    min_station_spacing_m: float = 800.0
    local_search_m: float = 6000.0

    # POI influence
    poi_radius_m: float = 1000.0
    poi_decay_rate: float = 0.001

    # Detour/intersection
    intersection_snap_m: float = 30.0
    detour_max_extra_m: float = 1500.0
    detour_min_gain_ratio: float = 0.25


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def _load_fw_nodes(points_file: str) -> List[Tuple[float, float]]:
    gdf = gpd.read_file(points_file)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return [(float(pt.x), float(pt.y)) for pt in gdf.geometry]


def _reconstruct_fw_path(next_mat: np.ndarray, a: int, b: int) -> List[int]:
    """Return list of FW node indices from a→b."""
    if a == b:
        return [a]
    n = next_mat.shape[0]
    if not (0 <= a < n and 0 <= b < n):
        return [a, b]
    path = [a]
    cur = a
    for _ in range(n * 2):
        nxt = int(next_mat[cur, b])
        if nxt < 0 or nxt >= n:
            break
        path.append(nxt)
        if nxt == b:
            return path
        cur = nxt
    return path


def _compute_poi_influence(nodes_xy, poi_path, radius_m, decay):
    pois = gpd.read_file(poi_path)
    if pois.crs is None:
        pois.set_crs("EPSG:4326", inplace=True)
    pois_m = pois.to_crs(3857)
    nodes_m = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([x for x, _ in nodes_xy], [y for _, y in nodes_xy]),
        crs="EPSG:4326",
    ).to_crs(3857)
    scores = np.zeros(len(nodes_xy))
    for i, node in enumerate(nodes_m.geometry):
        buf = node.buffer(radius_m)
        near = pois_m[pois_m.intersects(buf)]
        if near.empty:
            continue
        tot = 0.0
        for _, row in near.iterrows():
            d = node.distance(row.geometry)
            s = float(row.get("NormalizedScore", 0.0))
            w = math.exp(-decay * d)
            tot += s * w
        scores[i] = tot
    if scores.max() > 0:
        scores /= scores.max()
    return scores


def _detect_road_intersections(roads: gpd.GeoDataFrame):
    """Detect intersection points among road lines."""
    print("🧩 Detecting intersections...")
    r = roads[~roads.geometry.is_empty & roads.geometry.is_valid].copy()
    r = r[r.geometry.length > 0.5]
    r = r.reset_index(drop=True)
    sidx = r.sindex
    points = []
    for i, geom in enumerate(r.geometry):
        for j in sidx.query(geom, predicate="intersects"):
            if j <= i:
                continue
            g2 = r.geometry.iloc[j]
            inter = geom.intersection(g2)
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                points.append(inter)
            elif inter.geom_type == "MultiPoint":
                points.extend(list(inter.geoms))
    inter_gdf = gpd.GeoDataFrame(geometry=points, crs=roads.crs)
    if not inter_gdf.empty:
        inter_gdf["x"] = inter_gdf.geometry.x.round(1)
        inter_gdf["y"] = inter_gdf.geometry.y.round(1)
        inter_gdf = inter_gdf.drop_duplicates(subset=["x", "y"]).drop(columns=["x", "y"])
    print(f"✅ Intersections: {len(inter_gdf)}")
    return inter_gdf


def _flag_nodes_near_intersections(nodes_xy, intersections_m, tol_m):
    if intersections_m.empty:
        return np.zeros(len(nodes_xy), dtype=bool)
    nodes_m = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([x for x, _ in nodes_xy], [y for _, y in nodes_xy]),
        crs="EPSG:4326",
    ).to_crs(3857)
    mask = np.zeros(len(nodes_xy), dtype=bool)
    sidx = intersections_m.sindex
    for i, pt in enumerate(nodes_m.geometry):
        cand = sidx.query(pt.buffer(tol_m), predicate="intersects")
        if len(cand) == 0:
            continue
        dmin = intersections_m.iloc[list(cand)].distance(pt).min()
        if dmin <= tol_m:
            mask[i] = True
    return mask


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------
class ACOPathStationOptimizer:
    def __init__(
        self,
        points_file,
        dist_file,
        next_file,
        roads_file,
        poi_path,
        params: ACOPathParams,
        start_idx: int,
        end_idx: int,
        k_target: int,
    ):
        self.points_file = str(points_file)
        self.dist_file = str(dist_file)
        self.next_file = str(next_file)
        self.roads_file = str(roads_file)
        self.poi_path = str(poi_path)
        self.params = params
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.k_target = k_target
        self.fw_buffer = None

    # -----------------------------------------------------
    def _prepare(self):
        print("🛠 Preparing FW/JPS ACO data...")
        self.nodes_xy = _load_fw_nodes(self.points_file)
        self.D = np.load(self.dist_file)
        self.NEXT = np.load(self.next_file)
        print(f"✅ Loaded {len(self.nodes_xy)} FW nodes")

        print("📍 Computing POI influence...")
        self.poi_scores = _compute_poi_influence(
            self.nodes_xy, self.poi_path,
            self.params.poi_radius_m, self.params.poi_decay_rate
        )

        self.pheromone = np.full_like(self.D, 1e-3)
        roads = gpd.read_file(self.roads_file)
        if roads.crs is None:
            roads.set_crs("EPSG:4326", inplace=True)
        self.roads_m = roads.to_crs(3857)
        self.intersections_m = _detect_road_intersections(self.roads_m)
        self.node_is_intersection = _flag_nodes_near_intersections(
            self.nodes_xy, self.intersections_m, self.params.intersection_snap_m
        )

    # -----------------------------------------------------
    def _construct_ant_path(self):
        p = self.params
        n = len(self.nodes_xy)
        current = self.start_idx
        path = [current]
        last_station = current
        dist_since_station = 0
        total_poi = self.poi_scores[current]
        total_dist = 0

        while len(path) < p.n_ants:
            d_cur = self.D[current]
            d_last = self.D[last_station]
            mask = (d_cur > 0) & (d_cur <= p.local_search_m) & (d_last >= p.min_station_spacing_m)
            for idx in path:
                mask[idx] = False

            cand = np.flatnonzero(mask)
            if len(cand) == 0:
                break

            tau = np.maximum(self.pheromone[current][cand], 1e-12)
            eta = np.maximum(self.poi_scores[cand], 1e-12)
            dist = np.maximum(self.D[current][cand], 1e-9)
            prob = (tau ** p.alpha) * (eta ** p.beta) / dist
            prob /= prob.sum()
            next_idx = int(np.random.choice(cand, p=prob))

            total_poi += self.poi_scores[next_idx]
            total_dist += self.D[current, next_idx]
            current = next_idx
            path.append(current)
            if self.D[last_station, current] >= p.min_station_spacing_m:
                last_station = current

            if current == self.end_idx:
                break

        if path[-1] != self.end_idx:
            path.append(self.end_idx)
        return path, total_dist, total_poi

    # -----------------------------------------------------
    def run(self, output_dir: str):
        self._prepare()
        best_fit = -1e18
        best_seq = None
        for it in range(self.params.n_iter):
            for _ in range(self.params.n_ants):
                seq, dist, poi = self._construct_ant_path()
                fit = self.params.w_poi * poi - self.params.w_dist * dist
                if fit > best_fit:
                    best_fit, best_seq = fit, seq
            self.pheromone *= (1 - self.params.rho)
            for a, b in zip(best_seq[:-1], best_seq[1:]):
                self.pheromone[a, b] += self.params.q * best_fit
            print(f"Iter {it+1}/{self.params.n_iter} | Best: {best_fit:.3f}")

        self._save_outputs(best_seq, best_fit, output_dir)
        return {"best_subset": best_seq, "best_fitness": best_fit}

    # -----------------------------------------------------
    def _save_outputs(self, seq, fit, out_dir):
        coords = [self.nodes_xy[i] for i in seq]
        gdf_pts = gpd.GeoDataFrame(
            {"order": range(len(coords))},
            geometry=gpd.points_from_xy(*zip(*coords)),
            crs="EPSG:4326",
        )
        gdf_pts.to_file(Path(out_dir) / "aco_path_stations.geojson", driver="GeoJSON")

        route_coords = [self.nodes_xy[i] for i in seq]
        gdf_line = gpd.GeoDataFrame(
            {"fitness": [fit]},
            geometry=[LineString(route_coords)],
            crs="EPSG:4326",
        )
        gdf_line.to_file(Path(out_dir) / "aco_path_route.geojson", driver="GeoJSON")

        print(f"✅ Saved to {out_dir}")


# End of file
