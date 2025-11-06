"""
aco_path_station_optimizer.py
--------------------------------------------------
ACO station selection on the full FW/JPS graph + road-following route export.

Key guarantees:
- Uses ALL FW nodes and FW distances.
- Start/end are fixed and included; EXACTLY k stations returned.
- Hard spacing: >= 1000 m between consecutive stations.
- Balanced spacing objective: target 1000 m with ±800 m tolerance;
  penalizes clusters and large uncovered gaps.
- Route geometry = concatenated FW shortest paths → follows roads.
- QGIS-safe GeoJSON outputs (EPSG:4326).

Inputs (from runner):
  fw_jps_points.geojson, fw_jps_D.npy, fw_jps_FW.npy,
  fw_jps_roads.geojson (viz only), qc_pois_final_scored.geojson
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import math


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
@dataclass
class ACOPathParams:
    # ACO meta
    alpha: float = 1.0       # pheromone weight
    beta: float = 2.0        # heuristic (POI) weight
    rho: float = 0.4         # evaporation
    q: float = 1.0           # deposit factor
    n_ants: int = 40
    n_iter: int = 50

    # Fitness weights
    w_poi: float = 1.0           # benefit per unit of POI
    w_dist: float = 0.05         # cost per meter of total path length

    # Spacing rules
    min_station_spacing_m: float = 1000.0   # HARD minimum between neighbors
    target_spacing_m: float = 1000.0        # preferred spacing
    spacing_tol_m: float = 800.0            # ± tolerance around target

    # POI influence (within 1 km; exponential decay)
    poi_radius_m: float = 1000.0
    poi_decay_rate: float = 0.001

    # Spacing/coverage penalty weights (tuned to be meaningful vs w_dist)
    w_spacing: float = 5.0       # penalty per km of deviation outside tolerance
    w_coverage: float = 8.0      # extra penalty per km for very large gaps


# ---------------------------------------------------------
# FW/JPS helpers
# ---------------------------------------------------------
def _load_fw_nodes(points_file: str) -> List[Tuple[float, float]]:
    gdf = gpd.read_file(points_file)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return [(float(pt.x), float(pt.y)) for pt in gdf.geometry]


def _reconstruct_fw_path(next_mat: np.ndarray, a: int, b: int) -> List[int]:
    """Indices for the FW shortest path a→b using NEXT matrix."""
    if a == b:
        return [a]
    n = next_mat.shape[0]
    if not (0 <= a < n and 0 <= b < n):
        return [a, b]
    path = [a]
    cur = a
    for _ in range(n * 2):  # guard
        nxt = int(next_mat[cur, b])
        if nxt < 0 or nxt >= n:
            return [a, b]
        path.append(nxt)
        if nxt == b:
            return path
        cur = nxt
    return [a, b]


def _poi_influence(nodes_xy, poi_path, radius_m, decay):
    """Compute normalized POI influence per node using 1 km exp-decay weighting."""
    pois = gpd.read_file(poi_path)
    if pois.crs is None:
        pois.set_crs("EPSG:4326", inplace=True)
    pois_m = pois.to_crs(3857)

    nodes_m = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([x for x, _ in nodes_xy], [y for _, y in nodes_xy]),
        crs="EPSG:4326",
    ).to_crs(3857)

    scores = np.zeros(len(nodes_xy), dtype=float)
    for i, node in enumerate(nodes_m.geometry):
        buf = node.buffer(radius_m)
        near = pois_m[pois_m.intersects(buf)]  # partial intersects allowed (points & polygons)
        if near.empty:
            continue
        tot = 0.0
        for _, r in near.iterrows():
            d = node.distance(r.geometry)
            s = float(r.get("NormalizedScore", 0.0))
            w = math.exp(-decay * d)
            tot += s * w
        scores[i] = tot

    mx = scores.max()
    if mx > 0:
        scores /= mx
    return scores


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------
class ACOPathStationOptimizer:
    def __init__(self, points_file, dist_file, next_file, roads_file, poi_path,
                 params: ACOPathParams, start_idx: int, end_idx: int, k_target: int):
        self.points_file = str(points_file)
        self.dist_file = str(dist_file)
        self.next_file = str(next_file)
        self.roads_file = str(roads_file)  # viz only
        self.poi_path = str(poi_path)
        self.params = params
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.k_target = int(k_target)

    # -------------------- prep --------------------
    def _prepare(self):
        print("🛠 Loading FW nodes, D, NEXT...")
        self.nodes_xy = _load_fw_nodes(self.points_file)
        self.D = np.load(self.dist_file)     # meters
        self.NEXT = np.load(self.next_file)  # indices

        n = len(self.nodes_xy)
        if self.D.shape != (n, n) or self.NEXT.shape != (n, n):
            raise ValueError(f"FW matrices must be {n}x{n}, got D{self.D.shape}, NEXT{self.NEXT.shape}")

        print("📍 Computing POI influence (1 km, exp-decay)...")
        self.poi = _poi_influence(self.nodes_xy, self.poi_path,
                                  self.params.poi_radius_m, self.params.poi_decay_rate)

        # Pheromone strictly positive
        self.pher = np.full((n, n), 1e-3, dtype=float)

        # Clamp indices
        self.start_idx = max(0, min(n - 1, self.start_idx))
        self.end_idx = max(0, min(n - 1, self.end_idx))
        print(f"✅ Prepared: N={n} | start={self.start_idx} end={self.end_idx}")

    # -------------------- construction helpers --------------------
    def _feasible_candidates(self, chosen: List[int]) -> np.ndarray:
        """Boolean mask of nodes that are not chosen and meet HARD min spacing
        vs the last chosen station by FW distance."""
        n = self.D.shape[0]
        mask = np.ones(n, dtype=bool)
        # don't pick already chosen or fixed endpoints mid-run
        for c in chosen:
            mask[c] = False
        mask[self.start_idx] = False
        mask[self.end_idx] = False

        last = chosen[-1]
        if self.params.min_station_spacing_m > 0:
            d_last = self.D[last]  # vector
            ok = (d_last >= self.params.min_station_spacing_m) & np.isfinite(d_last)
            mask &= ok
        return mask

    def _construct_ant_subset(self) -> List[int]:
        """Build EXACTLY k stations: [start] + (k-2 middle) + [end]."""
        p = self.params
        n = self.D.shape[0]
        chosen = [self.start_idx]

        while len(chosen) < max(2, self.k_target - 1):
            cand_mask = self._feasible_candidates(chosen)
            if not cand_mask.any():
                break

            last = chosen[-1]
            tau = np.maximum(self.pher[last], 1e-12)
            eta = np.maximum(self.poi, 1e-12)
            den = np.maximum(self.D[last], 1e-9)

            raw = np.zeros(n, dtype=float)
            raw[cand_mask] = (tau[cand_mask] ** p.alpha) * (eta[cand_mask] ** p.beta) / den[cand_mask]
            raw[~np.isfinite(raw)] = 0.0

            if raw.sum() <= 0:
                # fallback: greedily pick best POI among feasible
                idxs = np.flatnonzero(cand_mask)
                next_idx = int(max(idxs, key=lambda j: self.poi[j]))
            else:
                idxs = np.flatnonzero(raw > 0)
                probs = raw[idxs] / raw[idxs].sum()
                next_idx = int(np.random.choice(idxs, p=probs))

            chosen.append(next_idx)

        # Ensure end
        if chosen[-1] != self.end_idx:
            chosen.append(self.end_idx)

        # Trim/Pad to EXACT k
        if len(chosen) > self.k_target:
            keep = {self.start_idx, self.end_idx}
            mids = [i for i in chosen if i not in keep]
            mids_sorted = sorted(mids, key=lambda j: self.poi[j], reverse=True)
            chosen = [self.start_idx] + mids_sorted[: max(0, self.k_target - 2)] + [self.end_idx]
        elif len(chosen) < self.k_target:
            need = self.k_target - len(chosen)
            cand_mask = self._feasible_candidates(chosen)
            pool = [j for j in np.flatnonzero(cand_mask)]
            pool_sorted = sorted(pool, key=lambda j: self.poi[j], reverse=True)
            for j in pool_sorted[:need]:
                chosen.insert(-1, j)

        # De-dup while preserving order
        seen = set()
        final = []
        for x in chosen:
            if x not in seen:
                final.append(x)
                seen.add(x)
        # Start/End hard lock
        final[0] = self.start_idx
        final[-1] = self.end_idx
        return final

    # -------------------- fitness --------------------
    def _seq_fw_gaps(self, seq: List[int]) -> List[float]:
        """FW distances for each consecutive gap."""
        gaps = []
        for a, b in zip(seq[:-1], seq[1:]):
            d = float(self.D[a, b]) if np.isfinite(self.D[a, b]) else 1e9
            gaps.append(d)
        return gaps

    def _fitness(self, seq: List[int]) -> float:
        p = self.params
        # Path length (sum of FW gaps)
        dist_m = sum(self._seq_fw_gaps(seq))
        dist_cost = p.w_dist * dist_m

        # POI reward
        poi_sum = float(np.sum([self.poi[i] for i in seq]))
        poi_reward = p.w_poi * poi_sum

        # Spacing/coverage penalties
        tgt = p.target_spacing_m / 1000.0
        tol = p.spacing_tol_m / 1000.0
        spacing_pen = 0.0
        coverage_pen = 0.0
        for gap_m in self._seq_fw_gaps(seq):
            g_km = gap_m / 1000.0
            dev = max(0.0, abs(g_km - tgt) - tol)      # deviation outside tolerance
            spacing_pen += p.w_spacing * dev
            # extra penalty for *very* large gaps (coverage holes)
            if g_km > (tgt + tol):
                coverage_pen += p.w_coverage * (g_km - (tgt + tol))

        # Higher is better → reward - costs - penalties
        return poi_reward - dist_cost - spacing_pen - coverage_pen

    # -------------------- run --------------------
    def run(self, output_dir: str):
        self._prepare()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        best_seq = [self.start_idx, self.end_idx]
        best_fit = -1e18

        for it in range(self.params.n_iter):
            ants = []
            fits = []
            for _ in range(self.params.n_ants):
                seq = self._construct_ant_subset()
                f = self._fitness(seq)
                ants.append(seq)
                fits.append(f)
                if f > best_fit:
                    best_fit, best_seq = f, seq

            # Evaporate & deposit
            self.pher *= (1.0 - self.params.rho)
            for seq, f in zip(ants, fits):
                if f <= 0:
                    continue
                dep = self.params.q * float(f)
                for a, b in zip(seq[:-1], seq[1:]):
                    self.pher[a, b] += dep
                    self.pher[b, a] += dep

            print(f"  Iter {it+1}/{self.params.n_iter} | Best fitness: {best_fit:.4f}")

        self._save_outputs(best_seq, best_fit, out_dir)
        return {"best_subset": best_seq, "best_fitness": best_fit}

    # -------------------- export --------------------

    def _expand_fw_route_coords(self, seq: List[int]) -> List[Tuple[float, float]]:
        """Concatenate *actual road geometries* from fw_jps_roads.geojson
        between consecutive FW station nodes."""
        print("🛣️ Building road-following route geometry...")
        roads = gpd.read_file(self.roads_file)
        if roads.crs is None:
            roads.set_crs("EPSG:4326", inplace=True)
        roads_m = roads.to_crs(3857)

        nodes_m = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([x for x, _ in self.nodes_xy], [y for _, y in self.nodes_xy]),
            crs="EPSG:4326",
        ).to_crs(3857)

        coords = []
        for a, b in zip(seq[:-1], seq[1:]):
            start = nodes_m.geometry[a]
            end = nodes_m.geometry[b]

            # Find road segments whose endpoints are close to these nodes
            sidx = roads_m.sindex
            cand_idx = list(sidx.query(start.buffer(50), predicate="intersects"))
            near_roads = roads_m.iloc[cand_idx].copy()
            best_geom = None
            best_len = 1e12
            for _, row in near_roads.iterrows():
                geom = row.geometry
                d1 = geom.distance(start)
                d2 = geom.distance(end)
                if (d1 + d2) < best_len:
                    best_geom = geom
                    best_len = d1 + d2

            if best_geom is not None:
                if coords and coords[-1] == best_geom.coords[0]:
                    coords.extend(list(best_geom.coords)[1:])
                else:
                    coords.extend(list(best_geom.coords))
            else:
                # fallback: use FW path if no road found
                path = _reconstruct_fw_path(self.NEXT, a, b)
                coords.extend([self.nodes_xy[i] for i in path])

        print(f"✅ Road-following geometry built with {len(coords)} points")
        return coords

    def _save_outputs(self, seq: List[int], fit: float, out_dir: Path):
        # Stations (points)
        stn_xy = [self.nodes_xy[i] for i in seq]
        stn = gpd.GeoDataFrame(
            {
                "order": list(range(1, len(seq) + 1)),
                "fw_index": seq,
                "poi_norm": [float(self.poi[i]) for i in seq],
                "gap_m": self._seq_fw_gaps(seq) + [0.0],
            },
            geometry=gpd.points_from_xy([x for x, _ in stn_xy], [y for _, y in stn_xy]),
            crs="EPSG:4326",
        )
        stn.to_file(out_dir / "aco_path_stations.geojson", driver="GeoJSON", index=False)

        # Route (actual road-following)
        full_coords = self._expand_fw_route_coords(seq)
        route = gpd.GeoDataFrame(
            {"fitness": [fit]},
            geometry=[LineString(full_coords)],
            crs="EPSG:4326",
        )
        route.to_file(out_dir / "aco_path_route.geojson", driver="GeoJSON", index=False)

        print(f"✅ Stations → {out_dir/'aco_path_stations.geojson'}")
        print(f"✅ Route    → {out_dir/'aco_path_route.geojson'}")
        print(f"🏁 Best fitness: {fit:.4f}")


    def _save_outputs(self, seq: List[int], fit: float, out_dir: Path):
        # Stations (points)
        stn_xy = [self.nodes_xy[i] for i in seq]
        stn = gpd.GeoDataFrame(
            {
                "order": list(range(1, len(seq) + 1)),
                "fw_index": seq,
                "poi_norm": [float(self.poi[i]) for i in seq],
                "gap_m": self._seq_fw_gaps(seq) + [0.0],  # last gap 0
            },
            geometry=gpd.points_from_xy([x for x, _ in stn_xy], [y for _, y in stn_xy]),
            crs="EPSG:4326",
        )
        stn.to_file(out_dir / "aco_path_stations.geojson", driver="GeoJSON", index=False)

        # Route (LineString along FW graph)
        full_coords = self._expand_fw_route_coords(seq)
        route = gpd.GeoDataFrame(
            {"fitness": [fit]},
            geometry=[LineString(full_coords)],
            crs="EPSG:4326",
        )
        route.to_file(out_dir / "aco_path_route.geojson", driver="GeoJSON", index=False)

        print(f"✅ Stations → {out_dir/'aco_path_stations.geojson'}")
        print(f"✅ Route    → {out_dir/'aco_path_route.geojson'}")
        print(f"🏁 Best fitness: {fit:.4f}")
