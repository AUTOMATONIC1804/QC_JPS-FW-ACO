"""
aco_path_station_optimizer.py
--------------------------------------------------
ACO station selection on the full FW/JPS graph + road-following route export.

Key points:
- Uses ALL FW nodes (no corridor filter).
- Distances from FW D.npy (meters). Route segments are FW paths via NEXT.npy.
- Start and end are FIXED and included in the k stations.
- EXACTLY k stations are chosen (>= min spacing by FW graph distance).
- Heuristic favors high-POI nodes that are closer by FW distance.
- Outputs are QGIS-safe (explicit CRS; GeoDataFrames).

Inputs (from runner):
  points_file : fw_jps_points.geojson
  dist_file   : fw_jps_D.npy
  next_file   : fw_jps_FW.npy
  roads_file  : fw_jps_roads.geojson   (viz only; not used for geometry)
  poi_path    : qc_pois_final_scored.geojson
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
    beta: float = 2.0        # heuristic weight
    rho: float = 0.4         # evaporation rate (0..1)
    q: float = 1.0           # deposit factor
    n_ants: int = 40
    n_iter: int = 50

    # Fitness tradeoff
    w_poi: float = 1.0       # reward for POI influence
    w_dist: float = 0.05     # penalty per meter of total path length

    # Station spacing (FW distance between ANY two chosen stations)
    min_station_spacing_m: float = 1000.0

    # POI influence (within 1km; exponential decay)
    poi_radius_m: float = 1000.0
    poi_decay_rate: float = 0.001


# ---------------------------------------------------------
# Basic helpers (FW/JPS)
# ---------------------------------------------------------
def _load_fw_nodes(points_file: str) -> List[Tuple[float, float]]:
    gdf = gpd.read_file(points_file)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return [(float(pt.x), float(pt.y)) for pt in gdf.geometry]


def _reconstruct_fw_path(next_mat: np.ndarray, a: int, b: int) -> List[int]:
    """Indices for the FW shortest path a→b (using NEXT matrix)."""
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
        near = pois_m[pois_m.intersects(buf)]  # counts partial intersects (points & polygons)
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

    # -------------------- construction --------------------
    def _feasible_candidates(self, chosen: List[int]) -> np.ndarray:
        """Boolean mask of nodes that satisfy:
           - not already chosen,
           - not start/end (except final append),
           - ≥ min spacing from ALL chosen (by FW distance, finite).
        """
        n = self.D.shape[0]
        mask = np.ones(n, dtype=bool)
        mask[self.start_idx] = False
        mask[self.end_idx] = False
        for c in chosen:
            mask[c] = False

        # spacing vs ALL chosen
        if self.params.min_station_spacing_m > 0:
            for j in range(n):
                if not mask[j]:
                    continue
                ok = True
                for c in chosen:
                    d = float(self.D[c, j])
                    if not np.isfinite(d) or d < self.params.min_station_spacing_m:
                        ok = False
                        break
                if not ok:
                    mask[j] = False
        return mask

    def _construct_ant_subset(self) -> List[int]:
        """Build EXACTLY k stations: [start] + (k-2 middle) + [end].
           Uses ACO probabilities over ALL FW nodes with spacing constraints.
        """
        p = self.params
        n = self.D.shape[0]
        chosen = [self.start_idx]

        while len(chosen) < max(2, self.k_target - 1):
            cand_mask = self._feasible_candidates(chosen)
            if not cand_mask.any():
                break

            # ACO probabilities: tau^alpha * (poi^beta / dist_from_last)
            last = chosen[-1]
            tau = np.maximum(self.pher[last], 1e-12)
            eta = np.maximum(self.poi, 1e-12)
            den = np.maximum(self.D[last], 1e-9)

            score = np.zeros(n, dtype=float)
            score[cand_mask] = (tau[cand_mask] ** p.alpha) * (eta[cand_mask] ** p.beta) / den[cand_mask]
            score[~np.isfinite(score)] = 0.0

            if score.sum() <= 0:
                # fallback: greedily pick highest POI among feasible
                idxs = np.flatnonzero(cand_mask)
                next_idx = int(max(idxs, key=lambda j: self.poi[j]))
            else:
                idxs = np.flatnonzero(score > 0)
                probs = score[idxs] / score[idxs].sum()
                next_idx = int(np.random.choice(idxs, p=probs))

            chosen.append(next_idx)

        # Append end
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
            # take best remaining by POI respecting spacing
            cand_mask = self._feasible_candidates(chosen)
            pool = [j for j in np.flatnonzero(cand_mask)]
            pool_sorted = sorted(pool, key=lambda j: self.poi[j], reverse=True)
            for j in pool_sorted[:need]:
                chosen.insert(-1, j)

        # Remove duplicates while keeping order
        seen = set()
        final = []
        for x in chosen:
            if x not in seen:
                final.append(x)
                seen.add(x)
        return final

    # -------------------- fitness --------------------
    def _seq_fw_distance(self, seq: List[int]) -> float:
        """Total FW distance when traveling through seq in order."""
        n = self.D.shape[0]
        total = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            if 0 <= a < n and 0 <= b < n and np.isfinite(self.D[a, b]):
                total += float(self.D[a, b])
            else:
                total += 1e9  # unreachable guard
        return total

    def _fitness(self, seq: List[int]) -> float:
        dist = self._seq_fw_distance(seq)
        poi_sum = float(np.sum([self.poi[i] for i in seq]))
        return self.params.w_poi * poi_sum - self.params.w_dist * dist

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
        """Concatenate FW paths between consecutive stations → road-following route."""
        coords = []
        last_xy = None
        for a, b in zip(seq[:-1], seq[1:]):
            seg = _reconstruct_fw_path(self.NEXT, a, b)
            for idx in seg:
                xy = self.nodes_xy[idx]
                if last_xy != xy:
                    coords.append(xy)
                    last_xy = xy
        return coords

    def _save_outputs(self, seq: List[int], fit: float, out_dir: Path):
        # Stations (points)
        stn_xy = [self.nodes_xy[i] for i in seq]
        stn = gpd.GeoDataFrame(
            {
                "order": list(range(1, len(seq) + 1)),
                "fw_index": seq,
                "poi_norm": [float(self.poi[i]) for i in seq],
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
