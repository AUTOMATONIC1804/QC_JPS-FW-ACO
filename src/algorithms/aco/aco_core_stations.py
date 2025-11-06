"""
aco_core_stations.py
---------------------------------------------------------
Stable ACO for station placement on FW nodes.
- Uses ONLY existing FW nodes (no snapping/new points)
- Enforces start/end
- Optional forward progress along route (by node_measures)
- Enforces min FW spacing between chosen stations (min_spacing_m)
- Elite (global-best) pheromone reinforcement
- Fully safe probability handling (no NaNs/negatives)
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class ACOStationParams:
    alpha: float = 1.0          # pheromone weight
    beta: float = 2.0           # heuristic weight
    rho: float = 0.15           # evaporation rate
    Q: float = 0.5              # deposit factor
    iterations: int = 100
    num_ants: int = 30
    alpha_dist: float = 0.001   # distance penalty (meters)
    beta_poi: float = 1.0       # POI benefit
    gamma_station: float = 0.2  # station count reward
    seed: int = 42              # random seed


class ACOStationOptimizer:
    def __init__(
        self,
        D: np.ndarray,
        poi_scores,                   # array or dict {base,category,proximity}
        params: ACOStationParams,
        start_idx: int,
        end_idx: int,
        k_target: int,
        node_measures: Optional[np.ndarray] = None,  # distance along route for each node (meters)
        min_spacing_m: float = 0.0,                  # e.g., 500
    ):
        np.random.seed(params.seed)

        self.D = np.array(D, dtype=float)
        self.params = params
        self.n = self.D.shape[0]
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.k_target = int(k_target)
        self.min_spacing_m = float(min_spacing_m)

        # --- POI score composition ---
        if isinstance(poi_scores, dict) and "base" in poi_scores:
            base = np.asarray(poi_scores["base"], dtype=float)
            cat = np.asarray(poi_scores.get("category", np.zeros_like(base)), dtype=float)
            prox = np.asarray(poi_scores.get("proximity", np.zeros_like(base)), dtype=float)
            self.poi = base + cat + prox
        else:
            self.poi = np.asarray(poi_scores, dtype=float)

        m = float(np.nanmax(self.poi)) if self.poi.size else 1.0
        self.poi = self.poi / (m if m > 0 else 1.0)

        # --- Heuristic and pheromone ---
        self.heur = 1.0 / (self.D + 1e-9)
        np.fill_diagonal(self.heur, 0.0)
        self.pher = np.ones((self.n, self.n), dtype=float)

        # --- Forward-order setup (based on line measure) ---
        if node_measures is None:
            self.meas = None
        else:
            self.meas = np.asarray(node_measures, dtype=float)

        # --- Tracking ---
        self.best_subset = None
        self.best_fitness = -np.inf
        self.fitness_history = []

    def _valid_next_indices(self, chosen):
        """Return candidate next nodes honoring forward order and spacing."""
        used = set(chosen)
        allowed = set(range(self.n)) - used - {self.start_idx}

        # forward constraint
        if self.meas is not None:
            last = chosen[-1]
            last_m = self.meas[last]
            allowed = {j for j in allowed if self.meas[j] >= last_m - 1e-6}

        # spacing constraint
        if self.min_spacing_m > 0:
            pruned = []
            for j in allowed:
                ok = True
                for c in chosen:
                    d = self.D[c, j]
                    if not np.isfinite(d) or d < self.min_spacing_m:
                        ok = False
                        break
                if ok:
                    pruned.append(j)
            allowed = set(pruned)

        return sorted(list(allowed))

    def _evaluate(self, subset):
        """Fitness = POI benefit − distance penalty + mild count reward."""
        poi_benefit = float(np.sum(self.poi[subset]))
        path_cost = 0.0
        for i in range(len(subset) - 1):
            d = self.D[subset[i], subset[i + 1]]
            path_cost += (d if np.isfinite(d) else 1e9)
        return (
            self.params.beta_poi * poi_benefit
            - self.params.alpha_dist * path_cost
            + self.params.gamma_station * len(subset)
        )

    def _construct_subset(self):
        """Start→…→End, forward-only, spacing-aware, exact k."""
        chosen = [self.start_idx]

        while len(chosen) < max(2, self.k_target - 1):
            candidates = self._valid_next_indices(chosen)
            if not candidates:
                # fallback if no candidates due to spacing
                candidates = sorted(list(set(range(self.n)) - set(chosen) - {self.start_idx}))
                if self.meas is not None:
                    last_m = self.meas[chosen[-1]]
                    candidates = [j for j in candidates if self.meas[j] >= last_m - 1e-6]
                if not candidates:
                    break

            last = chosen[-1]
            tau = np.array([self.pher[last, j] for j in candidates], dtype=float)
            eta = np.array([self.heur[last, j] * (self.poi[j] + 1e-6) for j in candidates], dtype=float)
            probs = np.power(np.clip(tau, 1e-12, None), self.params.alpha) * np.power(np.clip(eta, 0.0, None), self.params.beta)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                probs = np.ones_like(probs)
            probs /= probs.sum()

            next_idx = np.random.choice(candidates, p=probs)
            chosen.append(int(next_idx))

        # append end
        if chosen[-1] != self.end_idx:
            chosen.append(self.end_idx)

        # adjust to exact k
        if len(chosen) > self.k_target:
            keep = [self.start_idx, self.end_idx]
            middle = [i for i in chosen if i not in keep]
            middle_sorted = sorted(middle, key=lambda i: self.poi[i], reverse=True)
            middle_trim = middle_sorted[: max(0, self.k_target - 2)]
            chosen = [self.start_idx] + middle_trim + [self.end_idx]
        elif len(chosen) < self.k_target:
            need = self.k_target - len(chosen)
            pool = [j for j in range(self.n) if j not in chosen and j not in {self.start_idx, self.end_idx}]
            pool = sorted(pool, key=lambda i: self.poi[i], reverse=True)
            for j in pool[:need]:
                chosen.insert(-1, j)

        # final forward ordering (keep start first, end last)
        if self.meas is not None:
            mid = [i for i in chosen if i not in (self.start_idx, self.end_idx)]
            mid_sorted = sorted(mid, key=lambda i: self.meas[i])
            chosen = [self.start_idx] + mid_sorted + [self.end_idx]

        return chosen

    def _update_pheromones(self, ants, fits):
        self.pher *= (1.0 - self.params.rho)

        for ant, f in zip(ants, fits):
            if f <= 0:
                continue
            for i in range(len(ant) - 1):
                a, b = ant[i], ant[i + 1]
                delta = (self.params.Q * f) / (self.D[a, b] + 1e-6)
                self.pher[a, b] += delta
                self.pher[b, a] += delta

        if self.best_subset is not None and self.best_fitness > 0:
            bs = self.best_subset
            for i in range(len(bs) - 1):
                a, b = bs[i], bs[i + 1]
                delta_best = (self.params.Q * self.best_fitness) / (self.D[a, b] + 1e-6)
                self.pher[a, b] += delta_best
                self.pher[b, a] += delta_best

        np.clip(self.pher, 1e-9, 1e6, out=self.pher)

    def run(self):
        for it in range(self.params.iterations):
            ants = [self._construct_subset() for _ in range(self.params.num_ants)]
            fits = [self._evaluate(a) for a in ants]

            best_idx = int(np.argmax(fits))
            if fits[best_idx] > self.best_fitness:
                self.best_fitness = float(fits[best_idx])
                self.best_subset = ants[best_idx]

            self._update_pheromones(ants, fits)
            self.fitness_history.append(self.best_fitness)
            print(f"Iteration {it+1}/{self.params.iterations} → Global best: {self.best_fitness:.4f}")

        return {
            "best_subset": self.best_subset,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
        }


# -------------------------------------------------------------------------
# Example Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    n = 8
    D = np.random.rand(n, n) * 1000
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    poi = {"base": np.random.rand(n), "category": np.random.rand(n), "proximity": np.random.rand(n)}
    params = ACOStationParams(iterations=10, num_ants=5)
    optimizer = ACOStationOptimizer(D, poi, params, start_idx=0, end_idx=7, k_target=5)
    result = optimizer.run()
    print("Best subset:", result["best_subset"])
    print("Best fitness:", result["best_fitness"])
