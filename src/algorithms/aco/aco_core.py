# src/algorithms/aco/aco_core.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Dict
import numpy as np
import math
import random


Route = List[int]
CostFn = Callable[[Route], Tuple[float, bool]]  # returns (cost, is_valid)
HeuristicFn = Callable[[int, int], float]       # eta(i,j)


@dataclass
class ACOConfig:
    n_ants: int = 30
    n_iterations: int = 100
    alpha: float = 1.0        # pheromone influence
    beta: float = 3.0         # heuristic influence
    rho: float = 0.5          # evaporation rate (0..1)
    Q: float = 1.0            # pheromone deposit constant
    start_idx: int = 0
    end_idx: int = 1
    allow_revisit: bool = False
    max_steps: Optional[int] = None          # safety cap; defaults to n_nodes*2
    seed: Optional[int] = None               # reproducibility
    pheromone_init: float = 1.0
    pheromone_min: float = 1e-6
    pheromone_max: float = 1e6


class AntColony:
    """
    Generic Ant Colony Optimization engine operating on an effort/cost matrix.
    - Lower matrix values = better (we minimize).
    - Infeasible edges should be set to np.inf (or very large).

    You can inject:
      - route_cost_fn(route) -> (cost, valid): to add spacing/coverage penalties, etc.
      - heuristic_fn(i, j)   -> eta_ij: default is 1/(eps + cost_ij).

    Typical use:
        ac = AntColony(E, ACOConfig(...))
        best_route, best_cost, history = ac.run(route_cost_fn=my_cost_fn)
    """

    def __init__(
        self,
        effort_matrix: np.ndarray,
        config: ACOConfig,
        heuristic_fn: Optional[HeuristicFn] = None,
    ):
        self.C = np.asarray(effort_matrix, dtype=float)
        if self.C.ndim != 2 or self.C.shape[0] != self.C.shape[1]:
            raise ValueError("effort_matrix must be a square 2D array")

        self.n = self.C.shape[0]
        self.cfg = config
        if self.cfg.max_steps is None:
            self.cfg.max_steps = self.n * 2

        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Initialize pheromones
        self.tau = np.full_like(self.C, fill_value=self.cfg.pheromone_init, dtype=float)

        # Default heuristic: eta = 1 / (epsilon + cost)
        eps = 1e-9
        if heuristic_fn is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = 1.0 / (eps + self.C)
                eta[np.isinf(self.C)] = 0.0  # no heuristic on infeasible edges
            self.eta_matrix = eta
            self.heuristic_fn = None
        else:
            self.eta_matrix = None
            self.heuristic_fn = heuristic_fn

        # Precompute feasible adjacency mask
        self.feasible = np.isfinite(self.C) & (self.C < np.inf)

    # ------------------------ main loop ------------------------

    def run(
        self,
        route_cost_fn: Optional[CostFn] = None,
    ) -> Tuple[Route, float, Dict[str, List[float]]]:
        """
        Execute ACO and return the best route, its cost, and training history.
        """
        best_route: Route = []
        best_cost: float = math.inf
        history = {"best_cost": [], "mean_cost": []}

        for it in range(self.cfg.n_iterations):
            routes, costs = self._construct_solutions(route_cost_fn)

            # Pheromone evaporation
            self._evaporate()

            # Deposit pheromone for valid routes
            any_valid = False
            for route, cost in zip(routes, costs):
                if math.isfinite(cost):
                    any_valid = True
                    self._deposit(route, cost)

            # Track iteration stats
            if any_valid:
                valid_costs = [c for c in costs if math.isfinite(c)]
                iter_best_cost = min(valid_costs)
                iter_mean_cost = float(np.mean(valid_costs))
            else:
                iter_best_cost = math.inf
                iter_mean_cost = math.inf

            history["best_cost"].append(iter_best_cost)
            history["mean_cost"].append(iter_mean_cost)

            # Update global best
            if any_valid:
                idx = valid_costs.index(iter_best_cost)
                candidate_route = [r for r, c in zip(routes, costs) if math.isfinite(c)][idx]
                if iter_best_cost < best_cost:
                    best_cost = iter_best_cost
                    best_route = candidate_route

            # Optional: bounding pheromone to avoid numeric explosion
            np.clip(self.tau, self.cfg.pheromone_min, self.cfg.pheromone_max, out=self.tau)

        return best_route, best_cost, history

    # ------------------------ construction ------------------------

    def _construct_solutions(
        self,
        route_cost_fn: Optional[CostFn],
    ) -> Tuple[List[Route], List[float]]:
        routes: List[Route] = []
        costs: List[float] = []

        for _ in range(self.cfg.n_ants):
            route = self._build_single_route()
            if route_cost_fn is None:
                # default cost = sum of edge costs if path reaches end
                cost, ok = self._default_route_cost(route)
            else:
                cost, ok = route_cost_fn(route)

            if not ok:
                cost = math.inf

            routes.append(route)
            costs.append(cost)

        return routes, costs

    def _build_single_route(self) -> Route:
        start = self.cfg.start_idx
        end = self.cfg.end_idx

        route: Route = [start]
        visited = set([start])

        current = start
        steps = 0

        while current != end and steps < (self.cfg.max_steps or self.n * 2):
            nbrs = self._allowed_neighbors(current, visited)

            if not nbrs:
                # dead end; stop and return partial route (will be invalid)
                break

            j = self._sample_next(current, nbrs)
            route.append(j)
            if not self.cfg.allow_revisit:
                visited.add(j)
            current = j
            steps += 1

        return route

    def _allowed_neighbors(self, i: int, visited: set[int]) -> List[int]:
        feas = self.feasible[i].copy()
        if not self.cfg.allow_revisit:
            for v in visited:
                feas[v] = False
        feas[i] = False  # no self-loop
        return np.where(feas)[0].tolist()

    def _eta(self, i: int, j: int) -> float:
        if self.heuristic_fn is not None:
            return float(self.heuristic_fn(i, j))
        return float(self.eta_matrix[i, j])

    def _sample_next(self, i: int, neighbors: Sequence[int]) -> int:
        # transition probability p_ij ∝ (tau_ij^alpha) * (eta_ij^beta)
        alpha = self.cfg.alpha
        beta = self.cfg.beta

        tau_i = self.tau[i, neighbors]
        eta_i = np.array([self._eta(i, j) for j in neighbors], dtype=float)

        with np.errstate(over="ignore", invalid="ignore"):
            weights = (np.power(tau_i, alpha)) * (np.power(eta_i, beta))

        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            # fall back: pick neighbor with minimal raw cost
            j = int(neighbors[int(np.argmin(self.C[i, neighbors]))])
            return j

        probs = weights / total
        choice_idx = np.random.choice(len(neighbors), p=probs)
        return int(neighbors[choice_idx])

    # ------------------------ costs & pheromones ------------------------

    def _default_route_cost(self, route: Route) -> Tuple[float, bool]:
        if not route or route[-1] != self.cfg.end_idx:
            return math.inf, False
        # Sum edge costs along route
        total = 0.0
        for u, v in zip(route[:-1], route[1:]):
            c = self.C[u, v]
            if not np.isfinite(c):
                return math.inf, False
            total += c
        return total, True

    def _evaporate(self) -> None:
        self.tau *= (1.0 - self.cfg.rho)

    def _deposit(self, route: Route, cost: float) -> None:
        if cost <= 0 or not math.isfinite(cost):
            return
        deposit = self.cfg.Q / cost
        for u, v in zip(route[:-1], route[1:]):
            self.tau[u, v] += deposit
