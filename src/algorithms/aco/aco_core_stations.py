"""
aco_core_stations.py
-------------------------------------------------------
ACO-based optimization for selecting optimal station
locations along a given road network.

Integrates:
- FW distance matrix (road-constrained distances)
- POI influence scoring (using Haversine)
- Fixed start and end nodes (from path file)

Outputs:
- Selected station indices
- Fitness evolution over iterations
"""

import numpy as np
import random
from math import exp
from src.algorithms.aco.aco_utils import haversine_m


# ---------------------------------------------------
# Parameters
# ---------------------------------------------------
class ACOStationParams:
    def __init__(
        self,
        alpha=1.0,            # pheromone importance
        beta=2.0,             # heuristic importance
        rho=0.5,              # evaporation rate
        Q=100.0,              # pheromone deposit constant
        iterations=100,
        num_ants=20,
        alpha_dist=1.0,       # influence of distance cost
        beta_poi=1.0,         # influence of POI attractiveness
        gamma_station=1.0,    # diversity/spacing factor
        seed=42
    ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.iterations = iterations
        self.num_ants = num_ants
        self.alpha_dist = alpha_dist
        self.beta_poi = beta_poi
        self.gamma_station = gamma_station
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)


# ---------------------------------------------------
# ACO for Station Selection
# ---------------------------------------------------
class ACOStationOptimizer:
    def __init__(self, D, poi_scores, params: ACOStationParams, start_idx, end_idx, k_target):
        self.D = D                      # FW distance matrix (road-based)
        self.poi_scores = poi_scores    # POI normalized scores per node
        self.params = params
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.k_target = k_target        # number of stations to select (including start/end)
        self.n = len(D)

        # Initialize pheromone levels between nodes
        self.pheromone = np.ones((self.n, self.n)) * 1e-6
        np.fill_diagonal(self.pheromone, 0)

        # Initialize best
        self.best_subset = None
        self.best_fitness = -np.inf
        self.fitness_history = []


    # ---------------------------------------------------
    # Fitness Evaluation
    # ---------------------------------------------------
    def _evaluate_subset(self, subset):
        """Evaluate fitness using FW distances + POI weighting."""
        D = self.D
        poi_scores = self.poi_scores
        p = self.params

        # Ensure start/end are included
        if self.start_idx not in subset:
            subset = [self.start_idx] + subset
        if self.end_idx not in subset:
            subset = subset + [self.end_idx]

        # Sort to maintain order by index (optional)
        subset = sorted(set(subset))

        # 1️⃣ Total route cost (sum of FW distances between consecutive stations)
        route_cost = 0.0
        for i in range(len(subset) - 1):
            d = D[subset[i], subset[i + 1]]
            if np.isinf(d):  # disconnected case
                route_cost += 1e9
            else:
                route_cost += d

        # 2️⃣ Average POI score of selected nodes
        poi_mean = np.mean([poi_scores[i] for i in subset])

        # 3️⃣ Spacing diversity (maximize average distance between stations)
        dist_pairs = []
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                dist_pairs.append(D[subset[i], subset[j]])
        spacing = np.mean(dist_pairs) if dist_pairs else 0

        # 4️⃣ Combined fitness (maximize POI, spacing; minimize travel cost)
        fitness = (
            p.beta_poi * poi_mean
            + p.gamma_station * (spacing / 1000.0)
            - p.alpha_dist * (route_cost / 1000.0)
        )
        return fitness


    # ---------------------------------------------------
    # Construct Ant Subset (based on pheromone + heuristic)
    # ---------------------------------------------------
    def _construct_subset(self):
        n = self.n
        k = self.k_target
        start, end = self.start_idx, self.end_idx
        p = self.params

        # Precompute heuristic info (POI desirability / avg distance)
        desirability = np.zeros(n)
        for i in range(n):
            avg_dist = np.mean(self.D[i][~np.isinf(self.D[i])])
            desirability[i] = (self.poi_scores[i] + 1e-6) / (avg_dist + 1e-6)

        subset = {start, end}
        available = list(set(range(n)) - subset)

        while len(subset) < k and available:
            probs = []
            for j in available:
                tau = np.mean(self.pheromone[list(subset), j])
                eta = desirability[j]
                probs.append((tau ** p.alpha) * (eta ** p.beta))

            probs = np.array(probs)
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)

            chosen = np.random.choice(available, p=probs)
            subset.add(chosen)
            available.remove(chosen)

        return list(subset)


    # ---------------------------------------------------
    # Main Optimization Loop
    # ---------------------------------------------------
    def run(self):
        p = self.params

        print(f"🐜 Starting ACO with {p.num_ants} ants, {p.iterations} iterations...")
        for it in range(p.iterations):
            ants = [self._construct_subset() for _ in range(p.num_ants)]
            fitnesses = np.array([self._evaluate_subset(a) for a in ants])

            # Update best
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_subset = ants[best_idx]

            self.fitness_history.append(self.best_fitness)

            # Pheromone evaporation
            self.pheromone *= (1 - p.rho)

            # Pheromone deposit based on fitness
            for a, f in zip(ants, fitnesses):
                for i in range(len(a) - 1):
                    i1, i2 = a[i], a[i + 1]
                    if np.isinf(self.D[i1, i2]):  # skip invalid pairs
                        continue
                    delta = p.Q / (self.D[i1, i2] + 1)
                    self.pheromone[i1, i2] += delta * (f / (abs(f) + 1e-6))
                    self.pheromone[i2, i1] = self.pheromone[i1, i2]

            print(f"Iteration {it+1}/{p.iterations} → Best fitness: {self.best_fitness:.4f}")

        return {
            "best_subset": self.best_subset,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
        }
