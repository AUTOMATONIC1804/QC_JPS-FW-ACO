"""
aco_config.py
--------------
Global configuration for the ACO station optimization system.
Applies to all variants (JPS, A*, Dijkstra).
"""

ACO_CONFIG = {
    # === ACO Metaheuristic Parameters ===
    "alpha": 1.0,        # pheromone importance
    "beta": 2.0,         # heuristic importance
    "rho": 0.1,          # pheromone evaporation rate
    "Q": 100.0,          # pheromone deposit constant
    "iterations": 100,   # number of iterations per run
    "num_ants": 50,      # number of ants per iteration
    "seed": 42,          # random seed for reproducibility

    # === Station Fitness Weights ===
    "alpha_dist": 0.6,   # weight for minimizing total distance
    "beta_poi": 0.4,     # weight for maximizing POI coverage
    "gamma_station": 0.3,# penalty for deviating from target k

    # === Data Paths ===
    "data_dir": "data/processed/",
    "output_dir": "data/output/"
}
