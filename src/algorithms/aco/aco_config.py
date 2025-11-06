"""
aco_config.py
--------------
ACO metaheuristic and station scoring parameters.
"""

ACO_CONFIG = {
    # === ACO Parameters ===
    "alpha": 1.0,         # pheromone importance
    "beta": 3.0,          # heuristic importance (POI influence)
    "rho": 0.25,          # pheromone evaporation rate
    "Q": 1.0,             # pheromone deposit factor
    "iterations": 100,    # total iterations
    "num_ants": 50,       # ants per iteration
    "seed": 42,           # random seed for reproducibility

    # === Station Fitness Weights ===
    "alpha_dist": 0.005,  # minimize distance
    "beta_poi": 1.0,      # maximize POI coverage
    "gamma_station": 0.2, # balance for station count
}
