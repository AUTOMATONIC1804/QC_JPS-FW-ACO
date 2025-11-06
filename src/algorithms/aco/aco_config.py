"""
aco_config.py
--------------
Global ACO configuration for the FW/JPS intersection-aware pipeline.
These values are read by the runner and merged into ACOPathParams.

Tuning notes:
- alpha/beta control pheromone vs. heuristic (POI/Distance) influence.
- w_poi vs. w_dist trade off POI attraction against total path length.
- detour_* gates when POI-motivated detours are allowed.
"""

ACO_CONFIG = {
    # === ACO meta ===
    "alpha": 1.0,           # pheromone weight
    "beta": 2.0,            # heuristic weight
    "rho": 0.40,            # evaporation rate
    "q": 1.0,               # deposit factor
    "n_ants": 40,           # ants per iteration
    "n_iter": 50,           # total iterations

    # === Fitness weights ===
    "w_poi": 1.0,           # benefit weight for POI influence
    "w_dist": 0.05,         # penalty weight for total distance (meters)

    # === Station spacing & candidate search ===
    "min_station_spacing_m": 1000.0,  # ≥1 km between chosen stations (by FW graph distance)
    "local_search_m": 5000.0,         # candidate pool radius from current node (FW distance)

    # === POI influence (from qc_pois_final_scored.geojson) ===
    "poi_radius_m": 1000.0,           # nodes sample POIs within 1 km radius
    "poi_decay_rate": 0.001,          # exp(-decay * distance[m]) weight

    # === Intersection-aware detour policy ===
    "intersection_snap_m": 30.0,      # node is “at an intersection” within this tolerance (m)
    "detour_max_extra_m": 1200.0,     # allow at most this much extra detour distance (m)
    "detour_min_gain_ratio": 0.25,    # min POI gain per km of extra distance
}
