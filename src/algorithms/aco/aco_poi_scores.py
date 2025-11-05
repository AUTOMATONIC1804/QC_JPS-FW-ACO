"""
aco_poi_scores.py
-----------------
Computes per-node POI influence scores using:
- A 1 km spatial buffer around each FW node
- Exponential distance decay for weighting
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# === Paths ===
POI_PATH = r"D:\Quezon_City\data\processed\qc_pois_final_scored.geojson"
NODE_PATH = r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson"
OUTPUT_PATH = NODE_PATH  # overwrite or duplicate

BUFFER_M = 1000   # 1 km buffer around each node
DECAY_RATE = 0.001  # exponential distance decay


def haversine_m(p1, p2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371008.8
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1)*cos(lat2)*sin(dlon / 2)**2
    return R * 2 * asin(min(1.0, sqrt(a)))


def proximity_weight(distance_m):
    """Exponential decay weighting."""
    return np.exp(-DECAY_RATE * distance_m)


def compute_poi_influence(nodes_gdf, pois_gdf):
    # Work in projected CRS (meters)
    nodes_gdf = nodes_gdf.to_crs(3857)
    pois_gdf = pois_gdf.to_crs(3857)
    nodes_gdf["poi_score"] = 0.0

    print(f"Computing POI influence using {BUFFER_M/1000:.1f} km buffer per node...")

    for i, node in enumerate(nodes_gdf.itertuples(), 1):
        node_geom = node.geometry
        buffer_geom = node_geom.buffer(BUFFER_M)
        nearby_pois = pois_gdf[pois_gdf.intersects(buffer_geom)]

        if nearby_pois.empty:
            continue

        total_score = 0.0
        for _, poi in nearby_pois.iterrows():
            d = node_geom.distance(poi.geometry)
            score = poi.get("NormalizedScore", 0)
            total_score += score * proximity_weight(d)

        nodes_gdf.at[node.Index, "poi_score"] = total_score

        if i % 100 == 0 or i == len(nodes_gdf):
            print(f"  → Processed {i}/{len(nodes_gdf)} nodes")

    # Normalize POI scores (0–1)
    max_val = nodes_gdf["poi_score"].max()
    if max_val > 0:
        nodes_gdf["poi_score"] = nodes_gdf["poi_score"] / max_val

    return nodes_gdf.to_crs(4326)  # back to lon/lat for consistency


def main():
    print("📍 Loading POIs and FW nodes...")
    pois = gpd.read_file(POI_PATH)
    nodes = gpd.read_file(NODE_PATH)
    print(f"  → POIs: {len(pois)}, Nodes: {len(nodes)}")

    updated = compute_poi_influence(nodes, pois)
    updated.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"✅ Saved updated nodes with POI influence → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
