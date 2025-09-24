# Testing of the shortest path again 
# qc_shortest_path_batch.py
# Batch shortest-path test for all benchmark pairs in QC
# Saves GeoJSON per route + CSV summary

import os
import time
import csv
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString

# import benchmark loader
from qc_load_benchmarks import load_benchmark_points

# ---------- CONFIG ----------
graph_path = r"D:\Quezon_City\data\processed\qc_roads_major.graphml"
output_folder = r"D:\Quezon_City\data\outputs\benchmark_routes"
summary_csv = os.path.join(output_folder, "benchmark_summary.csv")
os.makedirs(output_folder, exist_ok=True)

def haversine_array(lat1, lon1, lat2_arr, lon2_arr):
    """Compute haversine distance from single point to array of coords (meters)."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_node(lat, lon, nodes_gdf):
    """Find nearest node by haversine distance."""
    lat_arr = nodes_gdf['y'].to_numpy()
    lon_arr = nodes_gdf['x'].to_numpy()
    dist = haversine_array(lat, lon, lat_arr, lon_arr)
    idx_sorted = np.argsort(dist)
    return nodes_gdf.index[idx_sorted[0]]


# ---------- MAIN ----------
t0 = time.time()
print("⏳ Loading graph...")
G = ox.load_graphml(graph_path)

# ensure largest connected component
if not nx.is_connected(G.to_undirected()):
    print("⚠️ Graph not fully connected, extracting largest component...")
    largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
    G = G.subgraph(largest_cc).copy()

nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

benchmarks = load_benchmark_points()
names = list(benchmarks.keys())
print(f"✅ Loaded {len(names)} benchmarks: {names}")

# prepare CSV
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end", "length_m", "steps", "runtime_s", "geojson_file"])

    # loop over all pairs
    for i, start_name in enumerate(names):
        for j, end_name in enumerate(names):
            if i == j:
                continue  # skip same point

            start_latlng = (benchmarks[start_name]["lat"], benchmarks[start_name]["lon"])
            end_latlng   = (benchmarks[end_name]["lat"], benchmarks[end_name]["lon"])

            print(f"\n🚏 {start_name} -> 🏁 {end_name}")

            # nearest node lookup
            orig_node = nearest_node(start_latlng[0], start_latlng[1], nodes_gdf)
            dest_node = nearest_node(end_latlng[0], end_latlng[1], nodes_gdf)

            if orig_node == dest_node:
                print("⚠️ Same snapped node, skipping...")
                continue

            # shortest path
            t1 = time.time()
            try:
                route = nx.shortest_path(G, orig_node, dest_node, weight="length")
                length_m = nx.shortest_path_length(G, orig_node, dest_node, weight="length")
                runtime_s = time.time() - t1
                print(f"✅ Path found: {length_m:.2f} m, {len(route)} steps in {runtime_s:.2f} s")
            except nx.NetworkXNoPath:
                print("❌ No path found.")
                continue

            # export GeoJSON
            route_edges_idx = []
            for u, v in zip(route[:-1], route[1:]):
                if (u, v, 0) in edges_gdf.index:
                    route_edges_idx.append((u, v, 0))
                else:
                    matches = [idx for idx in edges_gdf.index if idx[0] == u and idx[1] == v]
                    if not matches:
                        matches = [idx for idx in edges_gdf.index if idx[0] == v and idx[1] == u]
                    if matches:
                        route_edges_idx.append(matches[0])

            if route_edges_idx:
                route_gdf = edges_gdf.loc[route_edges_idx]
            else:
                coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in route]
                line = LineString(coords)
                route_gdf = gpd.GeoDataFrame({'length':[length_m]}, geometry=[line], crs="EPSG:4326")

            geojson_name = f"{start_name.replace(' ', '_')}_to_{end_name.replace(' ', '_')}.geojson"
            geojson_path = os.path.join(output_folder, geojson_name)
            route_gdf.to_file(geojson_path, driver="GeoJSON")

            # write summary row
            writer.writerow([start_name, end_name, length_m, len(route), f"{runtime_s:.2f}", geojson_name])

print(f"\n=== DONE ===")
print(f"All results saved to: {summary_csv}")
print(f"Total runtime: {time.time() - t0:.2f} s")
