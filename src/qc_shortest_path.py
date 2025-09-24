# Testing of a shortest path on major roads in QC
# qc_shortest_path.py
# Robust shortest-path test:
# - uses largest connected component
# - manual nearest-node via haversine (avoids scikit-learn issues)
# - if start/end snap to same node picks next nearest
# - exports GeoJSON and PNG with start/end markers
import os
import time
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# import benchmark loader
from qc_load_benchmarks import load_benchmark_points

# ---------- CONFIG ----------
graph_path = r"D:\Quezon_City\data\processed\qc_roads_major.graphml"
output_folder = r"D:\Quezon_City\data\processed"
os.makedirs(output_folder, exist_ok=True)

# Load benchmark points
benchmarks = load_benchmark_points()

# 👇 Choose start & end by name from benchmark_points.json
start_name = "SM Fairview"
end_name   = "SM North EDSA"

start_latlng = (benchmarks[start_name]["lat"], benchmarks[start_name]["lon"])
end_latlng   = (benchmarks[end_name]["lat"], benchmarks[end_name]["lon"])

print(f"🚏 Start: {start_name} -> {start_latlng}")
print(f"🏁 End:   {end_name} -> {end_latlng}")
# ----------------------------

def haversine_array(lat1, lon1, lat2_arr, lon2_arr):
    # returns meters
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


t0 = time.time()
print("⏳ Loading graph...")
G = ox.load_graphml(graph_path)
print(f"Loaded graph in {time.time()-t0:.2f} s")

t1 = time.time()
# keep largest connected component (undirected)
if not nx.is_connected(G.to_undirected()):
    print("⚠️ Graph not fully connected, extracting largest component...")
    largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
    G = G.subgraph(largest_cc).copy()
print(f"Largest component extracted in {time.time() - t1:.2f} s")

# Build nodes GeoDataFrame (unprojected; columns 'x' (lon) and 'y' (lat))
t2 = time.time()
nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
# nodes_gdf index = node ids; has 'x' and 'y' columns (lon, lat)
print(f"Prepared nodes GeoDataFrame in {time.time()-t2:.2f} s (n_nodes={len(nodes_gdf)})")


# compute nearest nodes by haversine to avoid scikit-learn / projection mismatch
t3 = time.time()
lat_arr = nodes_gdf['y'].to_numpy()
lon_arr = nodes_gdf['x'].to_numpy()

dist_start = haversine_array(start_latlng[0], start_latlng[1], lat_arr, lon_arr)
dist_end   = haversine_array(end_latlng[0], end_latlng[1],   lat_arr, lon_arr)

# get sorted node indices (node ids) by distance
sorted_start_idx = np.argsort(dist_start)
sorted_end_idx = np.argsort(dist_end)

orig_node = nodes_gdf.index[sorted_start_idx[0]]
dest_node = nodes_gdf.index[sorted_end_idx[0]]

# if they collapse to the same node, pick next nearest for destination
if orig_node == dest_node:
    print("⚠️ Start and end snapped to the same node — selecting next nearest for destination.")
    # find next dest that is different
    for idx in sorted_end_idx[1:]:
        cand = nodes_gdf.index[idx]
        if cand != orig_node:
            dest_node = cand
            break

print(f"Nearest nodes chosen in {time.time()-t3:.2f} s")
print(f"orig_node = {orig_node}, coord = ({nodes_gdf.loc[orig_node,'y']:.6f}, {nodes_gdf.loc[orig_node,'x']:.6f})")
print(f"dest_node = {dest_node}, coord = ({nodes_gdf.loc[dest_node,'y']:.6f}, {nodes_gdf.loc[dest_node,'x']:.6f})")

# --------------- compute shortest path ----------------
t4 = time.time()
try:
    route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    length_m = nx.shortest_path_length(G, orig_node, dest_node, weight="length")
    print(f"Shortest path computed in {time.time()-t4:.2f} s (length = {length_m:.2f} m, steps = {len(route)})")
except nx.NetworkXNoPath:
    print("❌ No path found between the chosen nodes (even in largest component). Exiting.")
    raise

# --------------- export route geometry ----------------
t5 = time.time()
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

# collect matching edge indices for the route
route_edges_idx = []
for u, v in zip(route[:-1], route[1:]):
    # try (u,v,0) first (common)
    if (u, v, 0) in edges_gdf.index:
        route_edges_idx.append((u, v, 0))
    else:
        # find any edge index matching (u,v) (or reversed)
        matches = [idx for idx in edges_gdf.index if idx[0] == u and idx[1] == v]
        if not matches:
            matches = [idx for idx in edges_gdf.index if idx[0] == v and idx[1] == u]  # try reversed
        if matches:
            route_edges_idx.append(matches[0])
        # else: skip (we'll fallback below)

if route_edges_idx:
    route_gdf = edges_gdf.loc[route_edges_idx]
else:
    # fallback: construct linestring from node coords (lon,lat)
    coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in route]  # x=lon, y=lat
    line = LineString(coords)
    route_gdf = gpd.GeoDataFrame({'length':[length_m]}, geometry=[line], crs="EPSG:4326")

route_file = os.path.join(output_folder, "qc_shortest_path.geojson")
route_gdf.to_file(route_file, driver="GeoJSON")
print(f"GeoJSON exported in {time.time()-t5:.2f} s -> {route_file}")

# --------------- plotting (projected for nicer layout) ----------------
t6 = time.time()
G_proj = ox.project_graph(G)  # projection just for plot
fig, ax = ox.plot_graph_route(
    G_proj, route, route_linewidth=3, node_size=0, bgcolor="white", show=False, close=False
)

# plot start / end markers on projected graph coords
x_start, y_start = G_proj.nodes[orig_node]['x'], G_proj.nodes[orig_node]['y']
x_end,   y_end   = G_proj.nodes[dest_node]['x'], G_proj.nodes[dest_node]['y']

ax.scatter(x_start, y_start, marker='o', s=100, facecolors='none', edgecolors='green', linewidths=2, label='Start')
ax.scatter(x_end,   y_end,   marker='X', s=100, color='red', label='End')
ax.legend(loc='best')

output_img = os.path.join(output_folder, "qc_shortest_path.png")
fig.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"PNG plotted and saved in {time.time()-t6:.2f} s -> {output_img}")

# --------------- summary ----------------
print("\n=== SUMMARY ===")
print(f"Total runtime: {time.time() - t0:.2f} s")
print(f"Route nodes count: {len(route)}")
print(f"Route length (meters): {length_m:.2f}")
print(f"GeoJSON saved to: {route_file}")
print(f"Image saved to: {output_img}")


