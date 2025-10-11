import osmnx as ox
import numpy as np

graph_path = "data/processed/qc_roads_major.graphml"
start_coords = (121.0596, 14.7324)   # lon, lat
goal_coords = (121.080857, 14.59297)

print("🔍 Loading graph...")
G = ox.load_graphml(graph_path)
print(f"Loaded with {len(G.nodes)} nodes")

# Extract node data
nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
lon_arr = nodes["x"].to_numpy()
lat_arr = nodes["y"].to_numpy()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def nearest_node(coord):
    lat, lon = coord[1], coord[0]
    dists = haversine(lat, lon, lat_arr, lon_arr)
    idx = np.argmin(dists)
    return nodes.index[idx], nodes.iloc[idx]["y"], nodes.iloc[idx]["x"], dists[idx]

start_node, s_lat, s_lon, s_dist = nearest_node(start_coords)
goal_node, g_lat, g_lon, g_dist = nearest_node(goal_coords)

print(f"🎯 Start snapped to node {start_node} at ({s_lat:.6f}, {s_lon:.6f}) — {s_dist:.2f} m away")
print(f"🏁 Goal  snapped to node {goal_node} at ({g_lat:.6f}, {g_lon:.6f}) — {g_dist:.2f} m away")

if start_node == goal_node:
    print("⚠️ Start and goal snapped to the same node!")
