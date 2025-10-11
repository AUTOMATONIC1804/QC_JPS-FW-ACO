"""
src/algorithms/dijkstra_runner.py
Run Dijkstra on the QC road graph (.graphml) using lon/lat input.
Exports PNG + GeoJSON with path, start, and goal.
"""

import os
import time
import json
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def run_dijkstra_benchmark(
    graph_path="data/processed/qc_roads_major.graphml",
    start_coords=(121.0596, 14.7324),
    goal_coords=(121.080857, 14.59297),
    output_dir="data/outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print(f"🚆 Running Dijkstra on {graph_path}")

    G = ox.load_graphml(graph_path)
    print(f"[OK] Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")

    # ✅ Extract largest connected component
    if not nx.is_connected(G.to_undirected()):
        largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"[OK] Using largest connected component: {len(G.nodes)} nodes")

    # --- Find nearest nodes via haversine ---
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    lat_arr, lon_arr = nodes_gdf["y"].to_numpy(), nodes_gdf["x"].to_numpy()

    sx, sy = start_coords
    gx, gy = goal_coords
    dist_start = haversine(sy, sx, lat_arr, lon_arr)
    dist_goal = haversine(gy, gx, lat_arr, lon_arr)

    start_node = nodes_gdf.index[np.argmin(dist_start)]
    goal_node = nodes_gdf.index[np.argmin(dist_goal)]

    if start_node == goal_node:
        print("⚠️ Start and goal snapped to same node, picking next nearest goal.")
        sorted_idx = np.argsort(dist_goal)
        for idx in sorted_idx[1:]:
            if nodes_gdf.index[idx] != start_node:
                goal_node = nodes_gdf.index[idx]
                break

    print(f"🎯 Start node: {start_node}, Goal node: {goal_node}")

    # ✅ Ensure both nodes are still in the graph
    if start_node not in G.nodes or goal_node not in G.nodes:
        print("⚠️ One or both nodes not in largest component, re-snapping...")
        # Recompute using the remaining G nodes
        nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
        lat_arr, lon_arr = nodes_gdf["y"].to_numpy(), nodes_gdf["x"].to_numpy()
        dist_start = haversine(sy, sx, lat_arr, lon_arr)
        dist_goal = haversine(gy, gx, lat_arr, lon_arr)
        start_node = nodes_gdf.index[np.argmin(dist_start)]
        goal_node = nodes_gdf.index[np.argmin(dist_goal)]
        print(f"✅ Resnapped to connected nodes: {start_node}, {goal_node}")

    # --- Run Dijkstra ---
    try:
        t1 = time.time()
        path = nx.shortest_path(G, source=start_node, target=goal_node, weight="length")
        path_length_m = nx.shortest_path_length(G, source=start_node, target=goal_node, weight="length")
        runtime_ms = (time.time() - t1) * 1000
        print(f"[OK] Dijkstra completed — Runtime: {runtime_ms:.2f} ms, Path length: {path_length_m:.2f} m")
    except nx.NetworkXNoPath:
        print("❌ Still no path found even in largest component.")
        return {
            "algorithm": "Dijkstra",
            "runtime_ms": None,
            "path_length_m": None,
            "steps": None,
        }

    # --- Visualization ---
    G_proj = ox.project_graph(G)
    fig, ax = ox.plot_graph_route(G_proj, path, route_linewidth=2, node_size=0, bgcolor="white", show=False, close=False)
    ax.scatter(G_proj.nodes[start_node]["x"], G_proj.nodes[start_node]["y"], color="green", s=60, label="Start")
    ax.scatter(G_proj.nodes[goal_node]["x"], G_proj.nodes[goal_node]["y"], color="red", s=60, marker="x", label="Goal")
    ax.legend()
    plt.savefig(f"{output_dir}/dijkstra_path.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Dijkstra visualization → {output_dir}/dijkstra_path.png")

    # --- GeoJSON Export ---
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
    route_gdf = gpd.GeoDataFrame(geometry=[LineString(coords)], crs="EPSG:4326")
    route_gdf.to_file(f"{output_dir}/dijkstra_path.geojson", driver="GeoJSON")
    print(f"[OK] Saved GeoJSON → {output_dir}/dijkstra_path.geojson")

    return {
        "algorithm": "Dijkstra",
        "runtime_ms": float(runtime_ms),
        "path_length_m": float(path_length_m),
        "steps": len(path)
    }



