"""
src/algorithms/dijkstra/dijkstra_main.py
Handles graph preparation and Dijkstra pathfinding logic.
"""

import networkx as nx
import osmnx as ox
from src.algorithms.dijkstra.dijkstra_utils import haversine


def prepare_graph(graph_path):
    """Load and preprocess OSMnx graph (convert to undirected, clean, etc.)"""
    G = ox.load_graphml(graph_path)
    print(f"[OK] Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Convert to undirected
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected(reciprocal=False)
        print("[OK] Graph converted to undirected")

    # Ensure edge lengths are floats
    for _, _, data in G.edges(data=True):
        if "length" in data and isinstance(data["length"], str):
            try:
                data["length"] = float(data["length"])
            except ValueError:
                data["length"] = 0.0

    # Keep largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"[OK] Kept largest component ({len(G.nodes)} nodes)")

    return G


def snap_nodes(G, start_coords, goal_coords):
    """Snap lon/lat coordinates to nearest nodes in the graph."""
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    lat_arr, lon_arr = nodes_gdf["y"].to_numpy(), nodes_gdf["x"].to_numpy()

    def nearest_node(lon, lat):
        dists = ((lat_arr - lat)**2 + (lon_arr - lon)**2)
        idx = dists.argmin()
        node_id = nodes_gdf.index[idx]
        dist_m = haversine(lat, lon, lat_arr[idx], lon_arr[idx])
        return node_id, dist_m

    start_node, dist_start = nearest_node(*start_coords)
    goal_node, dist_goal = nearest_node(*goal_coords)

    return start_node, goal_node, dist_start, dist_goal
