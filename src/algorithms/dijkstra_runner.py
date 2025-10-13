"""
src/algorithms/dijkstra_runner.py
Clean, consistent visualization (black background, white roads, top-right legend).
"""

import os
import json
import time
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, mapping
from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (in meters)."""
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def run_dijkstra_benchmark(
    graph_path="data/processed/qc_roads_major.graphml",
    start_coords=(121.0596, 14.7324),
    goal_coords=(121.080857, 14.59297),
    output_dir="data/outputs"
):
    print(f"🚆 Running Dijkstra on {graph_path}")

    G = ox.load_graphml(graph_path)
    print(f"[OK] Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Convert to undirected
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected(reciprocal=False)
        print("[OK] Graph converted to undirected")

    # Ensure 'length' is float
    for _, _, data in G.edges(data=True):
        if "length" in data and isinstance(data["length"], str):
            try:
                data["length"] = float(data["length"])
            except ValueError:
                data["length"] = 0.0

    # Largest component only
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"[OK] Kept largest component ({len(G.nodes)} nodes)")

    # Snap to nearest nodes
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
    print(f"🎯 Start node: {start_node} ({dist_start:.2f} m away)")
    print(f"🏁 Goal  node: {goal_node} ({dist_goal:.2f} m away)")

    if not nx.has_path(G, start_node, goal_node):
        print("❌ No path found between nodes.")
        return {"algorithm": "Dijkstra", "runtime_ms": None, "path_length_m": None, "steps": None}

    # Run Dijkstra
    t0 = time.time()
    path = nx.shortest_path(G, source=start_node, target=goal_node, weight="length")
    length_m = nx.shortest_path_length(G, source=start_node, target=goal_node, weight="length")
    runtime_ms = (time.time() - t0) * 1000
    adjusted_length = max(0, length_m - (dist_start + dist_goal))
    print(f"[OK] Dijkstra completed — Runtime: {runtime_ms:.2f} ms, Path length: {adjusted_length:.2f} m")

    # --- Visualization (black background + white roads, unified style) ---
    print("[OK] Rendering clean black map...")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Plot base map (white roads)
    ox.plot_graph(
        G, ax=ax, node_size=0, edge_color="white",
        edge_linewidth=0.6, bgcolor="black", show=False, close=False
    )

    # Dijkstra path (blue)
    x_coords = [G.nodes[n]["x"] for n in path]
    y_coords = [G.nodes[n]["y"] for n in path]
    ax.plot(x_coords, y_coords, color="#2196F3", linewidth=2.8, label="Dijkstra Path", zorder=3)

    # Start / Goal markers
    x_start, y_start = G.nodes[start_node]["x"], G.nodes[start_node]["y"]
    x_goal, y_goal = G.nodes[goal_node]["x"], G.nodes[goal_node]["y"]

    ax.scatter(x_start, y_start, s=120, facecolor="#00FF00", edgecolors="black", linewidth=1.2, zorder=5, label="Start")
    ax.scatter(x_goal, y_goal, s=140, color="red", marker="X", zorder=5, label="Goal")

    # Legend (top-right, white text, black background)
    leg = ax.legend(
        loc="upper right", frameon=True, facecolor="black",
        edgecolor="white", labelcolor="white", framealpha=0.8
    )
    for text in leg.get_texts():
        text.set_color("white")

    ax.axis("off")
    plt.tight_layout(pad=0)
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "dijkstra_path.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)

    print(f"[OK] Saved Dijkstra visualization → {out_png}")


    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(LineString(coords)), "properties": {"algorithm": "Dijkstra"}},
            {"type": "Feature", "geometry": mapping(Point(x_start, y_start)), "properties": {"role": "start"}},
            {"type": "Feature", "geometry": mapping(Point(x_goal, y_goal)), "properties": {"role": "goal"}},
        ],
    }
    with open(os.path.join(output_dir, "dijkstra_path.geojson"), "w") as f:
        json.dump(geojson, f)

    return {
        "algorithm": "Dijkstra",
        "runtime_ms": float(runtime_ms),
        "path_length_m": float(adjusted_length),
        "steps": len(path),
    }
