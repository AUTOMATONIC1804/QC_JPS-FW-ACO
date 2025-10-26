"""
src/algorithms/dijkstra_runner.py
Clean and modular Dijkstra pathfinding — now with curved OSM edge geometries.
"""

import os
import json
import time
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, mapping
from shapely.ops import linemerge

from src.algorithms.dijkstra.dijkstra_main import prepare_graph, snap_nodes


def build_curved_route(G, path_nodes):
    """
    Merge true OSM edge geometries along the path to form a smooth route.
    Falls back to straight lines if no geometry field exists.
    """
    edge_lines = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = G.get_edge_data(u, v, 0)
        geom = edge_data.get("geometry", None)
        if geom is None:
            geom = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"])
            ])
        edge_lines.append(geom)
    try:
        merged = linemerge(edge_lines)
    except Exception:
        # fallback if invalid merge
        coords = []
        for g in edge_lines:
            coords.extend(list(g.coords))
        merged = LineString(coords)
    return merged


def run_dijkstra_benchmark(
    graph_path="data/processed/qc_roads_major.graphml",
    start_coords=(121.0469586, 14.6500329), 
    goal_coords=(121.0018562, 14.617906), 
    output_dir="data/outputs"
):
    print(f"🚆 Running Dijkstra on {graph_path}")

    # --- Load and preprocess graph ---
    G = prepare_graph(graph_path)

    # --- Snap coordinates to nearest nodes ---
    start_node, goal_node, dist_start, dist_goal = snap_nodes(G, start_coords, goal_coords)
    print(f"🎯 Start node: {start_node} ({dist_start:.2f} m away)")
    print(f"🏁 Goal  node: {goal_node} ({dist_goal:.2f} m away)")

    if not nx.has_path(G, start_node, goal_node):
        print("❌ No path found between nodes.")
        return {"algorithm": "Dijkstra", "runtime_ms": None, "path_length_m": None, "steps": None}

    # --- Run Dijkstra ---
    t0 = time.time()
    path = nx.shortest_path(G, source=start_node, target=goal_node, weight="length")
    length_m = nx.shortest_path_length(G, source=start_node, target=goal_node, weight="length")
    runtime_ms = (time.time() - t0) * 1000
    adjusted_length = max(0, length_m - (dist_start + dist_goal))
    print(f"[OK] Dijkstra completed — Runtime: {runtime_ms:.2f} ms, Path length: {adjusted_length:.2f} m")

    # --- Build curved route from OSM edges ---
    print("[OK] Building curved geometry path...")
    curved_line = build_curved_route(G, path)

    # --- Visualization ---
    print("[OK] Rendering clean black map...")
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ox.plot_graph(
        G, ax=ax, node_size=0, edge_color="white",
        edge_linewidth=0.6, bgcolor="black", show=False, close=False
    )

    # Plot Dijkstra path (blue)
    ax.plot(*curved_line.xy, color="#2196F3", linewidth=2.8, label="Dijkstra Path", zorder=3)

    # Start / Goal markers
    x_start, y_start = G.nodes[start_node]["x"], G.nodes[start_node]["y"]
    x_goal, y_goal = G.nodes[goal_node]["x"], G.nodes[goal_node]["y"]

    ax.scatter(x_start, y_start, s=120, facecolor="#00FF00", edgecolors="black", linewidth=1.2, zorder=5, label="Start")
    ax.scatter(x_goal, y_goal, s=140, color="red", marker="X", zorder=5, label="Goal")

    # Legend styling (top-right, white text)
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

    # --- GeoJSON Export (both straight and curved) ---
    straight_coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]

    geojson = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "geometry": mapping(curved_line),
         "properties": {"algorithm": "Dijkstra", "geometry": "curved"}},
        {"type": "Feature", "geometry": mapping(Point(x_start, y_start)),
         "properties": {"role": "start"}},
        {"type": "Feature", "geometry": mapping(Point(x_goal, y_goal)),
         "properties": {"role": "goal"}},
        ],
    }


    out_geojson = os.path.join(output_dir, "dijkstra_path.geojson")
    with open(out_geojson, "w") as f:
        json.dump(geojson, f)
    print(f"[OK] Saved GeoJSON → {out_geojson}")

    # --- Return metrics ---
    return {
        "algorithm": "Dijkstra",
        "runtime_ms": float(runtime_ms),
        "path_length_m": float(adjusted_length),
        "steps": len(path),
    }
