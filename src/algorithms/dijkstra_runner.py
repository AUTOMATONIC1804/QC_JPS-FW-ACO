"""
src/algorithms/dijkstra_runner.py
Clean and modular Dijkstra pathfinding — now with curved OSM edge geometries.
Accepts (lat, lon) coordinate input for convenience when copying from QGIS.
Includes total runtime measurement for uniform benchmarking.
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
        coords = []
        for g in edge_lines:
            coords.extend(list(g.coords))
        merged = LineString(coords)
    return merged


def run_dijkstra_benchmark(
    graph_path="data/processed/qc_roads_major.graphml",
    start_coords=(14.7064939,121.0680891),  
    goal_coords=(14.6550249,121.0549384),    
    output_dir="data/outputs"
):
    """
    Run Dijkstra pathfinding using (lat, lon) input.
    Automatically swaps to (lon, lat) internally for processing.
    Includes total runtime measurement for consistency with JPS and A*.
    """
    print(f"\n=== 🚆 Running Dijkstra Benchmark ===")
    total_start = time.time()  # ⏱️ start total runtime

    try:
        # --- Load and preprocess graph ---
        print("[1] Loading and preparing OSMnx graph...")
        G = prepare_graph(graph_path)

        # --- Swap (lat, lon) → (lon, lat) for internal OSMnx compatibility ---
        start_coords_swapped = (start_coords[1], start_coords[0])
        goal_coords_swapped = (goal_coords[1], goal_coords[0])

        # --- Snap coordinates to nearest nodes ---
        print("[2] Snapping start and goal to nearest nodes...")
        start_node, goal_node, dist_start, dist_goal = snap_nodes(
            G, start_coords_swapped, goal_coords_swapped
        )
        print(f"   🎯 Start node: {start_node} ({dist_start:.2f} m away)")
        print(f"   🏁 Goal  node: {goal_node} ({dist_goal:.2f} m away)")

        if not nx.has_path(G, start_node, goal_node):
            print("❌ No path found between nodes.")
            total_runtime_ms = (time.time() - total_start) * 1000
            return {
                "algorithm": "Dijkstra",
                "runtime_ms": None,
                "total_runtime_ms": float(total_runtime_ms),
                "path_length_m": None,
                "steps": None,
            }

        # --- Run Dijkstra ---
        print("[3] Running Dijkstra shortest path search...")
        t0 = time.time()
        path = nx.shortest_path(G, source=start_node, target=goal_node, weight="length")
        length_m = nx.shortest_path_length(G, source=start_node, target=goal_node, weight="length")
        runtime_ms = (time.time() - t0) * 1000
        adjusted_length = max(0, length_m - (dist_start + dist_goal))
        print(f"[OK] Path found — Runtime: {runtime_ms:.2f} ms, Length: {adjusted_length:.2f} m, Steps: {len(path)}")

        # --- Build curved route from OSM edges ---
        print("[4] Building curved geometry path...")
        curved_line = build_curved_route(G, path)

        # --- Visualization ---
        print("[5] Rendering Dijkstra visualization...")
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

        ax.scatter(x_start, y_start, s=120, facecolor="#00FF00", edgecolors="black",
                   linewidth=1.2, zorder=5, label="Start")
        ax.scatter(x_goal, y_goal, s=140, color="red", marker="X",
                   zorder=5, label="Goal")

        # Legend styling
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
        print(f"✅ Saved visualization → {out_png}")

        # --- GeoJSON Export ---
        print("[6] Exporting GeoJSON route...")
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
        print(f"✅ Saved GeoJSON → {out_geojson}")

        # --- Final total runtime ---
        total_runtime_ms = (time.time() - total_start) * 1000
        print(f"[OK] Total runtime (load → export): {total_runtime_ms:.2f} ms")

        # --- Return metrics ---
        return {
            "algorithm": "Dijkstra",
            "runtime_ms": float(runtime_ms),
            "total_runtime_ms": float(total_runtime_ms),
            "path_length_m": float(adjusted_length),
            "steps": len(path),
        }

    except Exception as e:
        print("\n❌ Dijkstra failed with error:", e)
        total_runtime_ms = (time.time() - total_start) * 1000
        return {
            "algorithm": "Dijkstra",
            "runtime_ms": None,
            "total_runtime_ms": float(total_runtime_ms),
            "path_length_m": None,
            "steps": None,
        }
