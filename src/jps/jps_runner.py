# src/jps/jps_runner.py

import os
import json
import geopandas as gpd
import networkx as nx
import numpy as np
from .jps_core import JPS   # ✅ relative import

# Paths (relative to project root)
ROADS_FILE = "data/processed/qc_roads_clean.graphml"
BENCHMARK_FILE = "data/inputs/benchmark_points.json"
OUTPUT_GEOJSON = "data/outputs/qc_jps_path.geojson"
OUTPUT_LOG = "data/outputs/qc_jps_log.txt"

def load_graph():
    """Load cleaned road network from GraphML."""
    if not os.path.exists(ROADS_FILE):
        raise FileNotFoundError(f"❌ Missing {ROADS_FILE}")
    return nx.read_graphml(ROADS_FILE)

def load_benchmarks():
    """Load benchmark start/end points from JSON."""
    if not os.path.exists(BENCHMARK_FILE):
        raise FileNotFoundError(f"❌ Missing {BENCHMARK_FILE}")
    with open(BENCHMARK_FILE, "r") as f:
        return json.load(f)

def graph_to_grid(G, resolution=200):
    """
    Convert road graph to a grid (binary numpy array).
    0 = free, 1 = obstacle.
    """
    nodes = np.array([[float(d["y"]), float(d["x"])] for n, d in G.nodes(data=True)])
    min_y, min_x = nodes.min(axis=0)
    max_y, max_x = nodes.max(axis=0)

    width = int((max_x - min_x) * resolution)
    height = int((max_y - min_y) * resolution)

    # Initialize grid as all obstacles
    grid = np.ones((height, width), dtype=int)

    for u, v in G.edges():
        y1, x1 = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
        y2, x2 = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])

        r1, c1 = int((y1 - min_y) * resolution), int((x1 - min_x) * resolution)
        r2, c2 = int((y2 - min_y) * resolution), int((x2 - min_x) * resolution)

        # Draw straight line edges as free cells
        num = max(abs(r2 - r1), abs(c2 - c1)) + 1
        rr = np.linspace(r1, r2, num=num).astype(int)
        cc = np.linspace(c1, c2, num=num).astype(int)

        # 🔹 Clip indices to stay inside grid bounds
        rr = np.clip(rr, 0, grid.shape[0] - 1)
        cc = np.clip(cc, 0, grid.shape[1] - 1)

        grid[rr, cc] = 0

    return grid, (min_y, min_x), resolution


def latlon_to_grid(lat, lon, bbox, resolution):
    """Convert (lat, lon) to grid coordinates."""
    min_y, min_x = bbox
    return int((lat - min_y) * resolution), int((lon - min_x) * resolution)

def grid_to_latlon(r, c, bbox, resolution):
    """Convert grid coordinates back to (lat, lon)."""
    min_y, min_x = bbox
    return (r / resolution + min_y, c / resolution + min_x)

def run_jps():
    print("⏳ Loading road network...")
    G = load_graph()
    print(f"✅ Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")

    print("⏳ Loading benchmark points...")
    pois = load_benchmarks()
    print(f"✅ Loaded {len(pois)} POIs")

    print("⏳ Converting graph to grid...")
    grid, bbox, resolution = graph_to_grid(G, resolution=2000)  # ✅ higher resolution
    print(f"✅ Grid ready with shape {grid.shape}")

    # Define benchmark routes manually (pairs of POIs)
    benchmark_pairs = [
        ("Quezon Memorial Circle", "SM Fairview"),
        ("Vertis North", "Tandang Sora MRT Station"),
        ("Quezon Memorial Circle", "SM North EDSA")
    ]

    for key, (src, dst) in enumerate(benchmark_pairs, start=1):
        print(f"\n🚀 Running JPS for {src} → {dst}...")

        # ✅ Snap POIs to nearest road node
        start_lat, start_lon = snap_to_nearest_node(G, pois[src]["lat"], pois[src]["lon"])
        goal_lat, goal_lon = snap_to_nearest_node(G, pois[dst]["lat"], pois[dst]["lon"])

        start = latlon_to_grid(start_lat, start_lon, bbox, resolution)
        goal = latlon_to_grid(goal_lat, goal_lon, bbox, resolution)

        # ✅ Force start and goal cells to be walkable
        grid[start] = 0
        grid[goal] = 0

        jps = JPS(grid, start, goal, debug=True)
        path = jps.search()


        if path is None:
            print(f"❌ No path found for {src} → {dst}")
            continue

        # Convert back to lat/lon
        coords = [grid_to_latlon(r, c, bbox, resolution) for r, c in path]

        # Save as LineString GeoJSON
        out_file = f"data/outputs/qc_jps_path_{key}.geojson"
        gdf = gpd.GeoDataFrame(
            {"id": [0]},
            geometry=[
                gpd.points_from_xy(
                    [lon for lat, lon in coords],
                    [lat for lat, lon in coords]
                ).union_all()
            ],
            crs="EPSG:4326"  # ✅ ensure proper projection
        )
        gdf.to_file(out_file, driver="GeoJSON")

        # Save log (UTF-8 safe)
        log_file = f"data/outputs/qc_jps_log_{key}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Benchmark {key}: {src} → {dst}\n")
            f.write(f"Start: {start}\n")
            f.write(f"Goal: {goal}\n")
            f.write(f"Path length: {len(path)}\n")
            f.write(f"Grid steps: {path}\n")

        print(f"✅ Saved GeoJSON → {out_file}")
        print(f"📄 Log written → {log_file}")

def snap_to_nearest_node(G, lat, lon):
    """
    Snap (lat, lon) to the nearest graph node.
    Ensures POIs align with the road network.
    """
    nearest = min(
        G.nodes,
        key=lambda n: (
            (float(G.nodes[n]["y"]) - lat) ** 2
            + (float(G.nodes[n]["x"]) - lon) ** 2
        )
    )
    return float(G.nodes[nearest]["y"]), float(G.nodes[nearest]["x"])

if __name__ == "__main__":
    run_jps()

print("⏳ Loading road network...")
G = load_graph()
print(f"✅ Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")

print("⏳ Loading benchmark points...")
benchmarks = load_benchmarks()
print(f"✅ Loaded {len(benchmarks)} benchmarks")

print("⏳ Converting graph to grid...")
grid, bbox, resolution = graph_to_grid(G)
print(f"✅ Grid ready with shape {grid.shape}")

print("🚀 Running JPS...")
