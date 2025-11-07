"""
fw_dijkstra_runner.py (v8.1)
--------------------------------------------------------
Floyd–Warshall (FW) for Dijkstra corridor.
Now rebuilt to compute true Dijkstra-based distances
directly from the final exported clipped roads + nodes.

✅ Uses only real, buffer-clipped GeoJSON roads & nodes
✅ Distances: computed via Dijkstra over rebuilt local graph
✅ CRS: EPSG:3857 (metric)
✅ Compatible with OSMnx 1.4–2.3+ & NetworkX 3.x
"""

import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from src.algorithms.dijkstra.dijkstra_main import prepare_graph

# Coordinate systems
WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


# ---------------------------------------------------
# Floyd–Warshall core
# ---------------------------------------------------
def floyd_warshall_numpy(D):
    dist = D.copy()
    n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


# ---------------------------------------------------
# Route + buffer utilities
# ---------------------------------------------------
def load_route_line(route_geojson: str) -> LineString:
    gdf = gpd.read_file(route_geojson)
    if gdf.crs is None:
        warnings.warn("Route GeoJSON has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs(WGS84)
    gdf = gdf.to_crs(WGS84)
    geom = unary_union(gdf.geometry.values)
    if isinstance(geom, MultiLineString):
        coords = []
        for ls in geom.geoms:
            coords.extend(ls.coords)
        geom = LineString(coords)
    print(f"✅ Loaded route (length ≈ {LineString(geom).length:.0f})")
    return geom


def buffer_route(line_any: LineString, buffer_m: float) -> gpd.GeoDataFrame:
    """Create metric buffer polygon (CRS-agnostic safe)."""
    gseries = gpd.GeoSeries([line_any])
    if gseries.crs is None:
        gseries = gseries.set_crs(WGS84)
    line_m = gseries.to_crs(METRIC)
    buf = line_m.buffer(buffer_m)
    buf_gdf = gpd.GeoDataFrame(geometry=buf, crs=METRIC)
    print(f"🧱 Created buffer ±{buffer_m} m (metric)")
    return buf_gdf


# ---------------------------------------------------
# Node spacing filter
# ---------------------------------------------------
def filter_nodes_by_spacing(nodes_gdf, spacing_m):
    if nodes_gdf.empty:
        return nodes_gdf
    nodes_m = nodes_gdf.to_crs(METRIC).copy()
    nodes_m["x"] = nodes_m.geometry.x
    nodes_m["y"] = nodes_m.geometry.y
    nodes_m = nodes_m.sort_values(by=["x", "y"]).reset_index(drop=True)
    nodes_m = nodes_m.drop(columns=["x", "y"])
    kept_idx, kept_pts = [], []
    for i, row in nodes_m.iterrows():
        geom = row.geometry
        if all(geom.distance(p) >= spacing_m for p in kept_pts):
            kept_idx.append(i)
            kept_pts.append(geom)
    thinned = nodes_m.loc[kept_idx].copy()
    print(f"🧩 Filtered {len(nodes_gdf)} → {len(thinned)} nodes (~{spacing_m} m apart)")
    return thinned.to_crs(WGS84)


# ---------------------------------------------------
# Local Dijkstra from exported roads/nodes
# ---------------------------------------------------
def compute_local_dijkstra_matrix(roads_path, nodes_path):
    """Build graph from exported clipped roads and compute pairwise Dijkstra distances."""
    print("🔁 Rebuilding local graph from exported roads + nodes...")
    roads_gdf = gpd.read_file(roads_path).to_crs(METRIC)
    nodes_gdf = gpd.read_file(nodes_path).to_crs(METRIC)

    G_local = nx.Graph()
    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = [pt for line in geom.geoms for pt in line.coords]
        else:
            continue
        for a, b in zip(coords[:-1], coords[1:]):
            dist = Point(a).distance(Point(b))
            G_local.add_edge(a, b, weight=dist)

    print(f"🧱 Built local Dijkstra graph: {len(G_local.nodes)} nodes, {len(G_local.edges)} edges")

    # Snap nodes to nearest graph coordinate
    graph_coords = np.array(list(G_local.nodes))
    snap_nodes = []
    for p in nodes_gdf.geometry:
        arr = np.array([p.x, p.y])
        dists = np.linalg.norm(graph_coords - arr, axis=1)
        nearest = tuple(graph_coords[np.argmin(dists)])
        snap_nodes.append(nearest)

    print(f"🔗 Snapped {len(snap_nodes)} nodes to local graph vertices.")

    # Compute pairwise Dijkstra distances
    n = len(snap_nodes)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    print("🚦 Computing pairwise Dijkstra distances on local graph...")
    for i, src in enumerate(snap_nodes):
        lengths = nx.single_source_dijkstra_path_length(G_local, src, weight="weight")
        for j, dst in enumerate(snap_nodes):
            if dst in lengths:
                D[i, j] = lengths[dst]
        if i % 5 == 0 or i == n - 1:
            print(f"   ↳ {i+1}/{n} done")

    finite = np.isfinite(D).sum()
    print(f"🔍 Finite distances found: {finite}/{D.size}")
    return np.minimum(D, D.T)


# ---------------------------------------------------
# Runner
# ---------------------------------------------------
def run_fw_dijkstra(
    route_geojson="data/outputs/dijkstra_path.geojson",
    graphml_path="data/processed/qc_roads_major.graphml",
    buffer_m=2000,
    spacing_m=500,
    output_dir="data/outputs/floyd_warshall"
):
    print("🚆 FW (Dijkstra + Real Road Network v8.1)")
    print(f"Route: {route_geojson}")
    print(f"Graph: {graphml_path}")
    print(f"Buffer: ±{buffer_m} m | Spacing: {spacing_m} m")

    t0 = time.perf_counter()

    # 1️⃣ Load route + buffer
    route_line = load_route_line(route_geojson)
    buffer_gdf = buffer_route(route_line, buffer_m)

    # 2️⃣ Prepare graph (no clipping yet)
    print("📦 Preparing OSMnx graph...")
    G_full = prepare_graph(graphml_path)
    G_full = ox.project_graph(G_full, to_crs=METRIC)
    for u, v, d in G_full.edges(data=True):
        if "length" not in d:
            d["length"] = LineString([
                (G_full.nodes[u]["x"], G_full.nodes[u]["y"]),
                (G_full.nodes[v]["x"], G_full.nodes[v]["y"]),
            ]).length

    # 3️⃣ Try clipping to buffer polygon (best effort)
    poly = buffer_gdf.to_crs(WGS84).geometry.iloc[0]
    try:
        try:
            from osmnx.graph_utils import truncate_graph_polygon
        except ImportError:
            try:
                from osmnx.graph import truncate_graph_polygon
            except ImportError:
                try:
                    from osmnx.truncate import truncate_graph_polygon
                except ImportError:
                    from osmnx.utils_graph import truncate_graph_polygon

        try:
            G_clip = truncate_graph_polygon(G_full, poly, retain_all=True)
        except TypeError:
            G_clip = truncate_graph_polygon(G_full, poly)
        print(f"✅ Clipped graph: {len(G_clip.nodes)} nodes, {len(G_clip.edges)} edges")
    except Exception as e:
        print("⚠️ Clip failed; using full graph. Error:", e)
        G_clip = G_full

    # 4️⃣ Convert to GeoDataFrames
    try:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_clip)
    except Exception:
        print("⚙️ Using manual graph_to_gdfs fallback (NetworkX 3.x fix)")
        node_rows = [{"osmid": nid, "geometry": Point((d["x"], d["y"]))} for nid, d in G_clip.nodes(data=True)]
        nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=WGS84)
        edge_rows = []
        for u, v, data in G_clip.edges(data=True):
            geom = data.get("geometry", LineString([
                (G_clip.nodes[u]["x"], G_clip.nodes[u]["y"]),
                (G_clip.nodes[v]["x"], G_clip.nodes[v]["y"]),
            ]))
            edge_rows.append({"u": u, "v": v, "geometry": geom})
        edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=WGS84)

    # 5️⃣ Filter nodes by spacing and clip to buffer
    nodes_filtered = filter_nodes_by_spacing(nodes_gdf, spacing_m)
    poly_m = buffer_gdf.to_crs(METRIC).geometry.iloc[0]
    nodes_filtered = nodes_filtered.to_crs(METRIC)
    nodes_filtered = nodes_filtered[nodes_filtered.geometry.within(poly_m)]
    nodes_filtered = nodes_filtered.to_crs(WGS84)
    print(f"📏 After buffer clip: {len(nodes_filtered)} nodes")

    edges_clip = gpd.overlay(edges_gdf.to_crs(METRIC), buffer_gdf, how="intersection").to_crs(WGS84)
    print(f"🛣️ Clipped roads: {len(edges_clip)}")

    # 6️⃣ Export preliminary outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_crs(WGS84).to_file(out / "fw_dijkstra_buffer.geojson", driver="GeoJSON")
    edges_clip.to_file(out / "fw_dijkstra_roads.geojson", driver="GeoJSON")
    nodes_filtered.to_file(out / "fw_dijkstra_nodes.geojson", driver="GeoJSON")

    # 7️⃣ Compute Dijkstra + FW from exported roads/nodes
    roads_path = out / "fw_dijkstra_roads.geojson"
    nodes_path = out / "fw_dijkstra_nodes.geojson"
    D = compute_local_dijkstra_matrix(roads_path, nodes_path)
    FW = floyd_warshall_numpy(D)

    # 8️⃣ Save matrices
    np.save(out / "fw_dijkstra_D.npy", D)
    np.save(out / "fw_dijkstra_FW.npy", FW)

    # 9️⃣ Summary
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print("\n✅ Summary (Local Dijkstra + FW Integration)")
    print(f"🧩 Nodes: {len(nodes_filtered)} | 🛣️ Edges: {len(edges_clip)}")
    print(f"📐 Matrix: {D.shape} | ⏱ {elapsed_ms:.2f} ms | 📂 {out}")
    print(f"♾️ Non-finite: {np.count_nonzero(~np.isfinite(D))} / {D.size}")
    print(f"⚖️ Symmetric? {np.allclose(D, D.T, equal_nan=True)}")
    print("===================================")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_geojson", default="data/outputs/dijkstra_path.geojson")
    ap.add_argument("--graphml_path", default="data/processed/qc_roads_major.graphml")
    ap.add_argument("--buffer_m", type=float, default=2000)
    ap.add_argument("--spacing_m", type=float, default=500)
    ap.add_argument("--output_dir", default="data/outputs/floyd_warshall")
    args = ap.parse_args()

    run_fw_dijkstra(**vars(args))
