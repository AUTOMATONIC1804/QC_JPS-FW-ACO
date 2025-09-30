# Major Roads Extraction for QC
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
import os

# 1. Set place name
place_name = "Quezon City, Philippines"

# 2. Download road network (driveable roads only)
print("⏳ Downloading road network...")
G = ox.graph_from_place(place_name, network_type="drive")

# 3. Filter graph edges by highway type directly
major_road_types = {"primary", "secondary", "trunk"}
edges_to_keep = []
for u, v, k, data in G.edges(keys=True, data=True):
    if "highway" in data:
        hw = data["highway"]
        if isinstance(hw, list):
            if any(h in major_road_types for h in hw):
                edges_to_keep.append((u, v, k))
        else:
            if hw in major_road_types:
                edges_to_keep.append((u, v, k))

G_major = G.edge_subgraph(edges_to_keep).copy()
G_major.graph["crs"] = G.graph.get("crs", "epsg:4326")

# 4. Ensure geometries are present (manually add LineStrings if missing)
print("🛠 Adding edge geometries manually...")
for u, v, k, data in G_major.edges(keys=True, data=True):
    if "geometry" not in data:
        # Build LineString from node coordinates
        x1, y1 = G_major.nodes[u]["x"], G_major.nodes[u]["y"]
        x2, y2 = G_major.nodes[v]["x"], G_major.nodes[v]["y"]
        data["geometry"] = LineString([(x1, y1), (x2, y2)])

# 5. Ensure connectivity (largest connected component)
if not nx.is_connected(G_major.to_undirected()):
    print("⚠️ Graph not fully connected, extracting largest component...")
    largest_cc = max(nx.connected_components(G_major.to_undirected()), key=len)
    G_major = G_major.subgraph(largest_cc).copy()
    print(f"✅ Largest component kept (n_nodes={G_major.number_of_nodes()}, n_edges={G_major.number_of_edges()})")

# 6. Define output folder (your project structure)
output_folder = r"D:\Quezon_City\data\processed"
os.makedirs(output_folder, exist_ok=True)

# 7. Save outputs (with geometry + CRS)
ox.save_graphml(G_major, os.path.join(output_folder, "qc_roads_major.graphml"))
ox.save_graph_geopackage(G_major, os.path.join(output_folder, "qc_roads_major.gpkg"))  # for QGIS

# 8. Print summary
print("✅ Major roads extracted and saved!")
print(f"Final graph: {G_major.number_of_nodes()} nodes, {G_major.number_of_edges()} edges")
print(f"Files saved to: {output_folder}")
