# ==========================
# Major Roads Extraction for QC (Full Cleaned Version + Coordinate Path Removal)
# ==========================
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
import os
import matplotlib.pyplot as plt

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
        x1, y1 = G_major.nodes[u]["x"], G_major.nodes[u]["y"]
        x2, y2 = G_major.nodes[v]["x"], G_major.nodes[v]["y"]
        data["geometry"] = LineString([(x1, y1), (x2, y2)])

# 5. Remove specific roads by name
roads_to_remove = {
    "Regalado Avenue",
    "Republic Avenue",
    "Don Julio Gregorio",
    "Old Sauyo Road",
    "Santo Niño Street",
    "San Simon Street",
    "Sagingan Street",
    "Sampaguita Avenue",
    "Cirillo Street",
    "De Leon Street",
    "Holy Spirit Drive",
    "Gilmore Avenue",
    "Doña Hemady Street",
    "15th Avenue",
    "20th Avenue",
    "Road 1",
    "B. Soliven Street",
    "Chico Street",
    "Xavierville Avenue",
    "Senator Jose O. Vera Street",
    "N. Domingo Street",
    "Mayon Street",
    "Nicanor Reyes Street",
    "Dapitan Street",
    "Blumentritt Street",
    "Maria Clara Street",
    "C. Benitez Street",
}

edges_to_remove = [
    (u, v, k)
    for u, v, k, data in G_major.edges(keys=True, data=True)
    if "name" in data and any(r in str(data["name"]) for r in roads_to_remove)
]
print(f"🧹 Removing {len(edges_to_remove)} edges from {len(roads_to_remove)} specified roads...")
G_major.remove_edges_from(edges_to_remove)

# 5.1 Remove paths between multiple coordinate pairs
paths_to_remove = [
    # (lat_start, lon_start, lat_end, lon_end)
    (14.67266657, 121.06887480, 14.68643427, 121.06987192),
    (14.64045235, 121.01668294, 14.64269194, 121.02663382),
    (14.64276100, 121.02677690, 14.64269194, 121.02663382),
    (14.6169330,  121.0439476,  14.61502291, 121.04686011),
    (14.6371277,  120.9933058,  14.6362268,   120.9930675),
    (14.62603176, 120.98969815, 14.63691378, 120.99343081),
    (14.61498523, 121.04694000, 14.616487822, 121.051593070)
]

for idx, (lat1, lon1, lat2, lon2) in enumerate(paths_to_remove, start=1):
    print(f"\n📍 Processing path {idx}: ({lat1}, {lon1}) → ({lat2}, {lon2})")
    try:
        start_node = ox.distance.nearest_nodes(G_major, lon1, lat1)
        end_node = ox.distance.nearest_nodes(G_major, lon2, lat2)
        print(f"Nearest start node: {start_node}, end node: {end_node}")
        path_nodes = nx.shortest_path(G_major, source=start_node, target=end_node, weight="length")
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        print(f"🗺 Found path {idx} with {len(path_nodes)} nodes.")
        G_major.remove_edges_from(path_edges)
        G_major.remove_nodes_from(path_nodes)
        print(f"✅ Removed {len(path_edges)} edges and {len(path_nodes)} nodes for path {idx}.")
    except nx.NetworkXNoPath:
        print(f"⚠️ No path found for coordinates {idx}, skipping.")
    except Exception as e:
        print(f"⚠️ Error removing path {idx}: {e}")

# 6. Ensure connectivity (largest connected component)
if not nx.is_connected(G_major.to_undirected()):
    print("⚠️ Graph not fully connected, extracting largest component...")
    largest_cc = max(nx.connected_components(G_major.to_undirected()), key=len)
    G_major = G_major.subgraph(largest_cc).copy()
    print(f"✅ Largest component kept (n_nodes={G_major.number_of_nodes()}, n_edges={G_major.number_of_edges()})")

# 7. Define output folder
output_folder = r"D:\Quezon_City\data\processed"
os.makedirs(output_folder, exist_ok=True)

# 8. Save outputs
ox.save_graphml(G_major, os.path.join(output_folder, "qc_roads_major.graphml"))
ox.save_graph_geopackage(G_major, os.path.join(output_folder, "qc_roads_major.gpkg"))

# 9. Plot and save PNG
print("🖼 Saving visualization...")
fig, ax = ox.plot_graph(
    G_major,
    bgcolor="black",
    node_color="white",
    edge_color="gray",
    node_size=5,
    edge_linewidth=0.5,
    show=False,
    close=False
)
plt.title("Major Roads of Quezon City (Cleaned + Path Removed)", color="white")
plt.savefig(
    os.path.join(output_folder, "qc_roads_major.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="black"
)
plt.close(fig)

# 10. Print summary
print("✅ Major roads extracted, cleaned, and visualized!")
print(f"Final graph: {G_major.number_of_nodes()} nodes, {G_major.number_of_edges()} edges")
print(f"Files saved to: {output_folder}")

