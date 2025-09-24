# Major Roads Extraction for QC
import osmnx as ox
import geopandas as gpd
import networkx as nx
import os

# 1. Set place name
place_name = "Quezon City, Philippines"

# 2. Download road network (driveable roads only)
print("⏳ Downloading road network...")
G = ox.graph_from_place(place_name, network_type="drive")

# 3. Convert graph to GeoDataFrame (edges only)
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# 4. Filter edges to major roads only
major_road_types = ["primary", "secondary", "trunk"]  # "tertiary" not included
gdf_major = gdf_edges[gdf_edges["highway"].isin(major_road_types)]

# 5. Create new graph from filtered edges
gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)  # nodes only
G_major = ox.graph_from_gdfs(gdf_nodes, gdf_major)

# 6. Ensure connectivity (keep only the largest connected component)
if not nx.is_connected(G_major.to_undirected()):
    print("⚠️ Graph not fully connected, extracting largest component...")
    largest_cc = max(nx.connected_components(G_major.to_undirected()), key=len)
    G_major = G_major.subgraph(largest_cc).copy()
    print(f"✅ Largest component kept (n_nodes={G_major.number_of_nodes()}, n_edges={G_major.number_of_edges()})")

# 7. Define output folder (your project structure)
output_folder = r"D:\Quezon_City\data\processed"
os.makedirs(output_folder, exist_ok=True)

# 8. Save outputs
ox.save_graphml(G_major, os.path.join(output_folder, "qc_roads_major.graphml"))  # For algorithms
gdf_major.to_file(os.path.join(output_folder, "qc_roads_major_polylines.geojson"), driver="GeoJSON")  # For QGIS

# 9. Print summary
print("✅ Major roads extracted and saved!")
print(f"Total roads before filtering: {len(gdf_edges)}")
print(f"Total roads after filtering: {len(gdf_major)}")
print(f"Final connected graph: {G_major.number_of_nodes()} nodes, {G_major.number_of_edges()} edges")
print(f"Files saved to: {output_folder}")
