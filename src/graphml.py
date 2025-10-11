import osmnx as ox

# Load your graph
graph_path = "data/processed/qc_roads_major.graphml"
G = ox.load_graphml(graph_path)

# Convert to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# Export to GeoJSON
nodes.to_file("data/processed/qc_roads_major_nodes.geojson", driver="GeoJSON")
edges.to_file("data/processed/qc_roads_major_edges.geojson", driver="GeoJSON")

print("[OK] Exported nodes and edges for QGIS:")
print(" - data/processed/qc_roads_major_nodes.geojson")
print(" - data/processed/qc_roads_major_edges.geojson")
