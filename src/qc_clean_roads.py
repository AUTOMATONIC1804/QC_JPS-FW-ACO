#Cleaning of the QC Road Network
import osmnx as ox
import networkx as nx
import geopandas as gpd
import os

raw_dir = r"D:\Quezon_City\data\raw"
proc_dir = r"D:\Quezon_City\data\processed"
os.makedirs(proc_dir, exist_ok=True)

in_graph = os.path.join(raw_dir, "qc_roads_osm.graphml")
out_graph = os.path.join(proc_dir, "qc_roads_clean.graphml")
out_lines = os.path.join(proc_dir, "qc_roads_polylines.geojson")

print("📥 Loading raw graph...")
G = ox.load_graphml(in_graph)

# 1) Skip simplify_graph (already simplified by graph_from_place)

# 2) Keep the largest connected component
largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
G = G.subgraph(largest_cc).copy()

# 3) (Optional) Remove certain road types if you want only major drivable roads
remove = [
    (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
    if d.get("highway") in ["service", "track", "footway", "path"]
]
G.remove_edges_from(remove)

# Save cleaned graph
ox.save_graphml(G, out_graph)
print("✅ Cleaned graph saved:", out_graph)

# Save edges as polylines for QGIS
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
gdf_edges.to_file(out_lines, driver="GeoJSON")
print("✅ Road polylines saved:", out_lines)

print("Nodes:", len(G.nodes), "| Edges:", len(G.edges))

