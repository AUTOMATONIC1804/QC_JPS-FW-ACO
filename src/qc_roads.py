import osmnx as ox
import os

raw_dir = r"D:\Quezon_City\data\raw"
os.makedirs(raw_dir, exist_ok=True)

place_name = "Quezon City, Philippines"
out_file = os.path.join(raw_dir, "qc_roads_osm.graphml")

print("📥 Downloading road network for:", place_name)
G = ox.graph_from_place(place_name, network_type="drive")

ox.save_graphml(G, out_file)
print("✅ Raw road graph saved to:", out_file)
print("Nodes:", len(G.nodes), "| Edges:", len(G.edges))
