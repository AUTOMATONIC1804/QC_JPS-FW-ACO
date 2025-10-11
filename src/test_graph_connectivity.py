import osmnx as ox
import networkx as nx

graph_path = "data/processed/qc_roads_major.graphml"

print("🔍 Loading graph...")
G = ox.load_graphml(graph_path)

G_u = G.to_undirected()

print("✅ Graph loaded.")
print("Is connected:", nx.is_connected(G_u))
print("Number of connected components:", nx.number_connected_components(G_u))

# Optional: print size of each component
components = sorted(nx.connected_components(G_u), key=len, reverse=True)
print("Component sizes (top 5):", [len(c) for c in components[:5]])
