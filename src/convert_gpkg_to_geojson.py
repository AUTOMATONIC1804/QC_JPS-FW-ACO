"""
convert_gpkg_to_geojson.py — Auto Export
---------------------------------------
Automatically converts the GeoPackage:
    data/processed/qc_roads_major.gpkg
into two GeoJSON files:
    data/processed/qc_roads_major_nodes.geojson
    data/processed/qc_roads_major_edges.geojson
"""

import geopandas as gpd
from pathlib import Path

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
GPKG_PATH = Path("data/processed/qc_roads_major.gpkg")
OUTPUT_DIR = Path("data/processed")

def convert_gpkg():
    print("📦 Converting GeoPackage → GeoJSON")
    print(f"Input: {GPKG_PATH}")

    if not GPKG_PATH.exists():
        print(f"❌ ERROR: File not found: {GPKG_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export NODES
    try:
        print("🔹 Reading layer: nodes ...")
        nodes = gpd.read_file(GPKG_PATH, layer="nodes")
        nodes_out = OUTPUT_DIR / "qc_roads_major_nodes.geojson"
        nodes.to_file(nodes_out, driver="GeoJSON")
        print(f"✅ Exported {len(nodes)} nodes → {nodes_out}")
    except Exception as e:
        print(f"⚠️ Could not export 'nodes' layer: {e}")

    # Export EDGES
    try:
        print("🔸 Reading layer: edges ...")
        edges = gpd.read_file(GPKG_PATH, layer="edges")
        edges_out = OUTPUT_DIR / "qc_roads_major_edges.geojson"
        edges.to_file(edges_out, driver="GeoJSON")
        print(f"✅ Exported {len(edges)} edges → {edges_out}")
    except Exception as e:
        print(f"⚠️ Could not export 'edges' layer: {e}")

    print("\n🎉 Conversion complete.")

if __name__ == "__main__":
    convert_gpkg()
