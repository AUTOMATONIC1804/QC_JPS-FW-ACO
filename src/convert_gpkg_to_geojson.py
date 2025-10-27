"""
convert_gpkg_to_geojson.py — Auto Export + Enhanced Connectivity Fix v2.1
---------------------------------------------------------------------------
Automatically converts the GeoPackage:
    data/processed/qc_roads_major.gpkg
into:
    data/processed/qc_roads_major_nodes.geojson
    data/processed/qc_roads_major_edges.geojson

Then:
    - Merges nearby road segments (< 5 m apart)
    - Fills small connection gaps with buffer-union merge
    - Fixes broken connectivity and overlaps
    - Saves cleaned roads as:
        data/processed/qc_roads_major_edges.geojson
"""

import geopandas as gpd
from shapely.ops import linemerge, unary_union, snap
from shapely.geometry import MultiLineString
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
        return

    print("\n🧩 Proceeding to fix road connectivity ...")
    fix_road_connectivity(OUTPUT_DIR / "qc_roads_major_edges.geojson")

    print("\n🎉 Conversion and cleaning complete.")
    print("➡️ Use: data/processed/qc_roads_major_edges.geojson in fw_jps_runner")

# ---------------------------------------------------
# Connectivity Fixer (Enhanced)
# ---------------------------------------------------
def fix_road_connectivity(edges_path):
    print(f"📂 Loading {edges_path}")
    try:
        roads = gpd.read_file(edges_path)
    except Exception as e:
        print(f"❌ ERROR: Could not read edges file: {e}")
        return

    roads = roads.to_crs("EPSG:3857")
    print(f"🔧 Merging {len(roads)} road segments ...")

    # 1️⃣ Merge all geometries into one MultiLineString
    merged = unary_union(roads.geometry)

    # 2️⃣ Snap endpoints that are close (< 5 m instead of 2)
    snapped = snap(merged, merged, 5.0)

    # 3️⃣ Buffer–unary union trick to close tiny gaps at junctions (< 2 m)
    buffered = unary_union([geom.buffer(2.0) for geom in roads.geometry])
    merged_fill = unary_union(buffered)

    # Convert buffer polygons back to lines by taking their boundaries
    merged_lines = merged_fill.boundary

    # 4️⃣ Merge connected lines again
    if isinstance(merged_lines, MultiLineString):
        fixed = linemerge(merged_lines)
    else:
        fixed = merged_lines

    # 5️⃣ Explode back into LineStrings
    if isinstance(fixed, MultiLineString):
        clean_roads = gpd.GeoDataFrame(geometry=list(fixed.geoms), crs="EPSG:3857")
    else:
        clean_roads = gpd.GeoDataFrame(geometry=[fixed], crs="EPSG:3857")

    # 6️⃣ Optional final simplification (tiny precision clean)
    clean_roads["geometry"] = clean_roads.geometry.simplify(0.5, preserve_topology=True)

    # 7️⃣ Save the fixed roads OVER the same file name
    clean_out = edges_path.parent / "qc_roads_major_edges.geojson"
    clean_roads.to_crs("EPSG:4326").to_file(clean_out, driver="GeoJSON")

    print(f"✅ Fixed connectivity: {len(clean_roads)} continuous road lines.")
    print(f"💾 Saved → {clean_out}")

# ---------------------------------------------------
# Entry
# ---------------------------------------------------
if __name__ == "__main__":
    convert_gpkg()
