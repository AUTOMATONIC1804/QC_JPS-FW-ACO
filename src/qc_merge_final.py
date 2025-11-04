"""
qc_merge_final.py
-----------------------------------------
Merges:
    - qc_pois_clean_points.geojson
    - qc_pois_clean_polygons.geojson

Rules:
    • Keep points only from the dedicated points file.
    • Remove points and lines from the polygon file.
    • Keep only polygon geometries from the polygon file.
    • Keep only columns: Amenity, Building, Landuse, Name, Shop
    • Save as: qc_merge_final.geojson
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

# === Paths ===
base_dir = Path(r"D:\Quezon_City\data\processed") 
points_path = base_dir / "qc_pois_clean_points.geojson"
polygons_path = base_dir / "qc_pois_clean_polygons.geojson"
output_path = base_dir / "qc_merge_final.geojson"

# === Load ===
print("📂 Loading GeoJSON files...")
points_gdf = gpd.read_file(points_path)
polygons_gdf = gpd.read_file(polygons_path)

print(f"  → Points:   {len(points_gdf)} features")
print(f"  → Polygons: {len(polygons_gdf)} total features before filtering")

# === Filter polygon file: keep only Polygon + MultiPolygon ===
polygons_only = polygons_gdf[polygons_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
print(f"  → After filtering: {len(polygons_only)} polygons kept")

# === Keep only required columns ===
keep_cols = ["name", "amenity", "building", "landuse", "shop", "geometry"]

# Ensure missing columns exist before selecting
for col in keep_cols:
    if col not in polygons_only.columns:
        polygons_only[col] = None
    if col not in points_gdf.columns:
        points_gdf[col] = None

polygons_only = polygons_only[keep_cols]
points_gdf = points_gdf[keep_cols]

# === Merge ===
merged = gpd.GeoDataFrame(pd.concat([points_gdf, polygons_only], ignore_index=True), crs=points_gdf.crs)
print(f"✅ Merged total: {len(merged)} features")

# === Save ===
merged.to_file(output_path, driver="GeoJSON")
print(f"💾 Saved merged file as: {output_path}")
