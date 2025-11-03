"""
merge_qc_up_with_pois.py
------------------------------------------------------
Merges the special POIs (qc_up.geojson) with the cleaned
POI points and polygons.

Inputs:
- data/processed/qc_up.geojson
- data/processed/qc_pois_clean_points.geojson
- data/processed/qc_pois_clean_polygons.geojson

Outputs (same filenames, overwritten):
- data/processed/qc_pois_clean_points.geojson
- data/processed/qc_pois_clean_polygons.geojson
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
base_dir = Path(r"D:\Quezon_City")
processed_dir = base_dir / "data" / "processed"

up_path = processed_dir / "qc_up.geojson"
points_path = processed_dir / "qc_pois_clean_points.geojson"
polygons_path = processed_dir / "qc_pois_clean_polygons.geojson"

# ------------------------------------------------------------------
# 1️⃣ Load GeoDataFrames
# ------------------------------------------------------------------
print("📂 Loading datasets...")
gdf_up = gpd.read_file(up_path)
gdf_points = gpd.read_file(points_path)
gdf_polygons = gpd.read_file(polygons_path)

print(f"  → U.P. POIs:   {len(gdf_up)}")
print(f"  → Points:      {len(gdf_points)}")
print(f"  → Polygons:    {len(gdf_polygons)}")

# ------------------------------------------------------------------
# 2️⃣ Separate U.P. features into Points & Polygons
# ------------------------------------------------------------------
gdf_up_points = gdf_up[gdf_up.geometry.type == "Point"].copy()
gdf_up_polygons = gdf_up[gdf_up.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

# If there are polygons, also create centroid versions for point dataset
if not gdf_up_polygons.empty:
    centroids = gdf_up_polygons.copy()
    centroids["geometry"] = centroids.geometry.centroid
    gdf_up_points = pd.concat([gdf_up_points, centroids], ignore_index=True)

print(f"📍 U.P. Points (incl. centroids): {len(gdf_up_points)}")
print(f"🧱 U.P. Polygons: {len(gdf_up_polygons)}")

# ------------------------------------------------------------------
# 3️⃣ Align Columns (ensure consistent schema)
# ------------------------------------------------------------------
def align_columns(base, extra):
    cols = sorted(set(base.columns) | set(extra.columns))
    for g in [base, extra]:
        for c in cols:
            if c not in g.columns:
                g[c] = None
    return base[cols], extra[cols]

gdf_points, gdf_up_points = align_columns(gdf_points, gdf_up_points)
gdf_polygons, gdf_up_polygons = align_columns(gdf_polygons, gdf_up_polygons)

# ------------------------------------------------------------------
# 4️⃣ Merge + Deduplicate
# ------------------------------------------------------------------
def merge_and_dedup(base, extra):
    merged = pd.concat([base, extra], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs="EPSG:4326")
    merged["x"] = merged.geometry.centroid.x
    merged["y"] = merged.geometry.centroid.y
    merged = merged.drop_duplicates(subset=["name", "x", "y"], keep="first")
    return merged.drop(columns=["x", "y"])

print("🧩 Merging points...")
merged_points = merge_and_dedup(gdf_points, gdf_up_points)
print(f"✅ Points merged: {len(merged_points)} total")

print("🧩 Merging polygons...")
merged_polygons = merge_and_dedup(gdf_polygons, gdf_up_polygons)
print(f"✅ Polygons merged: {len(merged_polygons)} total")

# ------------------------------------------------------------------
# 5️⃣ Save Outputs (same filenames)
# ------------------------------------------------------------------
merged_points.to_file(points_path, driver="GeoJSON")
merged_polygons.to_file(polygons_path, driver="GeoJSON")

print("💾 Updated files (overwritten):")
print(f"  - Points → {points_path.name}")
print(f"  - Polygons → {polygons_path.name}")
print("✅ Merge complete.")
