# clean_pois_qc.py
# ------------------------------------------------------
# Cleans and filters both POI polygons and points before scoring
# with safe centroid computation (no CRS warnings)
# ------------------------------------------------------

import geopandas as gpd
import pandas as pd
import os

# === Paths ===
raw_dir = r"D:\Quezon_City\data\raw"
processed_dir = r"D:\Quezon_City\data\processed"

polys_in   = os.path.join(raw_dir, "qc_pois_1_polygons.geojson")
points_in  = os.path.join(raw_dir, "qc_pois_1_points.geojson")

polys_out  = os.path.join(processed_dir, "qc_pois_clean_polygons.geojson")
points_out = os.path.join(processed_dir, "qc_pois_clean_points.geojson")

boundary_path = os.path.join(raw_dir, "qc_boundary.geojson")


# === Helper Function ===
def clean_layer(in_path, out_path, label):
    if not os.path.exists(in_path):
        print(f"⚠ {label} input not found, skipping: {in_path}")
        return None

    pois = gpd.read_file(in_path).to_crs(epsg=4326)
    print(f"\n📂 Loaded {len(pois)} {label} POIs")

    # --- Clip to QC boundary ---
    if os.path.exists(boundary_path):
        qc = gpd.read_file(boundary_path).to_crs(epsg=4326)
        before_clip = len(pois)
        pois = pois[pois.within(qc.unary_union)]
        print(f"🧭 Clipped to Quezon City boundary: {before_clip} → {len(pois)}")
    else:
        print("⚠ No QC boundary found — skipping spatial clip.")

    # --- Keep essential columns only ---
    keep_cols = ["name", "category", "amenity", "building", "shop", "landuse", "geometry"]
    pois = pois[[c for c in keep_cols if c in pois.columns]]

    # --- Remove invalid/missing data ---
    before_drop = len(pois)
    pois = pois.dropna(subset=["category"])
    pois = pois[pois.is_valid & ~pois.is_empty]
    print(f"🧹 Removed invalid/missing geometries: {before_drop} → {len(pois)}")

    # --- Deduplicate safely using projected centroids (no warnings) ---
    # Temporarily reproject to UTM 51N for accurate centroids
    pois_proj = pois.to_crs(epsg=32651)
    centroids = pois_proj.geometry.centroid.to_crs(epsg=4326)

    pois["dup_key"] = pois["name"].fillna("") \
        + "_" + centroids.x.round(6).astype(str) \
        + "_" + centroids.y.round(6).astype(str)

    before_dups = len(pois)
    pois = pois.drop_duplicates(subset=["dup_key"]).drop(columns=["dup_key"])
    after_dups = len(pois)
    print(f"🧩 Removed duplicates: {before_dups} → {after_dups}")

    # --- Save cleaned file ---
    pois.to_file(out_path, driver="GeoJSON")
    print(f"✅ Saved cleaned {label} layer to: {out_path}")

    # --- Summary ---
    print("📊 POI counts by category:")
    print(pois.groupby("category").size())

    return pois


# === Run Cleaning for Both Layers ===
clean_layer(polys_in, polys_out, "polygon")
clean_layer(points_in, points_out, "point")

print("\n🎯 Both layers cleaned and exported successfully!")
