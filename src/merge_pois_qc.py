# merge_pois_qc.py
# ------------------------------------------------------
# Merging of Separate GeoJSON POI Files (Overpass Exports)
# + Validation, Integrity Checks, and Safe Centroid Conversion
# ------------------------------------------------------

import geopandas as gpd
import pandas as pd
import os

# === Paths ===
raw_dir = r"D:\Quezon_City\data\raw"
out_merged = os.path.join(raw_dir, "qc_pois_clean_merged.geojson")

# === Your Overpass-exported POI files ===
files = [
    ("qc_pois_all.geojson", "mixed"),
    #("qc_pois_health.geojson", "health"), 
    #("qc_pois_education.geojson", "education"),
    #("qc_pois_transport.geojson", "transport"),
    #("qc_pois_commercial.geojson", "commercial"),
    #("qc_pois_government.geojson", "government"),
    #("qc_pois_residential.geojson", "residential"), 
    #("qc_pois_recreation.geojson", "recreation"),
]

gdfs = []
meta_summary = []

# === Load and validate each file ===
for fname, cat in files:
    path = os.path.join(raw_dir, fname)
    if not os.path.exists(path):
        print(f"⚠ Missing {fname}, skipping")
        continue

    g = gpd.read_file(path)
    if g.empty:
        print(f"⚠ {fname} is empty, skipping")
        continue

    n_before = len(g)

    # Ensure 'name' column exists
    if "name" not in g.columns:
        g["name"] = None

    g["category"] = cat

    # Basic geometry validation
    geom_types = g.geometry.geom_type.value_counts().to_dict()
    meta_summary.append({
        "file": fname,
        "category": cat,
        "records": n_before,
        "geom_types": geom_types
    })

    gdfs.append(g)

# === Merge all POI layers ===
if not gdfs:
    raise SystemExit("❌ No POI files found. Make sure you exported them from Overpass Turbo!")

pois = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
pois = pois.to_crs(epsg=4326)

print("\n✅ Merge Complete")
print("📦 Total merged records:", len(pois))
print("🧱 Columns preserved:", list(pois.columns))

# === Safe Centroid Conversion (UTM projection) ===
print("\n🧭 Generating centroids safely using EPSG:32651...")
pois_points = pois.copy()

# Step 1: Project to UTM Zone 51N (meters)
pois_utm = pois_points.to_crs(epsg=32651)

# Step 2: Compute centroids in metric space
pois_utm["geometry"] = pois_utm.geometry.centroid

# Step 3: Reproject back to WGS84 (EPSG:4326)
pois_points = pois_utm.to_crs(epsg=4326)

# === Deduplicate by name + coordinates ===
pois_points["dup_key"] = pois_points["name"].fillna("") \
    + "_" + pois_points.geometry.x.round(6).astype(str) \
    + "_" + pois_points.geometry.y.round(6).astype(str)

dup_count_before = pois_points.duplicated(subset=["dup_key"]).sum()
pois_points = pois_points.drop_duplicates(subset=["dup_key"])
dup_count_after = pois_points.duplicated(subset=["dup_key"]).sum()

print(f"\n🔍 Duplicate Summary:")
print(f"Before deduplication: {dup_count_before}")
print(f"After deduplication:  {dup_count_after}")

# === Merge original polygons + centroid points into one unified file ===
print("\n🔗 Combining original geometries and centroids into one dataset...")
pois["geom_type"] = pois.geometry.geom_type.str.lower()
pois_points["geom_type"] = "centroid"

merged_all = gpd.GeoDataFrame(
    pd.concat([pois, pois_points], ignore_index=True),
    crs="EPSG:4326"
)

print(f"✅ Combined total features: {len(merged_all)}")

# === Save unified GeoJSON ===
merged_all.to_file(out_merged, driver="GeoJSON")
print(f"💾 Saved unified POIs (original + centroids) to: {out_merged}")

# === Validation Summaries ===
print("\n📊 POI counts by category:")
print(merged_all.groupby("category").size())

print("\n📊 Geometry type breakdown:")
print(merged_all.geometry.type.value_counts())

print("\n📋 Per-file Summary:")
meta_df = pd.DataFrame(meta_summary)
print(meta_df.to_string(index=False))

# === Optional QC boundary sanity check ===
try:
    qc_boundary_path = os.path.join(raw_dir, "qc_boundary.geojson")
    if os.path.exists(qc_boundary_path):
        qc = gpd.read_file(qc_boundary_path).to_crs(epsg=4326)
        within = merged_all.within(qc.unary_union)
        outside_count = (~within).sum()
        print(f"\n🧩 Outlier check: {outside_count} POI features lie outside QC boundary.")
    else:
        print("\n⚠ No qc_boundary.geojson found — skipping outlier check.")
except Exception as e:
    print(f"\n⚠ Boundary check skipped due to error: {e}")

print("\n🎯 POI merge + validation complete!")
