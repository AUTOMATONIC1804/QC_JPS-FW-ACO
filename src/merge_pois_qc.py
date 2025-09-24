# Merging of Separate GeoJSON POI Files
import geopandas as gpd
import pandas as pd
import os

raw_dir = r"D:\Quezon_City\data\raw"
out_poly = os.path.join(raw_dir, "qc_pois_1_polygons.geojson")
out_pts  = os.path.join(raw_dir, "qc_pois_1_points.geojson")

# Your actual Overpass-exported files
files = [
    ("qc_pois_health.geojson", "health"),
    ("qc_pois_education.geojson", "education"),
    ("qc_pois_transport.geojson", "transport"),
    ("qc_pois_commercial.geojson", "commercial"),
    ("qc_pois_government.geojson", "government"),
    ("qc_pois_residential.geojson", "residential"),
    ("qc_pois_recreation.geojson", "recreation"),  # 🌳 Parks / leisure
]

gdfs = []
for fname, cat in files:
    path = os.path.join(raw_dir, fname)
    if not os.path.exists(path):
        print(f"⚠ Missing {fname}, skipping")
        continue
    g = gpd.read_file(path)
    if g.empty:
        print(f"⚠ {fname} is empty, skipping")
        continue
    if "name" not in g.columns:
        g["name"] = None
    g["category"] = cat
    gdfs.append(g)

if not gdfs:
    raise SystemExit("No POI files found. Make sure you exported them from Overpass Turbo!")

# Merge into one GeoDataFrame
pois = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
pois = pois.to_crs(epsg=4326)

# --- Save polygons version ---
pois.to_file(out_poly, driver="GeoJSON")
print("✅ Saved polygons to:", out_poly)

# --- Save centroids version (for algorithms) ---
pois_points = pois.copy()
pois_points["geometry"] = pois_points.geometry.centroid

# Deduplicate by name + coordinates
pois_points["dup_key"] = pois_points["name"].fillna("") \
    + "_" + pois_points.geometry.x.round(6).astype(str) \
    + "_" + pois_points.geometry.y.round(6).astype(str)
pois_points = pois_points.drop_duplicates(subset=["dup_key"])

pois_points.to_file(out_pts, driver="GeoJSON")
print("✅ Saved centroid points to:", out_pts)

# --- Summary counts ---
print("\n📊 POI counts by category:")
print(pois.groupby("category").size())
