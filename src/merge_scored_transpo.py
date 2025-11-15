# merge_scored_transpo.py
# ------------------------------------------------------
# Merges qc_pois_final_scored.geojson with qc_pois_transportation_clean.geojson
# Uses schema from scored file, merges features with matching IDs
# ------------------------------------------------------

import geopandas as gpd
import pandas as pd
import os

# === Paths ===
raw_dir = r"D:\Quezon_City\data\raw"
processed_dir = r"D:\Quezon_City\data\processed"

scored_path = os.path.join(processed_dir, "qc_pois_final_scored.geojson")
transpo_path = os.path.join(raw_dir, "qc_pois_transportation_clean.geojson")
output_path = os.path.join(processed_dir, "qc_pois_final_scored_transpo.geojson")

print("=" * 60)
print("Merging Scored POIs with Transportation POIs")
print("=" * 60)

# === Load both files ===
if not os.path.exists(scored_path):
    raise FileNotFoundError(f"Missing scored POI file: {scored_path}")
if not os.path.exists(transpo_path):
    raise FileNotFoundError(f"Missing transportation POI file: {transpo_path}")

print(f"\n📂 Loading scored POIs: {scored_path}")
scored = gpd.read_file(scored_path).to_crs(epsg=4326)
print(f"   Loaded {len(scored)} features")

print(f"\n📂 Loading transportation POIs: {transpo_path}")
transpo = gpd.read_file(transpo_path).to_crs(epsg=4326)
print(f"   Loaded {len(transpo)} features")

# === Identify ID columns ===
id_cols_scored = [col for col in scored.columns if col.lower() in ["id", "@id"]]
id_cols_transpo = [col for col in transpo.columns if col.lower() in ["id", "@id"]]

print(f"\n🔍 ID columns in scored file: {id_cols_scored}")
print(f"🔍 ID columns in transpo file: {id_cols_transpo}")

# === Use schema from scored file as base ===
base_columns = list(scored.columns)
print(f"\n📋 Base schema ({len(base_columns)} columns) from scored file:")
print(f"   {', '.join(base_columns[:10])}..." if len(base_columns) > 10 else f"   {', '.join(base_columns)}")

# === Prepare ID matching ===
# Try to find matching ID field
id_field = None
if "id" in scored.columns and "id" in transpo.columns:
    id_field = "id"
elif "@id" in scored.columns and "@id" in transpo.columns:
    id_field = "@id"
elif "id" in scored.columns:
    # Create id in transpo if missing
    if "id" not in transpo.columns and "@id" in transpo.columns:
        transpo["id"] = transpo["@id"]
        id_field = "id"
elif "@id" in scored.columns:
    if "@id" not in transpo.columns and "id" in transpo.columns:
        transpo["@id"] = transpo["id"]
        id_field = "@id"

if id_field:
    print(f"\n🔗 Using '{id_field}' field for matching")
else:
    print("\n⚠️ No common ID field found - will merge all features without matching")

# === Merge features with matching IDs ===
merged_features = []
matched_ids = set()

if id_field:
    # Get IDs from both files (handle NaN)
    scored_ids = set(scored[id_field].dropna().astype(str))
    transpo_ids = set(transpo[id_field].dropna().astype(str))
    common_ids = scored_ids & transpo_ids
    
    print(f"\n🔗 Found {len(common_ids)} matching IDs")
    
    # Process matching features
    for match_id in common_ids:
        scored_row = scored[scored[id_field].astype(str) == match_id]
        transpo_row = transpo[transpo[id_field].astype(str) == match_id]
        
        if len(scored_row) > 0 and len(transpo_row) > 0:
            # Start with scored row (has the base schema)
            merged_row = scored_row.iloc[0].copy()
            
            # Merge attributes from transpo (fill missing values)
            for col in transpo.columns:
                if col == "geometry":
                    continue
                transpo_val = transpo_row.iloc[0][col]
                if pd.notna(transpo_val):
                    # If column exists in scored, prefer scored value unless it's null
                    if col in merged_row.index:
                        if pd.isna(merged_row[col]) or merged_row[col] is None:
                            merged_row[col] = transpo_val
                    else:
                        # New column from transpo
                        merged_row[col] = transpo_val
            
            # For geometry: prefer polygon over point, otherwise keep scored geometry
            scored_geom = scored_row.iloc[0].geometry
            transpo_geom = transpo_row.iloc[0].geometry
            
            if scored_geom.geom_type in ["Polygon", "MultiPolygon"]:
                merged_row.geometry = scored_geom
            elif transpo_geom.geom_type in ["Polygon", "MultiPolygon"]:
                merged_row.geometry = transpo_geom
            else:
                merged_row.geometry = scored_geom  # Default to scored
            
            merged_features.append(merged_row)
            matched_ids.add(match_id)
    
    print(f"✅ Merged {len(merged_features)} matching features")

# === Add non-matching features from scored file ===
if id_field:
    scored_unmatched = scored[~scored[id_field].astype(str).isin(matched_ids)]
else:
    scored_unmatched = scored

print(f"\n➕ Adding {len(scored_unmatched)} unmatched features from scored file")
for _, row in scored_unmatched.iterrows():
    merged_features.append(row)

# === Add non-matching features from transpo file ===
if id_field:
    transpo_unmatched = transpo[~transpo[id_field].astype(str).isin(matched_ids)]
else:
    transpo_unmatched = transpo

print(f"➕ Adding {len(transpo_unmatched)} unmatched features from transpo file")

# Ensure transpo features have all base columns
for _, row in transpo_unmatched.iterrows():
    transpo_row = pd.Series(dtype=object)
    # Start with base schema (from scored)
    for col in base_columns:
        if col == "geometry":
            transpo_row["geometry"] = row.geometry
        elif col in row.index:
            transpo_row[col] = row[col]
        else:
            transpo_row[col] = None
    
    # Add any additional columns from transpo
    for col in row.index:
        if col not in base_columns and col != "geometry":
            transpo_row[col] = row[col]
    
    merged_features.append(transpo_row)

# === Create merged GeoDataFrame ===
print(f"\n🔗 Creating merged GeoDataFrame...")
merged_gdf = gpd.GeoDataFrame(merged_features, crs="EPSG:4326")

# Ensure all base columns exist
for col in base_columns:
    if col not in merged_gdf.columns and col != "geometry":
        merged_gdf[col] = None

# Reorder columns to match base schema, then add any extras
extra_cols = [col for col in merged_gdf.columns if col not in base_columns]
final_cols = [col for col in base_columns if col in merged_gdf.columns] + extra_cols
merged_gdf = merged_gdf[final_cols]

print(f"✅ Merged dataset: {len(merged_gdf)} features, {len(merged_gdf.columns)} columns")

# === Save merged file ===
print(f"\n💾 Saving merged POIs to: {output_path}")
merged_gdf.to_file(output_path, driver="GeoJSON")
print(f"✅ Saved successfully!")

# === Summary ===
print("\n" + "=" * 60)
print("Merge Summary")
print("=" * 60)
print(f"📊 Total features: {len(merged_gdf)}")
if id_field:
    print(f"🔗 Matched by ID: {len(matched_ids)}")
    print(f"➕ From scored only: {len(scored_unmatched)}")
    print(f"➕ From transpo only: {len(transpo_unmatched)}")

print(f"\n📋 Geometry types:")
print(merged_gdf.geometry.geom_type.value_counts())

if "category" in merged_gdf.columns:
    print(f"\n📊 Categories:")
    print(merged_gdf["category"].value_counts())

print("\n🎯 Merge complete!")

