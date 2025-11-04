"""
merge_original_with_modified_exact.py
-----------------------------------------------------
Keeps only the features in qc_merge_final.geojson (2301 features)
but adds all the original attribute columns from qc_pois_clean_merged.geojson
(one-to-one: nearest match within a small distance).

Output: qc_pois_all_matched.geojson
"""

import geopandas as gpd
from pathlib import Path

# === Paths ===
base_dir = Path(r"D:\Quezon_City\data\processed")
orig_path = base_dir / "qc_pois_clean_merged.geojson"
mod_path  = base_dir / "qc_merge_final.geojson"
out_path  = base_dir / "qc_pois_all_matched.geojson"

# === Load files ===
print("📂 Loading layers...")
orig = gpd.read_file(orig_path)
mod  = gpd.read_file(mod_path)

print(f"  → Original: {len(orig)} features")
print(f"  → Modified: {len(mod)} features")

# Ensure CRS match
if orig.crs != mod.crs:
    mod = mod.to_crs(orig.crs)
    print(f"  → CRS adjusted to match ({orig.crs})")

# Add a stable id to the modified layer (so we can dedupe reliably after join)
mod = mod.copy()
mod["__mid"] = range(len(mod))

# Nearest LEFT join (1:1), with small distance cap to avoid wrong matches
# Tweak max_distance (in CRS units; degrees if EPSG:4326) by projecting to meters first.
# Project to UTM 51N for a meter-based distance threshold.
use_proj = orig.to_crs(32651)
mod_proj = mod.to_crs(32651)

print("🔗 Joining nearest attributes from original to modified (1:1)...")
jn = gpd.sjoin_nearest(
    mod_proj,
    use_proj,
    how="left",
    max_distance=25,          # meters — adjust if needed
    distance_col="__dist_m"
)

# Keep only the closest original feature per modified feature
jn = jn.sort_values(["__mid", "__dist_m"]).drop_duplicates(subset="__mid", keep="first")

# Bring back to WGS84 (or original CRS)
jn = jn.to_crs(orig.crs)

# Clean columns:
# - Drop right-side geometry if present
# - Remove sjoin helper columns
for col in ["geometry_right", "index_right"]:
    if col in jn.columns:
        jn = jn.drop(columns=[col])

# Ensure geometry is the modified geometry (left)
jn = jn.set_geometry("geometry")

# Optionally drop the distance + helper id from output
if "__dist_m" in jn.columns:
    unmatched = jn["__dist_m"].isna().sum()
else:
    unmatched = 0

jn = jn.drop(columns=["__mid", "__dist_m"], errors="ignore")

print(f"✅ Final merged feature count: {len(jn)} (should match modified: {len(mod)})")
if unmatched:
    print(f"⚠ Note: {unmatched} modified features had no nearby match in original (beyond max_distance).")

# Save result
jn.to_file(out_path, driver="GeoJSON")
print(f"💾 Saved enriched layer as: {out_path}")
