"""
clip_worldpop_to_qc.py
-------------------------------------------
Clips the WorldPop Philippines 2025 raster (100m)
to the Quezon City boundary.

Outputs:
- qc_population_2025.tif (clipped raster)
- qc_population_points.geojson (centroids + population values)
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import numpy as np
import pandas as pd
from pathlib import Path

# === Paths ===
worldpop_tif = Path("data/raw/phl_pop_2025.tif")
qc_boundary = Path("data/inputs/qc_boundary.geojson")
out_tif = Path("data/processed/qc_population_2025.tif")
out_points = Path("data/processed/qc_population_points.geojson")

# === 1. Load Quezon City boundary ===
qc = gpd.read_file(qc_boundary)
qc = qc.to_crs("EPSG:4326")  # Ensure same CRS as WorldPop raster

# === 2. Open and clip raster ===
with rasterio.open(worldpop_tif) as src:
    qc_mask = [qc.geometry.union_all()]  # merge polygons properly
    out_image, out_transform = mask(src, qc_mask, crop=True)
    out_meta = src.meta.copy()

# === 3. Update metadata ===
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

# === 4. Save clipped raster ===
out_tif.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"✅ Saved clipped raster: {out_tif}")

# === 5. Convert raster to point GeoDataFrame ===
# Extract pixel centers and values
with rasterio.open(out_tif) as src:
    band = src.read(1)
    rows, cols = np.where(band > 0)  # keep nonzero population cells
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    pop_values = band[rows, cols]

# Create GeoDataFrame
points = gpd.GeoDataFrame({
    "population": pop_values
}, geometry=gpd.points_from_xy(xs, ys), crs="EPSG:4326")

# Save to GeoJSON
points.to_file(out_points, driver="GeoJSON")
print(f"✅ Saved population points: {out_points}")
print(f"🧮 Total points: {len(points)}")
