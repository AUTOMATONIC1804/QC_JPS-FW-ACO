"""
src/jps/grid_utils.py
Utility functions for loading and visualizing grids.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, mapping


def load_clean_grid(tif_path="data/processed/qc_grid_clean.tif",
                    preview_png="data/outputs/grid_preview.png",
                    preview_geojson="data/outputs/grid_preview.geojson"):
    """
    Load cleaned QGIS raster (GeoTIFF) into a NumPy grid for JPS.
    Also outputs optional PNG + GeoJSON for sanity checking.
    """
    with rasterio.open(tif_path) as src:
        grid = src.read(1)  # first band
        transform = src.transform
        crs = src.crs

    print(f"[OK] Loaded cleaned grid from {tif_path}")
    print(f"Grid shape: {grid.shape}, CRS: {crs}")

    # --- PNG preview ---
    plt.figure(figsize=(8, 10))
    plt.imshow(grid, cmap="gray", interpolation="none")
    plt.title("QC Grid (1=road, 0=obstacle)")
    plt.axis("off")  # hides axis numbers and ticks
    plt.tight_layout()
    plt.savefig(preview_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[OK] Saved grid preview PNG → {preview_png}")

    # --- GeoJSON preview (road cells only) ---
    features = []
    rows, cols = np.where(grid == 1)  # only road cells
    for r, c in zip(rows, cols):
        x_min, y_max = transform * (c, r)        # top-left
        x_max, y_min = transform * (c + 1, r + 1)  # bottom-right
        poly = Polygon([
            (x_min, y_min), (x_min, y_max),
            (x_max, y_max), (x_max, y_min),
            (x_min, y_min)
        ])
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {"value": 1}
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(preview_geojson, "w") as f:
        json.dump(geojson, f)

    print(f"[OK] Saved grid preview GeoJSON → {preview_geojson}")

    return grid, transform, crs


def cell_to_coords(row, col, transform):
    """
    Convert grid cell indices -> real-world coordinates (cell center).
    """
    x, y = transform * (col + 0.5, row + 0.5)
    return (x, y)


def coords_to_cell(x, y, transform):
    """
    Convert real-world coordinates -> grid cell indices (row, col).
    """
    col, row = ~transform * (x, y)
    return int(row), int(col)
