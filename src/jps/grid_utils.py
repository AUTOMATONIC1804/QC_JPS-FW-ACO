# src/jps/grid_utils.py
import numpy as np
import rasterio

def load_clean_grid(tif_path="data/inputs/processed/qc_grid_clean.tif"):
    """
    Load cleaned QGIS raster (GeoTIFF) into a NumPy grid for JPS.
    Returns (grid, transform, crs).
    
    - grid: numpy 2D array (1=road, 0=obstacle)
    - transform: rasterio Affine transform (maps pixel <-> coordinates)
    - crs: coordinate reference system
    """
    with rasterio.open(tif_path) as src:
        grid = src.read(1)  # first band
        transform = src.transform
        crs = src.crs

    print(f"[OK] Loaded cleaned grid from {tif_path}")
    print(f"Grid shape: {grid.shape}, CRS: {crs}")
    return grid, transform, crs


def cell_to_coords(row, col, transform):
    """
    Convert grid cell indices -> real-world coordinates.
    """
    x, y = transform * (col + 0.5, row + 0.5)  # center of cell
    return (x, y)


def coords_to_cell(x, y, transform):
    """
    Convert real-world coordinates -> grid cell indices.
    """
    col, row = ~transform * (x, y)
    return int(row), int(col)
