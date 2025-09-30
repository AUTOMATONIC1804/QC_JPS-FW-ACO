# src/utils/qc_grid.py
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely import wkt
from src.utils.rasterize import graph_to_grid

def make_grid(
    graphml_path="data/processed/qc_roads_major.graphml",
    resolution=10,
    buffer_m=3,
    out_prefix="data/processed/qc_grid"
):
    """
    Generate a raw grid from the QC major roads file.
    Outputs .npy, .png, and .tif for later use (manual cleaning in QGIS).
    Grid encoding: 1 = road, 0 = obstacle
    """

    # 1. Load graph
    G = nx.read_graphml(graphml_path)

    # 2. Fix geometries (convert WKT strings → Shapely objects if needed)
    for _, _, data in G.edges(data=True):
        if "geometry" in data and isinstance(data["geometry"], str):
            try:
                data["geometry"] = wkt.loads(data["geometry"])
            except Exception:
                pass

    # 3. Convert edges to GeoDataFrame
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # 4. Ensure CRS and reproject
    if edges.crs is None:
        print("⚠️ No CRS found, assuming EPSG:4326")
        edges = edges.set_crs("EPSG:4326")

    edges = edges.to_crs("EPSG:3857")

    # 5. Rasterize from GeoDataFrame
    grid, meta = graph_to_grid(edges, resolution=resolution, buffer_m=buffer_m)

    # 6. Save numpy
    np.save(f"{out_prefix}_raw.npy", grid)

    # 7. Save preview PNG
    plt.imshow(grid, cmap="gray")
    plt.title("QC Major Roads Grid (1=road, 0=obstacle)")
    plt.savefig(f"{out_prefix}_raw.png", dpi=300)
    plt.close()

    # 8. Save GeoTIFF for QGIS
    xmin, ymax = meta["origin"]
    res = meta["resolution"]
    transform = from_origin(xmin, ymax, res, res)

    with rasterio.open(
        f"{out_prefix}_raw.tif",
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=grid.dtype,
        crs=meta["crs"],
        transform=transform,
    ) as dst:
        dst.write(grid, 1)

    print(f"[OK] Grid saved as {out_prefix}_raw.npy/.png/.tif")
    print(f"    Shape: {grid.shape}, Resolution: {res}m")
    return grid, meta

if __name__ == "__main__":
    make_grid()
