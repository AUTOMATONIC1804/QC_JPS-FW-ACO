import geopandas as gpd
import pandas as pd
from pathlib import Path

# --------------------------
# INPUT FILES
# --------------------------

# Set A – Regalado–SM North
a_full_route = r"D:\ROUTES\Regalado-SM North\JPS\aco_jps_full_route.geojson"
a_stations = r"D:\ROUTES\Regalado-SM North\JPS\aco_jps_stations.geojson"
a_buffers = r"D:\ROUTES\Regalado-SM North\JPS\aco_jps_station_buffers.geojson"

# Set B – Sacred Heart to Regalado
b_full_route = r"D:\ROUTES\Sacred Heart to Regalado\JPS\aco_jps_full_route.geojson"
b_stations = r"D:\ROUTES\Sacred Heart to Regalado\JPS\aco_jps_stations.geojson"
b_buffers = r"D:\ROUTES\Sacred Heart to Regalado\JPS\aco_jps_station_buffers.geojson"

# --------------------------
# OUTPUT DIRECTORY
# --------------------------

out_dir = Path(r"D:\Quezon_City\data\outputs\aco")
out_dir.mkdir(parents=True, exist_ok=True)

# Output paths
out_full_route = out_dir / "aco_jps_full_route.geojson"
out_stations = out_dir / "aco_jps_stations.geojson"
out_buffers = out_dir / "aco_jps_station_buffers.geojson"

# --------------------------
# HELPER: merge two GeoDataFrames
# --------------------------

def merge_geodataframes(path1, path2):
    g1 = gpd.read_file(path1)
    g2 = gpd.read_file(path2)

    # Align columns (so concatenation keeps everything)
    g = gpd.GeoDataFrame(
        pd.concat([g1, g2], ignore_index=True),
        crs=g1.crs
    )
    return g

# --------------------------
# MERGE EACH FILE TYPE
# --------------------------

print("Merging full route…")
full_route_merged = merge_geodataframes(a_full_route, b_full_route)
full_route_merged.to_file(out_full_route, driver="GeoJSON")

print("Merging stations…")
stations_merged = merge_geodataframes(a_stations, b_stations)
stations_merged.to_file(out_stations, driver="GeoJSON")

print("Merging buffers…")
buffers_merged = merge_geodataframes(a_buffers, b_buffers)
buffers_merged.to_file(out_buffers, driver="GeoJSON")

print("\nCompleted!")
print(f"Files saved inside: {out_dir}")
