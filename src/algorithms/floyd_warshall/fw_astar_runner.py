"""
fw_astar_runner.py
-----------------------------
Floyd–Warshall pipeline for A*-generated route.

Assumptions
===========
- A* route file is expected in EPSG:3857 (metric grid domain).
- If the file CRS differs/missing, fw_core applies/repairs CRS handling.

Reads
=====
- data/outputs/astar_path.geojson

Writes
======
- data/outputs/floyd_warshall/fw_astar_buffer.geojson
- data/outputs/floyd_warshall/fw_astar_roads.geojson
- data/outputs/floyd_warshall/fw_astar_points.geojson
- data/outputs/floyd_warshall/fw_astar_D.npy
- data/outputs/floyd_warshall/fw_astar_FW.npy
"""

from .fw_core import FWConfig, run_fw_pipeline

def main(
    route_geojson="data/outputs/astar_path.geojson",
    buffer_m=5000,
    spacing_m=500,
    output_dir="data/outputs/floyd_warshall",
    use_osmnx=False,
    edges_gpkg=None,
    edges_layer=None,
):
    cfg = FWConfig(
        buffer_m=buffer_m,
        spacing_m=spacing_m,
        use_osmnx=use_osmnx,
        edges_gpkg=edges_gpkg,
        edges_layer=edges_layer,
        expected_route_crs="EPSG:3857",   # A* route expected CRS
    )

    info = run_fw_pipeline(route_geojson, output_dir, "fw_astar", cfg)

    print("=== 🧭 A* → FW (buffered corridor) ===")
    print("Params:", info["params"])
    print("Counts:", info["counts"])
    print("Timings (ms):", info["timings_ms"])
    print(f"Outputs → {info['outputs']}")
    print("=== ✅ Done (A* → FW) ===")


if __name__ == "__main__":
    main()
