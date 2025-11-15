"""
ACO Dijkstra Runner (Route Generator)
-------------------------------------
Generates the true traversable corridor between ACO-selected stations
using Dijkstra on a local graph built from FW Dijkstra roads.

Workflow:
  0) Automatically run ACO Station Runner to generate station nodes
  1) Load ACO-selected station nodes (GeoJSON)
  2) Load FW Dijkstra roads and build local graph (like fw_dijkstra_runner)
  3) For each consecutive station pair:
       - Snap to nearest graph nodes
       - Run Dijkstra shortest path (length-weighted)
       - Save per-segment GeoJSON + metrics
  4) Merge all path segments into a final route
  5) Export route + report CSV
"""

from pathlib import Path
from typing import Union, Optional
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from src.algorithms.aco.aco_station_runner import run_aco_jps

WGS84 = "EPSG:4326"
METRIC = "EPSG:3857"


def _build_local_graph_from_roads(roads_path: Path) -> nx.Graph:
    """
    Build a local graph from FW Dijkstra roads GeoJSON.
    Same approach as compute_local_dijkstra_matrix in fw_dijkstra_runner.
    """
    print(f"🛣️ Loading FW Dijkstra roads: {roads_path}")
    roads_gdf = gpd.read_file(roads_path).to_crs(METRIC)
    
    G_local = nx.Graph()
    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = [pt for line in geom.geoms for pt in line.coords]
        else:
            continue
        for a, b in zip(coords[:-1], coords[1:]):
            dist = Point(a).distance(Point(b))
            G_local.add_edge(a, b, weight=dist)
    
    print(f"🧱 Built local Dijkstra graph: {len(G_local.nodes)} nodes, {len(G_local.edges)} edges")
    return G_local


def _snap_to_graph(graph: nx.Graph, point_m):
    """Snap a metric Point geometry to the nearest graph coordinate (return coord tuple and meter offset)."""
    graph_coords = np.array(list(graph.nodes))
    target = np.array([point_m.x, point_m.y], dtype=float)
    
    deltas = graph_coords - target
    dists = np.linalg.norm(deltas, axis=1)
    idx = int(np.argmin(dists))
    nearest_coord = tuple(graph_coords[idx])
    return nearest_coord, float(dists[idx])


def _build_path_line(graph: nx.Graph, path_coords) -> LineString:
    """Construct a LineString from path coordinates."""
    coords_list = [coord for coord in path_coords]
    return LineString(coords_list)


def run_aco_dijkstra_route(
    method: str = "dijkstra",
    stations_fp: Optional[Union[str, Path]] = None,
    roads_fp: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "data/outputs/aco",
):
    """
    Run ACO station optimization + graph-based Dijkstra routing between stations.
    Uses FW Dijkstra roads to build local graph (like JPS/A* use rasterized roads).
    """
    method_upper = method.upper()
    runner_total_start = time.perf_counter()

    # Stage 0: Ensure we have stations
    print(f"=== 🚉 STAGE 1: RUNNING ACO {method_upper} STATION OPTIMIZATION ===")
    stage1_start = time.perf_counter()
    stage1_result = run_aco_jps(n_stations=9, method=method)
    stage1_raw = time.perf_counter() - stage1_start
    stage1_compute_s = float(stage1_result.get("compute_time_s", stage1_raw)) if stage1_result else stage1_raw
    stage1_wait_s = float(stage1_result.get("interactive_wait_s", 0.0)) if stage1_result else 0.0
    if stage1_compute_s < 0:
        stage1_compute_s = max(0.0, stage1_raw - stage1_wait_s)
    print(f"[OK] Stage 1 runtime: {stage1_compute_s * 1000:.2f} ms ({stage1_compute_s:.2f} s)")
    print(f"\n✅ Stations generated successfully — proceeding to {method_upper} route generation...\n")

    # Stage 1: Route generation
    print(f"=== 🚆 STAGE 2: RUNNING ACO {method_upper} ROUTE GENERATION ===")
    stage2_start = time.perf_counter()

    # Resolve paths (auto-construct like JPS/A* runners)
    stations_fp = Path(stations_fp or f"data/outputs/aco/aco_{method}_stations.geojson")
    roads_fp = Path(roads_fp or f"data/outputs/floyd_warshall/fw_{method}_roads.geojson")
    outdir = Path(output_dir)
    segments_dir = outdir / f"aco_{method}_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    if not stations_fp.exists():
        raise FileNotFoundError(f"Missing station file: {stations_fp}")
    if not roads_fp.exists():
        raise FileNotFoundError(f"Missing FW roads file: {roads_fp}")

    stations_wgs = gpd.read_file(stations_fp).to_crs(WGS84).reset_index(drop=True)
    stations_m = stations_wgs.to_crs(METRIC)
    print(f"📍 Loaded {len(stations_wgs)} ACO-selected station nodes.")

    # Build local graph from FW roads (like JPS/A* rasterize roads)
    print(f"🛣️ Building local graph from FW {method_upper} roads...")
    G_local = _build_local_graph_from_roads(roads_fp)

    segment_records = []
    segment_geoms = []
    total_time_ms = 0.0
    total_length_m = 0.0

    for idx in range(len(stations_m) - 1):
        start_geom = stations_m.geometry.iloc[idx]
        end_geom = stations_m.geometry.iloc[idx + 1]

        # Snap to nearest graph coordinates (metric CRS)
        start_coord, dist_start = _snap_to_graph(G_local, start_geom)
        end_coord, dist_end = _snap_to_graph(G_local, end_geom)

        print(f"▶️ Segment {idx+1}/{len(stations_m)-1}: {start_coord} → {end_coord} (snap: {dist_start:.1f}m, {dist_end:.1f}m)")
        seg_t0 = time.perf_counter()

        try:
            path_coords = nx.shortest_path(G_local, start_coord, end_coord, weight="weight")
            length_m = nx.shortest_path_length(G_local, start_coord, end_coord, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            elapsed_ms = (time.perf_counter() - seg_t0) * 1000
            total_time_ms += elapsed_ms
            segment_records.append({
                "segment": f"{idx}-{idx+1}",
                "distance_m": np.nan,
                "runtime_ms": round(elapsed_ms, 2),
                "status": "failed",
                "start_node": str(start_coord),
                "end_node": str(end_coord),
                "snap_offset_start_m": round(dist_start, 2),
                "snap_offset_end_m": round(dist_end, 2),
            })
            print(f"❌ No path found for segment {idx}-{idx+1}")
            continue

        elapsed_ms = (time.perf_counter() - seg_t0) * 1000
        total_time_ms += elapsed_ms
        total_length_m += length_m

        # Build geometry (metric); convert to WGS for export
        segment_line_m = _build_path_line(G_local, path_coords)
        segment_line_wgs = gpd.GeoSeries([segment_line_m], crs=METRIC).to_crs(WGS84).iloc[0]

        segment_geoms.append(segment_line_wgs)
        segment_fp = segments_dir / f"segment_{idx}_{idx+1}.geojson"
        gpd.GeoDataFrame(
            {
                "segment": [f"{idx}-{idx+1}"],
                "distance_m": [length_m],
                "runtime_ms": [elapsed_ms],
                "start_node": [str(start_coord)],
                "end_node": [str(end_coord)],
                "snap_offset_start_m": [dist_start],
                "snap_offset_end_m": [dist_end],
            },
            geometry=[segment_line_wgs], crs=WGS84
        ).to_file(segment_fp, driver="GeoJSON")

        segment_records.append({
            "segment": f"{idx}-{idx+1}",
            "distance_m": round(length_m, 2),
            "runtime_ms": round(elapsed_ms, 2),
            "status": "ok",
            "start_node": str(start_coord),
            "end_node": str(end_coord),
            "snap_offset_start_m": round(dist_start, 2),
            "snap_offset_end_m": round(dist_end, 2),
        })
        print(f"✅ Segment {idx}-{idx+1}: {length_m:.2f} m ({elapsed_ms:.1f} ms)")

    # Merge segments
    if segment_geoms:
        merged_coords = [pt for geom in segment_geoms for pt in geom.coords]
        merged_line = LineString(merged_coords)
        gpd.GeoDataFrame(
            {"role": ["path"], "length_m": [total_length_m]},
            geometry=[merged_line],
            crs=WGS84
        ).to_file(outdir / f"aco_{method}_full_route.geojson", driver="GeoJSON")
        print(f"✅ Saved merged route → aco_{method}_full_route.geojson")
    else:
        print("⚠️ No valid segments to merge — route incomplete.")

    # Save report
    if segment_records:
        df_report = pd.DataFrame(segment_records)
        df_report.loc[len(df_report.index)] = {
            "segment": "TOTAL",
            "distance_m": round(total_length_m, 2),
            "runtime_ms": round(total_time_ms, 2),
            "status": "complete" if segment_geoms else "incomplete",
            "start_node": "",
            "end_node": "",
            "snap_offset_start_m": "",
            "snap_offset_end_m": "",
        }
        df_report.to_csv(outdir / f"aco_{method}_route_report.csv", index=False)
        print(f"🧾 Saved route report → aco_{method}_route_report.csv")

    stage2_elapsed = time.perf_counter() - stage2_start
    total_compute_s = stage1_compute_s + stage2_elapsed

    print(f"\n=== ✅ ACO {method_upper} Route Generation Complete ===")
    print(f"📏 Total distance: {total_length_m:.2f} m")
    print(f"[OK] Stage 1 runtime: {stage1_compute_s * 1000:.2f} ms ({stage1_compute_s:.2f} s)")
    print(f"[OK] Stage 2 runtime: {stage2_elapsed * 1000:.2f} ms ({stage2_elapsed:.2f} s)")
    print(f"[OK] ACO total runtime (Stage 1 + Stage 2): {total_compute_s * 1000:.2f} ms ({total_compute_s:.2f} s)")
    print("=============================================")


if __name__ == "__main__":
    run_aco_dijkstra_route()
 