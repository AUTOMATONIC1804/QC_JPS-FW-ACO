"""
aco_detour_refiner.py
--------------------------------------------------
Detour-aware route refinement (geometry-based, no NetworkX)

✅ Works directly with fw_jps_roads.geojson geometries
✅ Accepts ONLY branches that leave the main route and rejoin it later
✅ Uses POI gain vs. added distance decision rule
✅ Counts POIs (points or polygons) that partially intersect a 1 km buffer
✅ Robustly merges detours + handles empty geometries
✅ Logs each accepted detour for transparency
✅ Outputs both refined route and optional detour debug layer
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


# ---------------------------------------------------------
# Configuration & data models
# ---------------------------------------------------------

@dataclass
class DetourConfig:
    """Configuration for geometric detour logic."""
    max_extra_m: float = 1200.0       # Max allowed extra distance for a detour (m)
    min_gain_ratio: float = 0.25      # Min POI gain per km of extra distance
    poi_radius_m: float = 1000.0      # POI influence radius around detour (m)
    touch_tol_m: float = 10.0         # Tolerance for leave/rejoin when projecting (m)
    min_seg_len_m: float = 25.0       # Ignore tiny slivers


@dataclass
class DetourCandidate:
    """Stores metadata for each accepted detour."""
    leave_pt: Point
    rejoin_pt: Point
    branch_geom: LineString
    gain_poi: float
    extra_m: float
    ratio: float


# ---------------------------------------------------------
# Core helpers
# ---------------------------------------------------------

def _to_3857(g):
    if g.crs is None:
        g = g.set_crs(4326)
    return g.to_crs(3857)


def _project_measure(line_m: LineString, pt_m: Point) -> float:
    """Return the linear-referenced distance (m) of pt on line."""
    return float(line_m.project(pt_m))


def _poi_gain_for_segment(pois_m: gpd.GeoDataFrame, seg_m: LineString, radius_m: float) -> float:
    """Compute POI gain around a segment as mean of NormalizedScore of POIs within radius."""
    if pois_m is None or pois_m.empty:
        return 0.0
    buf = seg_m.buffer(radius_m)
    near = pois_m[pois_m.intersects(buf)]
    if near.empty:
        return 0.0
    scores = near.get("NormalizedScore")
    # If the attribute is missing, treat as zero
    if scores is None:
        return 0.0
    return float(np.mean(scores.fillna(0.0)))


# ---------------------------------------------------------
# Main detour logic
# ---------------------------------------------------------

def refine_with_intersection_branches(
    main_path: LineString,
    roads_file: str,
    pois_file: str,
    cfg: DetourConfig = DetourConfig(),
) -> Tuple[LineString, gpd.GeoDataFrame]:
    """
    Refine a route by adding realistic detours that:
      1) are actual road segments,
      2) touch/intersect the main path in two distinct places (leave & rejoin),
      3) rejoin further along the path (no looping backward),
      4) pass a POI gain vs. extra distance rule.

    Inputs
    ------
    main_path : LineString (WGS84)
    roads_file : path to fw_jps_roads.geojson
    pois_file : path to qc_pois_final_scored.geojson

    Returns
    -------
    refined_line : LineString (WGS84)
    debug_gdf    : GeoDataFrame of accepted detour segments (WGS84)
    """


    # Load and project
    roads = gpd.read_file(roads_file)
    pois = gpd.read_file(pois_file)
    roads_m = _to_3857(roads)
    pois_m = _to_3857(pois)
    main_m = _to_3857(gpd.GeoDataFrame(geometry=[main_path], crs="EPSG:4326")).geometry.iloc[0]

    # Fast reject tiny/invalid segments
    roads_m = roads_m[~roads_m.geometry.is_empty & roads_m.geometry.is_valid].copy()
    roads_m = roads_m[roads_m.geometry.length >= cfg.min_seg_len_m]

    # Spatial index for speed
    sidx = roads_m.sindex

    # Look only at segments that touch the main route buffer
    cand_idx = sidx.query(main_m.buffer(cfg.poi_radius_m), predicate="intersects")
    if len(cand_idx) == 0:
        print("⚪ No nearby road segments found for potential detours.")
        return main_path, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    detours: List[DetourCandidate] = []

    for i in cand_idx:
        seg = roads_m.geometry.iloc[int(i)]
        if seg.is_empty or seg.length < cfg.min_seg_len_m:
            continue

        # Intersection points between candidate segment and main route
        inter = seg.intersection(main_m)
        if inter.is_empty:
            continue

        # We only accept true LEAVE & REJOIN cases (two distinct touches)
        points = []
        if inter.geom_type == "Point":
            # Only one touch — not a valid leave + rejoin
            continue
        elif inter.geom_type == "MultiPoint":
            points = list(inter.geoms)
        elif inter.geom_type in ("LineString", "MultiLineString"):
            # Overlapping with main route — treat as already on the route; skip
            continue
        else:
            continue

        if len(points) < 2:
            continue

        # Choose two furthest touch points along the segment as leave/rejoin
        # (this guards against micro self-intersections)
        pts_m = points
        # Project both onto main route to know travel direction
        measures = [ _project_measure(main_m, p) for p in pts_m ]
        # Sort touches along the main path
        order = np.argsort(measures)
        leave_pt = pts_m[int(order[0])]
        rejoin_pt = pts_m[int(order[-1])]

        # Ensure forward progress along main route
        s_leave = _project_measure(main_m, leave_pt)
        s_rejoin = _project_measure(main_m, rejoin_pt)
        if s_rejoin <= s_leave + cfg.touch_tol_m:
            # Rejoin does not come later than leave — skip
            continue

        # Extra distance if we "detour" along the road segment instead of staying on main
        # Approximate: replace the main_path subsection [leave..rejoin] by this road segment.
        main_sub_len = float(main_m.segmentize(1.0).intersection(LineString([leave_pt, rejoin_pt])).length)
        # Fallback: if the above returns 0 (due to topology), approximate with chord distance
        if main_sub_len <= 0:
            main_sub_len = float(leave_pt.distance(rejoin_pt))

        extra = float(seg.length - main_sub_len)
        if extra < 0:
            extra = 0.0
        if extra > cfg.max_extra_m:
            continue

        # POI gain around this segment (1 km) using NormalizedScore
        poi_gain = _poi_gain_for_segment(pois_m, seg, cfg.poi_radius_m)
        denom_km = max(1e-6, extra / 1000.0)
        ratio = poi_gain / denom_km

        if ratio >= cfg.min_gain_ratio:
            print(f"🟢 Detour accepted: gain={poi_gain:.3f} | extra={extra:.1f} m | ratio={ratio:.2f}")
            detours.append(
                DetourCandidate(
                    leave_pt=leave_pt,
                    rejoin_pt=rejoin_pt,
                    branch_geom=seg,
                    gain_poi=poi_gain,
                    extra_m=extra,
                    ratio=ratio,
                )
            )

    # Merge selected detours with main path
    if len(detours) == 0:
        print("⚪ No qualifying detours — returning original route.")
        return main_path, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    merged = unary_union([main_m] + [d.branch_geom for d in detours])

    # Choose longest connected line (robust output)
    if merged.is_empty:
        refined_m = main_m
    elif merged.geom_type == "LineString":
        refined_m = merged
    elif merged.geom_type == "MultiLineString":
        refined_m = max(list(merged.geoms), key=lambda g: g.length)
    else:
        refined_m = main_m

    print(f"🧭 Refined length: {refined_m.length:.1f} m | Added detours: {len(detours)}")

    # Debug GDF (WGS84)
    debug_gdf = gpd.GeoDataFrame(
        [{
            "gain_poi": d.gain_poi,
            "extra_m": d.extra_m,
            "ratio": d.ratio,
            "geometry": d.branch_geom
        } for d in detours],
        crs=roads_m.crs
    ).to_crs(4326)

    return gpd.GeoSeries([refined_m], crs=roads_m.crs).to_crs(4326).iloc[0], debug_gdf


# ---------------------------------------------------------
# Export helper (robust and CRS-safe)
# ---------------------------------------------------------

def export_refined_route_with_debug(
    main_path_line: LineString,
    roads_file: str,
    pois_file: str,
    output_dir: str,
    cfg: DetourConfig = DetourConfig(),
):
    """
    High-level wrapper to run the detour logic and save outputs.
    Handles geometry merge robustness and CRS projection.
    """
    refined_line, debug_gdf = refine_with_intersection_branches(
        main_path_line, roads_file, pois_file, cfg
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refined_path = out_dir / "aco_route_refined.geojson"
    debug_path = out_dir / "aco_detours_debug.geojson"

    gpd.GeoSeries([refined_line], crs="EPSG:4326").to_file(refined_path, driver="GeoJSON")
    print(f"✅ Refined route saved → {refined_path}")

    if not debug_gdf.empty and not debug_gdf.geometry.is_empty.all():
        debug_gdf.to_file(debug_path, driver="GeoJSON")
        print(f"🧩 Debug detours saved → {debug_path}")
    else:
        print("⚪ No valid detour geometries to export (debug layer skipped).")

    return refined_path, debug_path
