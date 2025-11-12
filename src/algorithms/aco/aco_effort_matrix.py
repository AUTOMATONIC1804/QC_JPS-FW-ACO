# src/algorithms/aco/aco_effort_matrix.py
# -*- coding: utf-8 -*-

"""
Effort matrix builder for ACO station selection
-----------------------------------------------

Adapts Korzeń & Kruszyna (2025) "effort"-based ACO to our corridor problem.
Population potential is omitted. Attractiveness is derived from POI density/scores
within a 1 km buffer around each candidate node.

Includes:
✅ Deduplication of POIs by @id or id (polygon/point merged)
✅ Weighted score by category (from poi_weights)
✅ Proper CRS handling (EPSG:4326 → EPSG:3857)
✅ Normalized per-node score
✅ Top-POI name extraction per node (for ACO output inspection)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import geopandas as gpd


# ---- configuration -----------------------------------------------------------

@dataclass
class EffortParams:
    method: str  # "jps" | "dijkstra" | "astar"
    processed_dir: str = "data/processed"
    buffer_radius_m: float = 1000.0
    expected_speed_kmh: float = 35.0  # V_o
    poi_score_col: str = "score"
    category_col: str = "category"
    base_score_col: str = "base_score"

    def path(self, stem: str) -> str:
        return os.path.join(self.processed_dir, f"fw_{self.method}_{stem}")


# ---- core API ----------------------------------------------------------------

def compute_effort_matrix(
    params: EffortParams,
    pois_gdf: gpd.GeoDataFrame,
    poi_weights: Optional[Dict[str, float]] = None,
    detour_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Build and save the effort matrix for the selected method."""
    points_gdf, D, FW = _load_method_inputs(params)
    points_gdf, pois_gdf = _ensure_metric_crs(points_gdf, pois_gdf, "EPSG:3857")

    node_stats = _compute_node_poi_stats(
        points_gdf=points_gdf,
        pois_gdf=pois_gdf,
        buffer_radius_m=params.buffer_radius_m,
        poi_score_col=params.poi_score_col,
        category_col=params.category_col,
        base_score_col=params.base_score_col,
        poi_weights=poi_weights or {},
    )

    node_scores_norm = {int(k): v["score_norm"] for k, v in node_stats.items()}
    A = _build_attractiveness_matrix(len(points_gdf), node_scores_norm)

    L = D if D is not None else FW
    if L is None:
        raise RuntimeError("No distance matrix found (D or FW).")

    with np.errstate(invalid="ignore", divide="ignore"):
        L_c = np.nanmax(L)
    if not np.isfinite(L_c) or L_c <= 0:
        L_norm = np.zeros_like(L, dtype=float)
    else:
        L_norm = np.divide(L, L_c, out=np.zeros_like(L, dtype=float))
    V_ratio = _speed_ratio_matrix(L, FW, params.expected_speed_kmh)
    E = L_norm * V_ratio * A

    if detour_mask is not None:
        if detour_mask.shape != E.shape:
            raise ValueError("detour_mask has different shape than effort matrix.")
        penalty = np.where(detour_mask, 1.0, 10.0)
        E = E * penalty

    np.save(params.path("effort.npy"), E)
    with open(params.path("poi_stats.json"), "w", encoding="utf-8") as f:
        json.dump(node_stats, f, indent=2)

    return E, node_stats


# ---- PATH HANDLER ------------------------------------------------------------

def _load_method_inputs(params: EffortParams) -> Tuple[gpd.GeoDataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load points and matrices for the selected method, future-proofed for multiple variants."""
    method = params.method.lower()

    base_dir = "data/outputs/floyd_warshall"
    if method == "jps":
        points_fp = os.path.join(base_dir, "fw_jps_points.geojson")
        D_fp = os.path.join(base_dir, "fw_jps_D.npy")
        FW_fp = os.path.join(base_dir, "fw_jps_FW.npy")
    elif method in ["dijkstra", "dj"]:
        points_fp = os.path.join(base_dir, "fw_dijkstra_points.geojson")
        D_fp = os.path.join(base_dir, "fw_dijkstra_D.npy")
        FW_fp = os.path.join(base_dir, "fw_dijkstra_FW.npy")
    elif method in ["astar", "a_star", "a*"]:
        points_fp = os.path.join(base_dir, "fw_astar_points.geojson")
        D_fp = os.path.join(base_dir, "fw_astar_D.npy")
        FW_fp = os.path.join(base_dir, "fw_astar_FW.npy")
    else:
        points_fp = params.path("points.geojson")
        D_fp = params.path("D.npy")
        FW_fp = params.path("FW.npy")

    if not os.path.exists(points_fp):
        raise FileNotFoundError(points_fp)

    points_gdf = gpd.read_file(points_fp)
    D = np.load(D_fp) if os.path.exists(D_fp) else None
    FW = np.load(FW_fp) if os.path.exists(FW_fp) else None
    return points_gdf, D, FW


# ---- CRS + HELPER ------------------------------------------------------------

def _ensure_metric_crs(points_gdf, pois_gdf, metric_crs: str):
    """Project to metric CRS if necessary."""
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs("EPSG:4326", allow_override=True)
    if pois_gdf.crs is None:
        pois_gdf = pois_gdf.set_crs(points_gdf.crs, allow_override=True)
    if not _is_metric_crs(points_gdf.crs):
        points_gdf = points_gdf.to_crs(metric_crs)
    if pois_gdf.crs != points_gdf.crs:
        pois_gdf = pois_gdf.to_crs(points_gdf.crs)
    return points_gdf, pois_gdf


def _is_metric_crs(crs) -> bool:
    try:
        return "unit=m" in crs.to_string().lower() or crs.is_projected
    except Exception:
        return False


# ---- UPDATED POI SCORING (DEDUP + BUFFER INTERSECTS + TOP POIS) --------------

def _compute_node_poi_stats(points_gdf, pois_gdf, buffer_radius_m, poi_score_col, category_col, base_score_col, poi_weights):
    """Compute per-node POI stats (dedup by ID, count + weighted score, normalized, with top POI names)."""
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs("EPSG:4326")
    if pois_gdf.crs is None:
        pois_gdf = pois_gdf.set_crs("EPSG:4326")

    points_m = points_gdf.to_crs("EPSG:3857")
    pois_m = pois_gdf.to_crs("EPSG:3857")

    # --- Deduplicate by @id or id (prefer polygons)
    id_field = "@id" if "@id" in pois_m.columns else ("id" if "id" in pois_m.columns else None)
    if id_field:
        pois_m["_poly_priority"] = pois_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        pois_m = (
            pois_m.sort_values(by=[id_field, "_poly_priority"], ascending=[True, False])
                  .drop_duplicates(subset=[id_field], keep="first")
                  .drop(columns=["_poly_priority"])
                  .reset_index(drop=True)
        )

    # --- Buffer-based spatial join (intersects)
    buffers = points_m.copy()
    buffers["__buf_geom__"] = buffers.geometry.buffer(buffer_radius_m)
    right = buffers.set_geometry("__buf_geom__")[["__buf_geom__"]].rename_geometry("geometry")
    right["__node_idx__"] = np.arange(len(right))

    joined = gpd.sjoin(pois_m, right, how="inner", predicate="intersects")

    # --- Compute per-POI score
    if poi_score_col in joined.columns:
        joined["__poi_score__"] = joined[poi_score_col].astype(float)
    else:
        w = joined[category_col].map(poi_weights).fillna(1.0)
        joined["__poi_score__"] = w * joined.get(base_score_col, 1.0)

    # --- Deduplicate within each node by ID again (for safety)
    if id_field and id_field in joined.columns:
        joined = joined.sort_values(by=["__node_idx__", id_field])
        joined = joined.drop_duplicates(subset=["__node_idx__", id_field], keep="first")

    # --- Group by node
    grp = joined.groupby("__node_idx__")
    counts = grp.size().to_dict()
    scores = grp["__poi_score__"].sum().to_dict()
    max_score = max(scores.values()) if scores else 0.0

    # --- Extract top POI names per node
    node_stats = {}
    n_nodes = len(points_gdf)
    name_cols = [col for col in ["name", "name_left", "name_right"] if col in joined.columns]

    for idx in range(n_nodes):
        c = float(counts.get(idx, 0))
        s = float(scores.get(idx, 0.0))
        sn = (s / max_score) if max_score > 0 else 0.0

        near = joined[joined["__node_idx__"] == idx]
        if len(near) > 0:
            if name_cols:
                near["_poi_name_"] = near[name_cols].bfill(axis=1).iloc[:, 0]
            else:
                near["_poi_name_"] = "Unnamed"

            top_pois = (
                near.sort_values("__poi_score__", ascending=False)
                    .head(5)["_poi_name_"].fillna("Unnamed").tolist()
            )
        else:
            top_pois = []

        node_stats[str(idx)] = {
            "count": c,
            "score": s,
            "score_norm": sn,
            "top_pois": top_pois,
        }

    return node_stats


# ---- MATRICES ---------------------------------------------------------------

def _build_attractiveness_matrix(n, node_scores_norm):
    """A_ij = 1 - mean(norm_poi_i, norm_poi_j)."""
    s = np.zeros(n, dtype=float)
    for i in range(n):
        s[i] = float(node_scores_norm.get(i, 0.0))
    M = (s[:, None] + s[None, :]) * 0.5
    A = 1.0 - M
    np.fill_diagonal(A, 1.0 - s)
    return A


def _speed_ratio_matrix(L, FW, expected_speed_kmh):
    """Compute V_o / V_sr (km/h)."""
    if FW is None:
        return np.ones_like(L, dtype=float)
    fw_median = np.nanmedian(FW)
    l_median = np.nanmedian(L)
    time_like = fw_median > 0 and fw_median < 1e5 and l_median > 0
    if not time_like:
        return np.ones_like(L, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        v_m_per_s = np.divide(L, FW, out=np.zeros_like(L, dtype=float), where=(FW > 0))
        v_kmh = v_m_per_s * 3.6
        ratio = np.divide(expected_speed_kmh, v_kmh, out=np.ones_like(L, dtype=float), where=(v_kmh > 0))
    return np.clip(ratio, 0.25, 4.0)
