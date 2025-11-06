"""
check_fw_integrity.py
--------------------------------
Quick integrity check for Floyd–Warshall + POI data
before running ACO station optimization.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path

# === Paths ===
POINTS_FILE = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_points.geojson")
DIST_FILE   = Path(r"D:\Quezon_City\data\outputs\floyd_warshall\fw_jps_D.npy")
POI_FILE    = Path(r"D:\Quezon_City\data\processed\qc_pois_final_scored.geojson")

print("\n🔍 Checking data integrity for ACO inputs...\n")

# ---------------------------------------------------
# 1️⃣ Check Points File
# ---------------------------------------------------
print("📍 Checking FW points file...")
try:
    nodes = gpd.read_file(POINTS_FILE)
    print(f"  → Loaded {len(nodes)} nodes")
    print(f"  → CRS: {nodes.crs}")
    print(f"  → Example coordinates:\n{nodes.head(3)}\n")

    if nodes.geometry.is_empty.any():
        print("  ⚠️ Empty geometries found!")
    if nodes.duplicated(subset="geometry").any():
        print("  ⚠️ Duplicate node coordinates detected.")
    if nodes.geometry.x.isna().any() or nodes.geometry.y.isna().any():
        print("  ⚠️ Found NaN coordinates!")
except Exception as e:
    print(f"  ❌ Failed to read points file: {e}")


# ---------------------------------------------------
# 2️⃣ Check Distance Matrix
# ---------------------------------------------------
print("\n📏 Checking FW distance matrix...")
try:
    D = np.load(DIST_FILE)
    print(f"  → Matrix shape: {D.shape}")
    print(f"  → Finite entries: {(np.isfinite(D)).sum()} / {D.size}")

    if D.shape[0] != D.shape[1]:
        print("  ❌ Matrix not square!")
    if np.any(D < 0):
        print("  ⚠️ Negative distances detected!")
    if np.any(np.isnan(D)):
        print("  ⚠️ NaN values present in distance matrix.")
    if not np.allclose(D, D.T, atol=1e-6):
        print("  ⚠️ Matrix is not symmetric!")

    diag_zero = np.allclose(np.diag(D), 0)
    print(f"  → Diagonal all zeros: {diag_zero}")
except Exception as e:
    print(f"  ❌ Failed to load distance matrix: {e}")


# ---------------------------------------------------
# 3️⃣ Check POI File
# ---------------------------------------------------
print("\n🏙️ Checking POI scoring file...")
try:
    pois = gpd.read_file(POI_FILE)
    cols = [c for c in pois.columns if "Score" in c or "Category" in c or "Classified" in c]
    print(f"  → Found columns: {cols}")
    print(f"  → {len(pois)} POIs loaded")

    if "NormalizedScore" in pois.columns:
        min_s, max_s = pois["NormalizedScore"].min(), pois["NormalizedScore"].max()
        print(f"  → NormalizedScore range: {min_s:.3f} – {max_s:.3f}")
        if min_s < 0 or max_s > 1.05:
            print("  ⚠️ Normalized scores outside expected range (0–1).")
    else:
        print("  ⚠️ No NormalizedScore column found!")

    if pois["Category"].isna().any():
        print("  ⚠️ Missing Category values.")
except Exception as e:
    print(f"  ❌ Failed to read POI file: {e}")

print("\n✅ Integrity check complete.\n")
