"""
fw_matrix_inspect.py
-------------------------------------------------
Quick inspection tool for any Floyd–Warshall matrix (.npy files).

✅ Loads D.npy and FW.npy
✅ Prints shape, sample values, and stats
✅ Optionally prints full matrix for small networks (<20 nodes)
"""

import numpy as np
from pathlib import Path

# === CONFIG ===
# Change this path to your current FW output directory
fw_dir = Path(r"D:\Quezon_City\data\outputs\floyd_warshall")
prefix = "fw_dijkstra"   # or "fw_dijkstra", "fw_aco", etc.

# === Load files ===
D_path = fw_dir / f"{prefix}_D.npy"
FW_path = fw_dir / f"{prefix}_FW.npy"

print(f"📂 Loading matrices from: {fw_dir}")
if not D_path.exists() or not FW_path.exists():
    raise FileNotFoundError("❌ Missing D.npy or FW.npy in the specified directory.")

D = np.load(D_path)
FW = np.load(FW_path)

# === Basic info ===
print("\n🧮 Matrix Info:")
print(f"   D shape:  {D.shape}")
print(f"   FW shape: {FW.shape}")

if D.shape != FW.shape:
    print("⚠️ Shape mismatch — something may be wrong!")

# === Summary statistics ===
finite_vals = D[np.isfinite(D)]
fw_finite_vals = FW[np.isfinite(FW)]

print("\n📊 Distance Matrix (D) Stats:")
print(f"   Min: {finite_vals.min():.2f} m")
print(f"   Max: {finite_vals.max():.2f} m")
print(f"   Mean: {finite_vals.mean():.2f} m")

print("\n📊 Floyd–Warshall Matrix (FW) Stats:")
print(f"   Min: {fw_finite_vals.min():.2f} m")
print(f"   Max: {fw_finite_vals.max():.2f} m")
print(f"   Mean: {fw_finite_vals.mean():.2f} m")

# === Optional: print a slice of matrix ===
n = D.shape[0]
if n <= 20:
    print("\n🧩 Full Distance Matrix (D):")
    print(np.round(D, 1))
    print("\n🧩 Full FW Matrix (after APSP):")
    print(np.round(FW, 1))
else:
    print(f"\n⚙️ Matrix too large ({n}x{n}). Showing top-left 10x10 slice:")
    print("\nD (Haversine pairwise):")
    print(np.round(D[:10, :10], 1))
    print("\nFW (Floyd–Warshall result):")
    print(np.round(FW[:10, :10], 1))

# === Optional: Check for infinities or symmetry issues ===
n_inf = np.sum(~np.isfinite(D))
n_inf_fw = np.sum(~np.isfinite(FW))
print(f"\n♾️ Non-finite entries: D={n_inf}, FW={n_inf_fw}")

if not np.allclose(D, D.T, equal_nan=True):
    print("⚠️ D matrix is not symmetric.")
else:
    print("✅ D matrix is symmetric.")

if not np.allclose(FW, FW.T, equal_nan=True):
    print("⚠️ FW matrix is not symmetric.")
else:
    print("✅ FW matrix is symmetric.")

print("\n✅ Inspection complete.")
