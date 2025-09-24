# qc_load_benchmarks.py
import json
import os

# Define the path to the benchmark points JSON
benchmark_file = r"D:\Quezon_City\data\inputs\benchmark_points.json"

def load_benchmark_points():
    """
    Load benchmark coordinates from JSON file with error handling.

    Returns:
        dict: {landmark_name: {"lat": float, "lon": float}}
    """
    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f"❌ Benchmark file not found: {benchmark_file}")

    if os.path.getsize(benchmark_file) == 0:
        raise ValueError(f"❌ Benchmark file is empty: {benchmark_file}")

    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            points = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format in {benchmark_file}: {e}")

    return points

# For quick testing
if __name__ == "__main__":
    try:
        benchmarks = load_benchmark_points()
        print("✅ Loaded benchmark points:")
        for name, coords in benchmarks.items():
            print(f"  {name}: {coords['lat']}, {coords['lon']}")
    except Exception as e:
        print(str(e))
