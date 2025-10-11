"""
src/run_pathfinding_comparison.py
Compare JPS (grid-based) vs Dijkstra and A* (graph-based).
"""

import json
from src.algorithms.jps_runner import run_jps_benchmark
from src.algorithms.dijkstra_runner import run_dijkstra_benchmark
from src.algorithms.astar_runner import run_astar_benchmark


def main():
    print("=== 🚆 PATHFINDING COMPARISON START ===")

    results = []
    results.append(run_jps_benchmark())
    results.append(run_dijkstra_benchmark())
    results.append(run_astar_benchmark())

    print("\n=== ✅ COMPARISON COMPLETE ===")
    for r in results:
        if r["path_length_m"] is None:
            print(f"{r['algorithm']}: ❌ No path found.")
        else:
            print(f"{r['algorithm']}: {r['runtime_ms']:.2f} ms | {r['path_length_m']:.2f} m | {r['steps']} steps")

    with open("data/outputs/pathfinding_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Saved results → data/outputs/pathfinding_comparison.json")


if __name__ == "__main__":
    main()

