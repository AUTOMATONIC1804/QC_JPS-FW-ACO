"""
src/run_pathfinding_comparison.py
Compare performance of different pathfinding algorithms (JPS, Dijkstra, A*).
"""

import json
from src.algorithms.jps_runner import run_jps_benchmark


def main():
    print("=== 🚆 PATHFINDING COMPARISON START ===")

    results = []
    jps_metrics = run_jps_benchmark()
    results.append(jps_metrics)

    # Placeholder — soon:
    # dijkstra_metrics = run_dijkstra_benchmark()
    # astar_metrics = run_astar_benchmark()
    # results.extend([dijkstra_metrics, astar_metrics])

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
