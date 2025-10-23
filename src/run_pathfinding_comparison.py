"""
src/run_pathfinding_comparison.py
Compare performance of JPS (grid), Dijkstra (road graph), and A* (grid).
"""

import json
from src.algorithms.jps_runner import run_jps_benchmark
from src.algorithms.dijkstra_runner import run_dijkstra_benchmark
from src.algorithms.astar_runner import run_astar_benchmark


def main():
    print("=== 🚆 PATHFINDING COMPARISON START ===")

    results = []

    # --- Run JPS ---
    print("\n=== 🟨 Running Jump Point Search (Grid) ===")
    jps_metrics = run_jps_benchmark()
    results.append(jps_metrics)

    # --- Run Dijkstra ---
    print("\n=== 🔵 Running Dijkstra (Road Graph) ===")
    dijkstra_metrics = run_dijkstra_benchmark()
    results.append(dijkstra_metrics)

    # --- Run A* ---
    print("\n=== 🟩 Running A* (Grid) ===")
    astar_metrics = run_astar_benchmark()
    results.append(astar_metrics)

    # --- Summary ---
    print("\n=== ✅ COMPARISON COMPLETE ===")
    for r in results:
        if r is None:
            print("❌ One algorithm failed to return results.")
            continue
        if r.get("path_length_m") is None:
            print(f"{r.get('algorithm', 'Unknown')}: ❌ No path found.")
        else:
            print(f"{r['algorithm']}: {r['runtime_ms']:.2f} ms | "
                f"{r['path_length_m']:.2f} m | {r['steps']} steps")

    # --- Save results ---
    with open("data/outputs/pathfinding_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Saved results → data/outputs/pathfinding_comparison.json")


if __name__ == "__main__":
    main()
