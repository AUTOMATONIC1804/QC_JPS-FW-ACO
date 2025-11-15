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
    
    # Get coordinates once for all algorithms
    print("\nEnter coordinates (lat, lon format)")
    try:
        start_input = input("Start coordinates (lat, lon): ").strip()
        if not start_input:
            start_coords = (14.7327857, 121.0611778)
        else:
            start_parts = [x.strip() for x in start_input.split(",")]
            if len(start_parts) != 2:
                raise ValueError("Start coordinates must be in format: lat, lon")
            start_coords = (float(start_parts[0]), float(start_parts[1]))
        
        goal_input = input("Goal coordinates (lat, lon): ").strip()
        if not goal_input:
            goal_coords = (14.656511, 121.031089)
        else:
            goal_parts = [x.strip() for x in goal_input.split(",")]
            if len(goal_parts) != 2:
                raise ValueError("Goal coordinates must be in format: lat, lon")
            goal_coords = (float(goal_parts[0]), float(goal_parts[1]))
        
        print(f"✅ Using coordinates: Start ({start_coords[0]}, {start_coords[1]}), Goal ({goal_coords[0]}, {goal_coords[1]})\n")
    except (ValueError, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print("⚠️ Invalid input, using default coordinates.")
        start_coords = (14.7327857, 121.0611778)
        goal_coords = (14.656511, 121.031089)

    results = []

    # --- Run JPS ---
    print("\n=== 🟨 Running Jump Point Search (Grid) ===")
    jps_metrics = run_jps_benchmark(start_coords=start_coords, goal_coords=goal_coords)
    results.append(jps_metrics)

    # --- Run Dijkstra ---
    print("\n=== 🔵 Running Dijkstra (Road Graph) ===")
    dijkstra_metrics = run_dijkstra_benchmark(start_coords=start_coords, goal_coords=goal_coords)
    results.append(dijkstra_metrics)

    # --- Run A* ---
    print("\n=== 🟩 Running A* (Grid) ===")
    astar_metrics = run_astar_benchmark(start_coords=start_coords, goal_coords=goal_coords)
    results.append(astar_metrics)

    # --- Summary ---
    print("\nCOMPARISON COMPLETE")
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
