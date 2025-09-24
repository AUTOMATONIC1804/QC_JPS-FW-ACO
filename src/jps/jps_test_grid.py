# src/jps/jps_test_grid.py

import numpy as np
import matplotlib.pyplot as plt
from src.jps.jps_core import JPS  # ✅ absolute import

def make_grid():
    # 0 = free, 1 = obstacle
    grid = np.zeros((10, 10), dtype=int)

    # Add some obstacles
    grid[4, 2:8] = 1  # horizontal wall
    grid[6, 3:9] = 1  # another wall
    return grid

def plot_grid(grid, path=None, start=None, goal=None):
    plt.imshow(grid, cmap="Greys", origin="upper")

    if path:
        y, x = zip(*path)  # unpack path
        plt.plot(x, y, color="red", linewidth=2, marker="o")

    if start:
        plt.scatter(start[1], start[0], c="green", s=100, label="Start")
    if goal:
        plt.scatter(goal[1], goal[0], c="blue", s=100, label="Goal")

    plt.legend()
    plt.title("JPS Pathfinding (Test Grid)")
    plt.show()

if __name__ == "__main__":
    grid = make_grid()

    start = (0, 0)
    goal = (9, 9)

    jps = JPS(grid, start, goal, debug=True)
    path = jps.search()

    if path:
        print("✅ Path found:", path)
        print(f"Path length: {len(path)}")
        plot_grid(grid, path, start, goal)
    else:
        print("❌ No path found")
        plot_grid(grid, None, start, goal)
