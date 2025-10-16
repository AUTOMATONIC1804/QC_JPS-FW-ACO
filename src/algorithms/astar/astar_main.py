"""
src/algorithms/astar/astar_main.py
Core A* implementation for grid-based pathfinding.
Uses 8-directional movement and octile distance heuristic.
"""

import heapq
from math import sqrt
from .astar_heuristics import octile_distance


def astar_search(grid, start, goal):
    """Perform A* search on grid with 8-directional movement."""
    rows, cols = grid.matrix.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: octile_distance(start, goal)}

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid.matrix[neighbor[0], neighbor[1]] == 0:
                    continue  # obstacle

                step_cost = sqrt(2) if dr != 0 and dc != 0 else 1
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + octile_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found
