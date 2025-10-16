"""
src/algorithms/astar/astar_utils.py
Utility functions for A*:
- snap_to_nearest_road: repositions start/goal to nearest valid cell.
"""

from collections import deque


def snap_to_nearest_road(grid, start_cell, max_radius=50):
    """If start_cell is obstacle, find nearest road (value=1) via BFS."""
    r0, c0 = start_cell
    if grid.matrix[r0, c0] == 1:
        return start_cell

    rows, cols = grid.matrix.shape
    visited = set()
    q = deque([(r0, c0, 0)])

    while q:
        r, c, d = q.popleft()
        if d > max_radius:
            break
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if 0 <= r < rows and 0 <= c < cols and grid.matrix[r, c] == 1:
            return (r, c)

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                q.append((rr, cc, d + 1))

    raise ValueError(f"No nearby road within {max_radius} cells")
