"""
src/jps/jps_main.py
Jump Point Search (JPS) implementation.
Based on Harabor & Grastien (2012, 2014) and helizac repo.
"""

import heapq
from src.jps.jps_node import Node
from src.jps.jps_heuristics import octile


def jump_point_search(grid, start, goal, heuristic=octile):
    """
    JPS main entrypoint.
    grid: Grid object
    start, goal: (row, col)
    Returns: list of (row, col) path cells or None
    """
    open_set = []
    closed = set()

    start_node = Node(*start, g=0, h=heuristic(start, goal))
    heapq.heappush(open_set, (start_node.f, start_node))

    while open_set:
        _, current = heapq.heappop(open_set)

        if (current.row, current.col) == goal:
            return reconstruct_path(current)

        closed.add((current.row, current.col))

        for neighbor in identify_successors(grid, current, goal, heuristic):
            if (neighbor.row, neighbor.col) not in closed:
                heapq.heappush(open_set, (neighbor.f, neighbor))

    return None


def identify_successors(grid, node, goal, heuristic):
    successors = []
    for nx, ny in prune_neighbors(grid, node):
        jp = jump(grid, nx, ny, node.row, node.col, goal)
        if jp:
            jx, jy, g_cost = jp
            new_node = Node(jx, jy,
                            g=node.g + g_cost,
                            h=heuristic((jx, jy), goal),
                            parent=node)
            successors.append(new_node)
    return successors


def prune_neighbors(grid, node):
    """
    Neighbor pruning step (Harabor & Grastien, 2012).
    """
    x, y = node.row, node.col
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(x+dx, y+dy) for dx, dy in dirs if grid.passable(x+dx, y+dy)]


def jump(grid, x, y, px, py, goal):
    """
    Jump step: recursively moves in direction until forced neighbor or goal.
    """
    dx = x - px
    dy = y - py

    if not grid.passable(x, y):
        return None

    if (x, y) == goal:
        return (x, y, 1)

    # Forced neighbor checks
    if dx != 0 and dy != 0:  # diagonal
        if (grid.passable(x-dx, y+dy) and not grid.passable(x-dx, y)) or \
           (grid.passable(x+dx, y-dy) and not grid.passable(x, y-dy)):
            return (x, y, 1)
    else:  # straight
        if dx != 0:
            if (grid.passable(x+dx, y+1) and not grid.passable(x, y+1)) or \
               (grid.passable(x+dx, y-1) and not grid.passable(x, y-1)):
                return (x, y, 1)
        else:
            if (grid.passable(x+1, y+dy) and not grid.passable(x+1, y)) or \
               (grid.passable(x-1, y+dy) and not grid.passable(x-1, y)):
                return (x, y, 1)

    # Recurse further
    return jump(grid, x+dx, y+dy, x, y, goal)


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.row, node.col))
        node = node.parent
    return path[::-1]
