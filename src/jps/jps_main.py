"""
src/jps/jps_main.py
Jump Point Search algorithm (8-way).
Based on Harabor & Grastien (2012, 2014).
"""

import heapq
from src.jps.jps_node import Node


# 8 directions
DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]


def jump(grid, row, col, dr, dc, goal):
    """
    Perform recursive jump in direction (dr, dc).
    """
    r, c = row + dr, col + dc
    if not grid.passable(r, c):
        return None
    if (r, c) == goal:
        return (r, c)

    # Forced neighbor checks
    if dr != 0 and dc != 0:
        # Diagonal
        if (grid.passable(r - dr, c + dc) and not grid.passable(r - dr, c)) or \
           (grid.passable(r + dr, c - dc) and not grid.passable(r, c - dc)):
            return (r, c)
    else:
        # Horizontal / Vertical
        if dr != 0:
            if (grid.passable(r + dr, c + 1) and not grid.passable(r, c + 1)) or \
               (grid.passable(r + dr, c - 1) and not grid.passable(r, c - 1)):
                return (r, c)
        else:
            if (grid.passable(r + 1, c + dc) and not grid.passable(r + 1, c)) or \
               (grid.passable(r - 1, c + dc) and not grid.passable(r - 1, c)):
                return (r, c)

    # Diagonal recursion
    if dr != 0 and dc != 0:
        if jump(grid, r, c, dr, 0, goal) or jump(grid, r, c, 0, dc, goal):
            return (r, c)

    return jump(grid, r, c, dr, dc, goal)


def get_successors(grid, node, goal, heuristic):
    successors = []
    for dr, dc in DIRECTIONS:
        jp = jump(grid, node.row, node.col, dr, dc, goal)
        if jp:
            r, c = jp
            g_cost = node.g + ((dr * dc) and 1.414 or 1.0)
            h_cost = heuristic((r, c), goal)
            successors.append(Node(r, c, g_cost, h_cost, node))
    return successors


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.row, node.col))
        node = node.parent
    return path[::-1]


def jump_point_search(grid, start, goal, heuristic):
    """
    Run Jump Point Search.
    start, goal: (row, col)
    """
    open_list = []
    start_node = Node(start[0], start[1], 0, heuristic(start, goal))
    heapq.heappush(open_list, (start_node.f, start_node))
    closed_set = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if (current.row, current.col) == goal:
            return reconstruct_path(current)

        closed_set.add((current.row, current.col))

        for successor in get_successors(grid, current, goal, heuristic):
            if (successor.row, successor.col) in closed_set:
                continue
            heapq.heappush(open_list, (successor.f, successor))

    return None
