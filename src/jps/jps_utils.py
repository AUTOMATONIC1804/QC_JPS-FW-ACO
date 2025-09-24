# src/jps/jps_utils.py

# Movement directions: (row, col) deltas
DIRECTIONS = [
    (-1, 0),   # up
    (1, 0),    # down
    (0, -1),   # left
    (0, 1),    # right
    (-1, -1),  # up-left
    (-1, 1),   # up-right
    (1, -1),   # down-left
    (1, 1),    # down-right
]

def walkable(grid, node):
    """Check if a node is inside the grid and not an obstacle."""
    x, y = node
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x][y] == 0

def get_neighbors(node, grid, parent=None):
    """
    Get neighbors for a node according to JPS pruning rules.
    If no parent, return all directions (for start node).
    Otherwise, return only natural + forced neighbors.
    """
    if parent is None:
        return DIRECTIONS

    px, py = parent
    x, y = node
    dx = (x - px, y - py)

    neighbors = []

    # Straight moves
    if dx[0] == 0 or dx[1] == 0:
        if dx[0] != 0:  # vertical
            for d in [(dx[0], 0), (dx[0], -1), (dx[0], 1)]:
                neighbors.append(d)
        else:  # horizontal
            for d in [(0, dx[1]), (-1, dx[1]), (1, dx[1])]:
                neighbors.append(d)

    # Diagonal moves
    else:
        neighbors.extend([
            (dx[0], 0),   # horizontal
            (0, dx[1]),   # vertical
            dx            # diagonal itself
        ])

    return neighbors

def jump(node, direction, grid, goal, debug=False):
    """
    Recursive jump along a direction.
    Returns the next jump point or None if blocked.
    """
    x, y = node
    dx, dy = direction
    nx, ny = x + dx, y + dy

    if not walkable(grid, (nx, ny)):
        return None
    if (nx, ny) == goal:
        if debug:
            print(f"🎯 Found goal while jumping to {(nx, ny)}")
        return (nx, ny)

    # Forced neighbor check
    forced = False
    if dx != 0 and dy != 0:  # diagonal
        if (walkable(grid, (nx - dx, ny)) and not walkable(grid, (nx - dx, y))) \
           or (walkable(grid, (nx, ny - dy)) and not walkable(grid, (x, ny - dy))):
            forced = True
    else:  # straight
        if dx != 0:  # vertical
            if (walkable(grid, (nx, ny + 1)) and not walkable(grid, (x, ny + 1))) \
               or (walkable(grid, (nx, ny - 1)) and not walkable(grid, (x, ny - 1))):
                forced = True
        elif dy != 0:  # horizontal
            if (walkable(grid, (nx + 1, ny)) and not walkable(grid, (nx + 1, y))) \
               or (walkable(grid, (nx - 1, ny)) and not walkable(grid, (nx - 1, y))):
                forced = True

    if forced:
        if debug:
            print(f"⚡ Forced neighbor found at {(nx, ny)}")
        return (nx, ny)

    # Recursive continue
    if dx != 0 and dy != 0:
        if jump((nx, ny), (dx, 0), grid, goal, debug) or jump((nx, ny), (0, dy), grid, goal, debug):
            if debug:
                print(f"↩️ Diagonal recursion stop at {(nx, ny)}")
            return (nx, ny)

    return jump((nx, ny), (dx, dy), grid, goal, debug)
