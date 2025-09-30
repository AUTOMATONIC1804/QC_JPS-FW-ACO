# src/utils/coords.py
from shapely.geometry import LineString

def grid_to_world(cell, meta):
    """
    Convert grid indices -> world coordinates.
    Args:
        cell: (row, col)
        meta: dict from graph_to_grid
    Returns:
        (x, y) in world coords
    """
    row, col = cell
    xmin, ymax = meta["origin"]
    res = meta["resolution"]
    x = xmin + (col + 0.5) * res
    y = ymax - (row + 0.5) * res
    return x, y

def path_to_linestring(path_cells, meta):
    """
    Convert a JPS path (list of cells) to a LineString.
    """
    coords = [grid_to_world(cell, meta) for cell in path_cells]
    return LineString(coords)
