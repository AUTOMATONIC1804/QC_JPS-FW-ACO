"""
src/jps/jps_grid.py
Handles grid-based map for JPS.
"""

import numpy as np

class Grid:
    def __init__(self, matrix):
        """
        matrix: 2D numpy array (1=passable, 0=obstacle)
        """
        self.matrix = matrix
        self.rows, self.cols = matrix.shape

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, r, c):
        return self.in_bounds(r, c) and self.matrix[r, c] == 1

    def neighbors(self, r, c):
        """Return all possible 8-direction moves."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(r+dr, c+dc) for dr, dc in dirs if self.passable(r+dr, c+dc)]
