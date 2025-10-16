"""
src/algorithms/jps/jps_grid.py
Grid wrapper for JPS.
"""

import numpy as np


class Grid:
    def __init__(self, matrix):
        """
        matrix: 2D numpy array where 1=road (passable), 0=obstacle.
        """
        self.matrix = matrix
        self.height, self.width = matrix.shape

    def in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def passable(self, row, col):
        return self.in_bounds(row, col) and self.matrix[row, col] == 1

    def neighbors(self, row, col):
        """
        Return valid 8-way neighbors of (row, col).
        Prevents corner cutting through obstacles.
        """
        results = []
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # N, S, W, E
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # NW, NE, SW, SE
        ]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if not self.passable(r, c):
                continue

            # For diagonal moves, check corner cutting
            if abs(dr) + abs(dc) == 2:
                if not (self.passable(row + dr, col) and self.passable(row, col + dc)):
                    continue
            results.append((r, c))

        return results
