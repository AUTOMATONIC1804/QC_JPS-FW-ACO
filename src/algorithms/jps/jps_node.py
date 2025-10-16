"""
src/algorithms/jps/jps_node.py
Node class used in Jump Point Search.
"""

class Node:
    def __init__(self, row, col, g=0, h=0, parent=None):
        self.row = row
        self.col = col
        self.g = g          # cost from start
        self.h = h          # heuristic to goal
        self.f = g + h      # total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"Node(r={self.row}, c={self.col}, g={self.g:.1f}, h={self.h:.1f})"
