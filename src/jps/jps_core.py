# src/jps/jps_core.py

import heapq
from .jps_utils import get_neighbors, jump  # ✅ relative import for -m usage

class JPS:
    def __init__(self, grid, start, goal, debug=False):
        """
        Jump Point Search implementation.
        :param grid: 2D numpy array (0=free, 1=obstacle)
        :param start: tuple (row, col)
        :param goal: tuple (row, col)
        :param debug: print debug logs if True
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.open_list = []
        self.closed = set()
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        self.debug = debug

    def heuristic(self, a, b):
        """Octile distance heuristic (grid-based)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (2**0.5 - 2) * min(dx, dy)

    def reconstruct_path(self, current):
        """Rebuild path by following parent links backwards."""
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]

    def search(self):
        """Run the JPS algorithm and return the path (list of (row,col))."""
        heapq.heappush(self.open_list, (self.f_score[self.start], self.start))

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            if self.debug:
                print(f"🔎 Expanding: {current}")

            if current == self.goal:
                if self.debug:
                    print("🎯 Goal reached!")
                return self.reconstruct_path(current)

            self.closed.add(current)

            for direction in get_neighbors(current, self.grid, self.came_from.get(current)):
                nxt = jump(current, direction, self.grid, self.goal, self.debug)
                if nxt is None:
                    continue

                tentative_g = self.g_score[current] + self.heuristic(current, nxt)

                if nxt in self.closed and tentative_g >= self.g_score.get(nxt, float("inf")):
                    continue

                if tentative_g < self.g_score.get(nxt, float("inf")):
                    self.came_from[nxt] = current
                    self.g_score[nxt] = tentative_g
                    self.f_score[nxt] = tentative_g + self.heuristic(nxt, self.goal)
                    heapq.heappush(self.open_list, (self.f_score[nxt], nxt))

                    if self.debug:
                        print(f"➡️  Jumped to {nxt}, g={tentative_g:.2f}, f={self.f_score[nxt]:.2f}")

        if self.debug:
            print("❌ No path found")
        return None
