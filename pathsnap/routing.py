"""Bounded shortest paths on the directed road graph (P3).

Distances are measured along directed arcs. A query from an observation projected onto
arc a to one on arc b includes the remaining fraction of a, the node-to-node shortest
path, and the traversed fraction of b. Dijkstra is bounded and cached by source node.
"""
from __future__ import annotations

from collections import OrderedDict
import heapq

import numpy as np

from .network import RoadNetwork


class Router:
    def __init__(self, net: RoadNetwork, cache_size: int = 256):
        self.net = net
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[int, float], tuple[dict[int, float], dict[int, tuple[int, int]]]] = OrderedDict()

    def _bounded_dijkstra(self, source: int, max_m: float):
        bound = max(0.0, float(max_m))
        key = (int(source), round(bound, 3))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        dist: dict[int, float] = {int(source): 0.0}
        # node -> (previous node, arc used to enter node)
        prev: dict[int, tuple[int, int]] = {}
        queue = [(0.0, int(source))]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != dist.get(node) or cost > bound:
                continue
            for aid in self.net.out_arcs(node):
                aid = int(aid)
                nxt = int(self.net.arcs[aid, 1])
                nd = cost + float(self.net.arc_len_m[aid])
                if nd <= bound and nd < dist.get(nxt, np.inf):
                    dist[nxt] = nd
                    prev[nxt] = (node, aid)
                    heapq.heappush(queue, (nd, nxt))
        value = (dist, prev)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _middle(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, max_m: float):
        a, b = int(arc_a), int(arc_b)
        fa, fb = float(np.clip(frac_a, 0.0, 1.0)), float(np.clip(frac_b, 0.0, 1.0))
        if a < 0 or b < 0 or a >= self.net.n_arcs or b >= self.net.n_arcs:
            return np.inf, []
        start_node = int(self.net.arcs[a, 1])
        goal_node = int(self.net.arcs[b, 0])
        prefix = (1.0 - fa) * float(self.net.arc_len_m[a])
        suffix = fb * float(self.net.arc_len_m[b])
        remaining = float(max_m) - prefix - suffix
        if remaining < 0:
            return np.inf, []
        if start_node == goal_node:
            return prefix + suffix, []
        distances, previous = self._bounded_dijkstra(start_node, remaining)
        middle_cost = distances.get(goal_node, np.inf)
        if not np.isfinite(middle_cost):
            return np.inf, []
        path = []
        node = goal_node
        while node != start_node:
            item = previous.get(node)
            if item is None:
                return np.inf, []
            parent, aid = item
            path.append(aid)
            node = parent
        path.reverse()
        return prefix + middle_cost + suffix, path

    def route_distance(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, max_m: float) -> float:
        """Shortest directed distance (m) between two fractional arc positions; inf if out of bound."""
        if int(arc_a) == int(arc_b) and float(frac_b) >= float(frac_a):
            distance = (float(frac_b) - float(frac_a)) * float(self.net.arc_len_m[int(arc_a)])
            return distance if distance <= max_m else np.inf
        distance, _ = self._middle(arc_a, frac_a, arc_b, frac_b, max_m)
        return float(distance)

    def shortest_arc_path(self, arc_a: int, arc_b: int, max_m: float) -> list[int]:
        """Intermediate directed arc ids from arc_a to arc_b (exclusive)."""
        a, b = int(arc_a), int(arc_b)
        if a == b:
            return []
        distance, path = self._middle(a, 0.0, b, 0.0, max_m)
        return path if np.isfinite(distance) else []
