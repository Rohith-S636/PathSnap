"""Bounded shortest paths on the directed road graph (P3)."""
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
        dist, prev = {int(source): 0.0}, {}
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
                    dist[nxt], prev[nxt] = nd, (node, aid)
                    heapq.heappush(queue, (nd, nxt))
        value = (dist, prev)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    @staticmethod
    def _reconstruct(source: int, target: int, previous: dict[int, tuple[int, int]]) -> list[int]:
        path, node = [], int(target)
        while node != int(source):
            item = previous.get(node)
            if item is None:
                return []
            parent, aid = item
            path.append(aid)
            node = parent
        return list(reversed(path))

    def _middle(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, max_m: float):
        a, b = int(arc_a), int(arc_b)
        if min(a, b) < 0 or max(a, b) >= self.net.n_arcs:
            return np.inf, []
        fa, fb = float(np.clip(frac_a, 0, 1)), float(np.clip(frac_b, 0, 1))
        source, target = int(self.net.arcs[a, 1]), int(self.net.arcs[b, 0])
        prefix = (1.0 - fa) * float(self.net.arc_len_m[a])
        suffix = fb * float(self.net.arc_len_m[b])
        remaining = float(max_m) - prefix - suffix
        if remaining < 0:
            return np.inf, []
        if source == target:
            return prefix + suffix, []
        dist, prev = self._bounded_dijkstra(source, remaining)
        middle = dist.get(target, np.inf)
        if not np.isfinite(middle):
            return np.inf, []
        return prefix + middle + suffix, self._reconstruct(source, target, prev)

    def route_distance(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, max_m: float) -> float:
        """Shortest directed distance between fractional positions; inf if unreachable/out of bound."""
        a, b = int(arc_a), int(arc_b)
        if min(a, b) < 0 or max(a, b) >= self.net.n_arcs:
            return float(np.inf)
        fa, fb = float(np.clip(frac_a, 0, 1)), float(np.clip(frac_b, 0, 1))
        if a == b and fb >= fa:
            distance = (fb - fa) * float(self.net.arc_len_m[a])
            return float(distance) if distance <= max_m else float(np.inf)
        distance, _ = self._middle(a, fa, b, fb, max_m)
        return float(distance)

    def shortest_arc_path(self, arc_a: int, arc_b: int, max_m: float) -> list[int]:
        """Intermediate arcs from arc_a to arc_b (exclusive), excluding both endpoint arcs."""
        a, b = int(arc_a), int(arc_b)
        if a == b or min(a, b) < 0 or max(a, b) >= self.net.n_arcs:
            return []
        source, target = int(self.net.arcs[a, 1]), int(self.net.arcs[b, 0])
        if source == target:
            return []
        dist, prev = self._bounded_dijkstra(source, max_m)
        if target not in dist:
            return []
        return self._reconstruct(source, target, prev)
