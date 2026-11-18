"""Shortest-path distances on the directed road graph, used by HMM transitions.  Owner: P3

Kept separate from network.py so P1 and P3 never edit the same file.
Needs: network distance from (arc a, position fa) to (arc b, position fb), bounded by a max
distance so Dijkstra stays cheap on 300k-arc networks (cache per source node within a segment).
Return np.inf when unreachable.
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


class Router:
    def __init__(self, net: RoadNetwork):
        raise NotImplementedError

    def route_distance(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, max_m: float) -> float:
        raise NotImplementedError

    def shortest_arc_path(self, arc_a: int, arc_b: int, max_m: float) -> list[int]:
        """Intermediate arc ids from arc_a to arc_b (exclusive) — used to fill gaps in the final route."""
        raise NotImplementedError
