"""Road network: projected geometry, spatial index, connectivity.  Owner: P1

Arcs are DIRECTED (reverse-arc share varies 7%..100% per trip -> one-way roads exist).
Networks range from ~650 to ~307k arcs, so candidate lookup MUST use a spatial index
(shapely STRtree over segments, or scipy cKDTree over densified arc points).
"""
from __future__ import annotations

import numpy as np

from .geo import LocalProjector
from .io import Trip


class RoadNetwork:
    def __init__(self, trip: Trip):
        """Build from a Trip (route is NOT used). Must set at least:

        self.proj        LocalProjector centred on the track centroid
        self.node_xy     (V, 2) projected node coordinates (m)
        self.arcs        (E, 2) from/to node ids
        self.arc_len_m   (E,) arc lengths
        self.arc_bearing (E,) arc bearings in degrees
        """
        raise NotImplementedError

    def query_arcs(self, x: float, y: float, radius_m: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Up to k nearest arcs within radius of (x, y). Returns (arc_ids, dist_m, proj_frac), sorted by dist."""
        raise NotImplementedError

    def successor_arcs(self, arc_id: int) -> np.ndarray:
        """Arc ids whose from_node == to_node of arc_id (legal next arcs)."""
        raise NotImplementedError

    def arc_polyline_lonlat(self, arc_ids: np.ndarray) -> list[tuple[float, float]]:
        """(lat, lon) vertex list for a sequence of arcs — used for maps / API output. NOTE lat first."""
        raise NotImplementedError
