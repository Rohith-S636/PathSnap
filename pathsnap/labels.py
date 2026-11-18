"""Derive a reference arc for EVERY GPS point from the .route arc sequence.  Owner: P1  (critical path)

The .route file lists the arcs travelled IN ORDER but has far fewer rows than the track
(e.g. 525 arcs vs 1315 points), so there is no given per-point label. Suggested approach:
monotone alignment — assign point i to route position r_i with r_i non-decreasing, minimising
total point-to-arc distance (DP / Viterbi over route positions, banded for long trips).
Points farther than CFG labels.max_dist_m from their aligned arc -> UNLABELLED (-1).

Document this as an assumption in the report: labels are derived, not given, and the
reference route itself may be imperfect (route contiguity is 98-100%, not always 100%).
Used ONLY for training data and server-side evaluation — never inside prediction.
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def align_points_to_route(net: RoadNetwork, track_xy: np.ndarray, route: np.ndarray, max_dist_m: float) -> np.ndarray:
    """Returns (N,) int array: reference arc_id per GPS point, or -1 (schema.UNLABELLED)."""
    raise NotImplementedError
