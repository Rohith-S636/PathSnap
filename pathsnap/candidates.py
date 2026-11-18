"""Candidate generation: for each GPS point, the k nearest arcs within radius.  Owner: P1

Output is the candidate table defined in schema.py (KEY_COLS + CANDIDATE_BASE_COLS [+ label]).
Points with zero candidates still need to be recorded so HMM can handle them (keep a row with
arc_id = -1, or return the list of empty point_idx separately — agree with P3 and document here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Trip
from .network import RoadNetwork


def segment_track(t: np.ndarray, split_s: float) -> np.ndarray:
    """(N,) segment_id: increments wherever the time gap to the previous point exceeds split_s."""
    raise NotImplementedError


def generate_candidates(trip: Trip, net: RoadNetwork, k: int, radius_m: float,
                        point_labels: np.ndarray | None = None) -> pd.DataFrame:
    """Build the candidate table for one trip. If point_labels is given, add the `label` column."""
    raise NotImplementedError
