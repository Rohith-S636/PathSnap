"""Feature engineering on the candidate table.  Owner: P2

Input: candidate table for ONE trip (+ the trip's projected track and RoadNetwork).
Output: same rows with all schema.FEATURE_COLS filled. Must be identical at train and inference
time (the API calls this exact function). Any fitted transform (scaler etc.) goes INSIDE the
sklearn Pipeline in models.py, fitted on training folds only — not here.

Ideas: heading difference (GPS movement bearing vs arc bearing, direction-aware for one-ways),
speed, rank of candidate by distance, distance minus min distance at that point, number of
candidates, change in distance vs previous point, arc length, node degree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import RoadNetwork


def add_features(cands: pd.DataFrame, track_xy: np.ndarray, t: np.ndarray, net: RoadNetwork) -> pd.DataFrame:
    raise NotImplementedError
