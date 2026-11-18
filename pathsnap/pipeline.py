"""End-to-end inference for ONE trip. The API, scripts/eval_hmm.py and maps all call this.  Owner: P3

GROUND TRUTH NEVER ENTERS HERE: the trip is loaded with with_route=False.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictionResult:
    trip_id: str
    n_points: int
    classifier_arc: np.ndarray  # (N,) argmax-probability arc per point (classifier-only), -1 if none
    hmm_arc: np.ndarray  # (N,) Viterbi arc per point, -1 if none
    route_arcs: list[int]  # final continuous route: deduplicated hmm_arc with gaps filled by shortest paths
    timings_ms: dict = field(default_factory=dict)  # candidates / features / model / viterbi / total


class PathSnapPredictor:
    """Loads artifacts once (model, feature list, hmm config), then predicts many trips."""

    def __init__(self, artifacts_dir: str):
        raise NotImplementedError

    def predict(self, partition_dir: str, trip_id: str) -> tuple[PredictionResult, object]:
        """Returns (result, net) — net (RoadNetwork) is returned so callers can draw geometry."""
        raise NotImplementedError
