"""Server-side evaluation: compares a PredictionResult with the reference route.  Owner: P3

The ONLY prediction-time module allowed to read .route. Returns numbers only — the reference
geometry is never sent to the UI.
"""
from __future__ import annotations

from .pipeline import PredictionResult


def evaluate_prediction(result: PredictionResult, partition_dir: str, net) -> dict:
    """Load route, derive per-point labels (labels.align_points_to_route), then return e.g.
    {top1_acc_classifier, top1_acc_hmm, route_continuity_classifier, route_continuity_hmm,
     candidate_miss_rate, route_precision, route_recall, route_f1}. Use metrics.py only."""
    raise NotImplementedError
