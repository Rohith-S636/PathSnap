"""Metric definitions — ONE implementation used by training, HMM evaluation and the API.  Owner: P2

Definitions (write these exact definitions into the report):
  candidate-level  : accuracy, precision, recall, macro-F1 (over classes {0,1}), log loss,
                     confusion matrix — on rows with label != -1.
  top1_accuracy    : per labelled GPS point, argmax-probability candidate == reference arc.
                     Points whose reference arc is not among candidates count as WRONG.
  candidate_miss_rate : share of labelled points whose reference arc is absent from candidates.
  route_continuity : share of consecutive point pairs whose predicted arcs are equal or reachable
                     within CFG metrics.continuity_max_hops arc hops (network successors).
  sequence-level   : arc-set precision/recall/F1 of the predicted route vs the reference route.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import RoadNetwork


def candidate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    raise NotImplementedError


def top1_accuracy(point_pred_arc: np.ndarray, point_ref_arc: np.ndarray) -> float:
    """Both (N,) arrays of arc_id; ignore points with ref == -1."""
    raise NotImplementedError


def candidate_miss_rate(cands: pd.DataFrame, point_ref_arc: np.ndarray) -> float:
    raise NotImplementedError


def route_continuity(point_pred_arc: np.ndarray, net: RoadNetwork, max_hops: int) -> float:
    raise NotImplementedError


def route_overlap(pred_route: np.ndarray, ref_route: np.ndarray) -> dict:
    """Arc-set precision / recall / F1 between predicted and reference arc sequences."""
    raise NotImplementedError
