"""Server-side evaluation; this is the only inference-adjacent module that reads .route."""
from __future__ import annotations

from collections import deque

import numpy as np

from . import schema
from .candidates import generate_candidates
from .config import CFG
from .io import load_trip
from .labels import align_points_to_route
from .pipeline import PredictionResult
from .network import RoadNetwork


def _top1(pred: np.ndarray, ref: np.ndarray) -> float:
    mask = ref >= 0
    return float(np.mean(pred[mask] == ref[mask])) if mask.any() else float("nan")


def _continuity(pred: np.ndarray, net: RoadNetwork, max_hops: int) -> float:
    valid = [(int(a), i) for i, a in enumerate(pred) if int(a) >= 0]
    pairs = [(a, b) for (a, _), (b, _) in zip(valid, valid[1:])]
    if not pairs:
        return float("nan")
    def reachable(a, target):
        if a == target:
            return True
        seen = {a}
        frontier = [a]
        for _ in range(max(0, int(max_hops))):
            nxt = []
            for arc in frontier:
                for succ in net.successor_arcs(arc):
                    succ = int(succ)
                    if succ == target:
                        return True
                    if succ not in seen:
                        seen.add(succ)
                        nxt.append(succ)
            frontier = nxt
            if not frontier:
                break
        return False
    return float(np.mean([reachable(a, b) for a, b in pairs]))


def _overlap(pred, ref):
    p, r = set(map(int, pred)), set(map(int, ref))
    if not p and not r:
        return 1.0, 1.0, 1.0
    precision = len(p & r) / len(p) if p else 0.0
    recall = len(p & r) / len(r) if r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_prediction(result: PredictionResult, partition_dir: str, net: RoadNetwork) -> dict:
    """Return pointwise, candidate-coverage, continuity and route-set metrics.

    This function deliberately loads the reference route separately from prediction. Reference
    route geometry/IDs are reduced to numeric metrics and are never part of PredictionResult.
    """
    trip = load_trip(partition_dir, result.trip_id, with_route=True)
    xy = net.project_track(trip.track)
    ref = align_points_to_route(net, xy, trip.route, float(CFG["labels"]["max_dist_m"]))
    cands = generate_candidates(
        trip, net, int(CFG["candidates"]["k"]), float(CFG["candidates"]["radius_m"]),
        split_s=float(CFG["gaps"]["split_s"]), track_xy=xy,
    )
    labelled = ref >= 0
    present = np.zeros(len(ref), dtype=bool)
    if len(cands):
        hit = cands.loc[cands["arc_id"].to_numpy() == ref[cands["point_idx"].to_numpy()], "point_idx"].to_numpy()
        # A reference arc is present when a row for that point has the same arc id.
        pairs = set(zip(cands["point_idx"].astype(int), cands["arc_id"].astype(int)))
        present = np.asarray([(i, int(a)) in pairs if a >= 0 else False for i, a in enumerate(ref)])
    miss = float(1.0 - present[labelled].mean()) if labelled.any() else float("nan")
    p, r, f1 = _overlap(result.route_arcs, trip.route)
    return {
        "top1_acc_classifier": _top1(result.classifier_arc, ref),
        "top1_acc_hmm": _top1(result.hmm_arc, ref),
        "route_continuity_classifier": _continuity(result.classifier_arc, net, CFG["metrics"]["continuity_max_hops"]),
        "route_continuity_hmm": _continuity(result.hmm_arc, net, CFG["metrics"]["continuity_max_hops"]),
        "candidate_miss_rate": miss,
        "route_precision": p,
        "route_recall": r,
        "route_f1": f1,
        "n_labelled_points": int(labelled.sum()),
        "n_points": int(len(ref)),
    }
