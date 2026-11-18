"""Tests for P2 modules: metrics, features, model zoo."""
import numpy as np
import pandas as pd
import pytest

from pathsnap import schema
from pathsnap.candidates import generate_candidates
from pathsnap.config import CFG
from pathsnap.features import FEATURE_COLS, add_features
from pathsnap.io import load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import (attach_predictions, candidate_metrics, collapse_route, route_continuity,
                              route_overlap, top1_accuracy, top1_from_candidates)
from pathsnap.models import MODEL_FAMILIES, build_model
from pathsnap.network import RoadNetwork


def test_candidate_metrics_perfect_and_counts():
    m = candidate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.9, 0.8]))
    assert m["macro_f1"] == 1.0 and m["tp"] == 2 and m["tn"] == 2 and m["log_loss"] < 0.25


def test_top1_and_attach():
    cands = pd.DataFrame({"trip_id": "a", "point_idx": [0, 0, 1, 1], "arc_id": [5, 6, 7, 8], "prob": [0.2, 0.7, 0.9, 0.1]})
    top = top1_from_candidates(cands)
    assert top.arc_id.tolist() == [6, 7]
    pts = pd.DataFrame({"trip_id": "a", "point_idx": [0, 1, 2], "ref_arc": [6, 8, 9]})
    out = attach_predictions(pts, top)
    assert out.pred_arc.tolist() == [6, 7, -1]  # point 2 has no candidate -> -1, counted wrong
    assert top1_accuracy(out.pred_arc, out.ref_arc) == pytest.approx(1 / 3)
    assert top1_accuracy(np.array([1, 2]), np.array([1, -1])) == 1.0  # unlabelled ignored


def test_route_helpers():
    np.testing.assert_array_equal(collapse_route(np.array([3, 3, -1, 4, 4, 3])), [3, 4, 3])
    ov = route_overlap(np.array([1, 2, 3]), np.array([2, 3, 4, 5]))
    assert ov["route_precision"] == pytest.approx(2 / 3) and ov["route_recall"] == 0.5


def _small():
    trip = load_trip(CFG["paths"]["train_dir"], "00000046", with_route=True)
    net = RoadNetwork(trip)
    xy = net.project_track(trip.track)
    lab = align_points_to_route(net, xy, trip.route, 50.0)
    return trip, net, xy, lab


def test_continuity_of_reference_labels_is_high():
    trip, net, xy, lab = _small()
    assert route_continuity(lab, net, 3) > 0.95


def test_features_complete_and_finite():
    trip, net, xy, lab = _small()
    c = generate_candidates(trip, net, 8, 60.0, lab, track_xy=xy)
    f = add_features(c, xy, trip.track[:, 2], net)
    assert list(schema.FEATURE_COLS) == FEATURE_COLS
    assert all(col in f.columns for col in FEATURE_COLS) and len(f) == len(c)
    assert np.isfinite(f[FEATURE_COLS].to_numpy(dtype=float)).all()
    assert f["heading_diff_deg"].between(0, 180).all()


@pytest.mark.parametrize("name", list(MODEL_FAMILIES))
def test_every_model_gives_probabilities(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURE_COLS))).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
    p = build_model(name, {"n_neighbors": 5} if name == "knn" else None).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (400,) and np.all((p >= 0) & (p <= 1))
