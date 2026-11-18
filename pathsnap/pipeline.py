"""End-to-end inference for one trip. Ground truth is never loaded by prediction code."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import joblib
import numpy as np

from . import schema
from .candidates import generate_candidates
from .config import CFG
from .features import add_features
from .hmm import viterbi_decode
from .io import load_trip
from .network import RoadNetwork
from .routing import Router


@dataclass
class PredictionResult:
    trip_id: str
    n_points: int
    classifier_arc: np.ndarray
    hmm_arc: np.ndarray
    route_arcs: list[int]
    timings_ms: dict = field(default_factory=dict)


class PathSnapPredictor:
    """Load optional P2 classifier artifacts and P3 decoder configuration once."""

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.model_name = "distance-baseline"
        self.feature_cols = ["dist_m"]
        self.candidate_cfg = dict(CFG["candidates"])
        self.hmm_cfg = dict(CFG["hmm"])
        meta_path = self.artifacts_dir / "model_meta.json"
        model_path = self.artifacts_dir / "model.joblib"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model_name = str(meta.get("model_name", self.model_name))
            self.feature_cols = list(meta.get("feature_cols", self.feature_cols))
            self.candidate_cfg.update(meta.get("candidates", {}))
        if model_path.exists():
            self.model = joblib.load(model_path)
        hmm_path = self.artifacts_dir / "hmm_config.json"
        if hmm_path.exists():
            self.hmm_cfg.update(json.loads(hmm_path.read_text(encoding="utf-8")))

    def _score_candidates(self, cands):
        if self.model is None:
            # P3 pre-P2 baseline: closer candidates receive higher normalized scores.
            sigma = max(float(self.hmm_cfg.get("sigma_m", 10.0)), 1e-6)
            cands["prob"] = np.exp(-0.5 * np.square(cands["dist_m"].to_numpy(float) / sigma))
            return cands
        features = add_features
        try:
            enriched = features(cands, self._track_xy, self._track_t, self._net)
        except NotImplementedError as exc:
            raise RuntimeError(
                "A classifier artifact is present, but pathsnap.features.add_features is not implemented. "
                "Complete P2 feature engineering before enabling model inference."
            ) from exc
        absent = [col for col in self.feature_cols if col not in enriched.columns]
        if absent:
            raise ValueError(f"model metadata requires missing feature columns: {absent}")
        matrix = enriched[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(matrix)
            classes = list(getattr(self.model, "classes_", range(probabilities.shape[1])))
            positive = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            enriched["prob"] = probabilities[:, positive]
        else:
            raise TypeError("The exported classifier must implement predict_proba().")
        return enriched

    @staticmethod
    def _classifier_top1(cands, n_points):
        out = np.full(n_points, -1, dtype=np.int64)
        if cands.empty:
            return out
        for point, rows in cands.groupby("point_idx", sort=False):
            score = rows["prob"].to_numpy(float) if "prob" in rows else -rows["dist_m"].to_numpy(float)
            out[int(point)] = int(rows.iloc[int(np.argmax(score))]["arc_id"])
        return out

    @staticmethod
    def _continuous_route(decoded, segment_ids, router, max_m):
        route: list[int] = []
        previous = None
        previous_segment = None
        for arc, segment in zip(decoded, segment_ids):
            arc, segment = int(arc), int(segment)
            if arc < 0:
                previous = None
                previous_segment = None
                continue
            if previous is None or previous_segment != segment:
                if not route or route[-1] != arc:
                    route.append(arc)
            elif arc != previous:
                bridge = router.shortest_arc_path(previous, arc, max_m)
                for mid in bridge:
                    if not route or route[-1] != mid:
                        route.append(int(mid))
                if not route or route[-1] != arc:
                    route.append(arc)
            previous, previous_segment = arc, segment
        return route

    def predict(self, partition_dir: str, trip_id: str) -> tuple[PredictionResult, RoadNetwork]:
        timings = {}
        start = time.perf_counter()
        t0 = time.perf_counter()
        trip = load_trip(partition_dir, trip_id, with_route=False)
        net = RoadNetwork(trip)
        xy = net.project_track(trip.track)
        timings["network"] = (time.perf_counter() - t0) * 1000
        self._net, self._track_xy, self._track_t = net, xy, trip.track[:, 2]
        t0 = time.perf_counter()
        cands = generate_candidates(
            trip, net, int(self.candidate_cfg["k"]), float(self.candidate_cfg["radius_m"]),
            split_s=float(self.hmm_cfg.get("gap_split_s", CFG["gaps"]["split_s"])), track_xy=xy,
        )
        timings["candidates"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cands = self._score_candidates(cands)
        classifier_arc = self._classifier_top1(cands, len(trip.track))
        timings["model"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        router = Router(net)
        hmm_arc = viterbi_decode(cands, "prob", router, xy, self.hmm_cfg)
        timings["viterbi"] = (time.perf_counter() - t0) * 1000
        segment_ids = np.zeros(len(trip.track), dtype=np.int64)
        if len(cands):
            segment_ids = np.zeros(len(trip.track), dtype=np.int64)
            for point, group in cands.groupby("point_idx"):
                segment_ids[int(point)] = int(group["segment_id"].iloc[0])
        max_bridge = float(self.hmm_cfg.get("route_fill_max_m", 2000.0))
        route_arcs = self._continuous_route(hmm_arc, segment_ids, router, max_bridge)
        timings["total"] = (time.perf_counter() - start) * 1000
        result = PredictionResult(trip_id, len(trip.track), classifier_arc, hmm_arc, route_arcs, timings)
        return result, net
