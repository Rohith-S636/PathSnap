"""Log-space HMM/Viterbi route reconstruction over per-observation arc candidates (P3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .routing import Router


def _log_emissions(rows: pd.DataFrame, prob_col: str | None, cfg: dict) -> np.ndarray:
    mode = str(cfg.get("emission", "distance")).lower()
    sigma = max(float(cfg.get("sigma_m", 10.0)), 1e-6)
    distance = rows["dist_m"].to_numpy(dtype=float)
    # Gaussian distance likelihood, evaluated directly in log space.
    log_distance = -0.5 * np.square(distance / sigma) - np.log(sigma * np.sqrt(2.0 * np.pi))
    use_prob = prob_col is not None and prob_col in rows.columns and mode in {"classifier", "product"}
    if use_prob:
        p = np.clip(rows[prob_col].to_numpy(dtype=float), 1e-12, 1.0)
        log_prob = np.log(p)
        return log_prob + log_distance if mode == "product" else log_prob
    return log_distance


def _decode_run(rows: pd.DataFrame, points: np.ndarray, router: Router, cfg: dict,
                prob_col: str | None) -> np.ndarray:
    """Decode one run of consecutive observations that all have candidates."""
    n = len(points)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    groups = [rows.loc[rows["point_idx"] == int(i)].sort_values("arc_id", kind="stable") for i in points]
    if any(g.empty for g in groups):
        return out
    states = [g["arc_id"].to_numpy(dtype=np.int64) for g in groups]
    fracs = [g["proj_frac"].to_numpy(dtype=float) for g in groups]
    emissions = [_log_emissions(g, prob_col, cfg) for g in groups]
    scores = emissions[0].copy()
    backs: list[np.ndarray] = [np.full(len(states[0]), -1, dtype=np.int32)]
    beta = max(float(cfg.get("beta_m", 5.0)), 1e-6)
    factor = float(cfg.get("max_route_factor", 4.0))
    max_extra = float(cfg.get("max_route_extra_m", 50.0))
    for i in range(1, n):
        prev_xy = cfg["_track_xy"][points[i - 1]]
        curr_xy = cfg["_track_xy"][points[i]]
        straight = float(np.hypot(*(curr_xy - prev_xy)))
        max_route = max(0.0, factor * straight + max_extra)
        next_scores = np.full(len(states[i]), -np.inf, dtype=float)
        back = np.full(len(states[i]), -1, dtype=np.int32)
        for j, arc in enumerate(states[i]):
            best = -np.inf
            best_k = -1
            for k, prev_arc in enumerate(states[i - 1]):
                if not np.isfinite(scores[k]):
                    continue
                route_d = router.route_distance(
                    int(prev_arc), float(fracs[i - 1][k]),
                    int(arc), float(fracs[i][j]), max_route,
                )
                if not np.isfinite(route_d):
                    continue
                transition = -abs(route_d - straight) / beta
                candidate = scores[k] + transition
                if candidate > best:
                    best, best_k = candidate, k
            if best_k >= 0:
                next_scores[j] = best + emissions[i][j]
                back[j] = best_k
        # A disconnected transition layer starts a fresh Viterbi segment at this observation.
        if not np.isfinite(next_scores).any():
            next_scores = emissions[i].copy()
            back.fill(-1)
        scores = next_scores
        backs.append(back)
    state_idx = int(np.argmax(scores))
    for i in range(n - 1, -1, -1):
        out[i] = states[i][state_idx]
        parent = int(backs[i][state_idx]) if i < len(backs) else -1
        if parent < 0:
            if i > 0:
                state_idx = int(np.argmax(emissions[i - 1]))
        else:
            state_idx = parent
    return out


def viterbi_decode(cands: pd.DataFrame, prob_col: str, router: Router,
                   track_xy: np.ndarray, hmm_cfg: dict) -> np.ndarray:
    """Return one decoded arc per GPS point; -1 denotes an observation with no candidate.

    Candidate rows retain their point_idx/arc_id correspondence. segment_id boundaries and
    observations with no candidates split independent decoding runs. The decoder restarts
    from emissions when every incoming transition is invalid or exceeds the configured bound.
    """
    n_points = len(track_xy)
    decoded = np.full(n_points, -1, dtype=np.int64)
    if cands.empty or n_points == 0:
        return decoded
    required = {"point_idx", "segment_id", "arc_id", "proj_frac", "dist_m"}
    missing = required.difference(cands.columns)
    if missing:
        raise ValueError(f"candidate table missing HMM columns: {sorted(missing)}")
    work = cands.copy()
    if prob_col not in work.columns:
        prob_col = None
    cfg = dict(hmm_cfg)
    cfg["_track_xy"] = np.asarray(track_xy, dtype=float)
    point_segment = work.groupby("point_idx", sort=True)["segment_id"].first().to_dict()
    candidate_points = set(int(i) for i in point_segment)
    ordered = sorted(candidate_points)
    if not ordered:
        return decoded
    run: list[int] = []
    run_segment = None
    def flush():
        if run:
            decoded[np.asarray(run, dtype=int)] = _decode_run(work, np.asarray(run, dtype=int), router, cfg, prob_col)
    for point in ordered:
        seg = int(point_segment[point])
        if point < 0 or point >= n_points:
            continue
        if run and (point != run[-1] + 1 or seg != run_segment):
            flush()
            run.clear()
        if not run:
            run_segment = seg
        run.append(point)
    flush()
    return decoded
