"""Log-space HMM/Viterbi route reconstruction over per-observation arc candidates (P3)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .routing import Router


def _log_emissions(rows: pd.DataFrame, prob_col: str | None, cfg: dict) -> np.ndarray:
    mode = str(cfg.get("emission", "distance")).lower()
    sigma = max(float(cfg.get("sigma_m", 10.0)), 1e-6)
    d = rows["dist_m"].to_numpy(dtype=float)
    log_d = -0.5 * np.square(d / sigma) - np.log(sigma * np.sqrt(2.0 * np.pi))
    if prob_col and prob_col in rows.columns and mode in {"classifier", "product"}:
        p = np.clip(rows[prob_col].to_numpy(dtype=float), 1e-12, 1.0)
        return np.log(p) + log_d if mode == "product" else np.log(p)
    return log_d


def _decode_chunk(groups, points, router, cfg, prob_col):
    states = [g["arc_id"].to_numpy(dtype=np.int64) for g in groups]
    fracs = [g["proj_frac"].to_numpy(dtype=float) for g in groups]
    emissions = [_log_emissions(g, prob_col, cfg) for g in groups]
    scores = emissions[0].copy()
    backs = [np.full(len(states[0]), -1, dtype=np.int32)]
    beta = max(float(cfg.get("beta_m", 5.0)), 1e-6)
    factor, extra = float(cfg.get("max_route_factor", 4.0)), float(cfg.get("max_route_extra_m", 50.0))
    for i in range(1, len(points)):
        straight = float(np.linalg.norm(cfg["_track_xy"][points[i]] - cfg["_track_xy"][points[i - 1]]))
        bound = max(0.0, factor * straight + extra)
        next_scores = np.full(len(states[i]), -np.inf)
        back = np.full(len(states[i]), -1, dtype=np.int32)
        for j, arc in enumerate(states[i]):
            for k, prev_arc in enumerate(states[i - 1]):
                if not np.isfinite(scores[k]):
                    continue
                rd = router.route_distance(int(prev_arc), fracs[i - 1][k],
                                           int(arc), fracs[i][j], bound)
                if np.isfinite(rd):
                    value = scores[k] - abs(rd - straight) / beta
                    if value > next_scores[j]:
                        next_scores[j], back[j] = value + emissions[i][j], k
        if not np.isfinite(next_scores).any():
            raise RuntimeError("Internal Viterbi error: chunk crosses a dead-end transition")
        scores = next_scores
        backs.append(back)
    state = int(np.argmax(scores))
    decoded = np.empty(len(points), dtype=np.int64)
    for i in range(len(points) - 1, -1, -1):
        decoded[i] = states[i][state]
        if i:
            state = int(backs[i][state])
    return decoded


def viterbi_decode(cands: pd.DataFrame, prob_col: str, router: Router,
                   track_xy: np.ndarray, hmm_cfg: dict) -> np.ndarray:
    """Return an arc id for every GPS point; -1 for observations with no candidate.

    Decoding restarts at segment boundaries, missing-candidate observations, and when no
    transition is reachable from any currently reachable Viterbi state.
    """
    xy = np.asarray(track_xy, dtype=float)
    out = np.full(len(xy), -1, dtype=np.int64)
    if cands.empty or len(xy) == 0:
        return out
    required = {"point_idx", "segment_id", "arc_id", "proj_frac", "dist_m"}
    missing = required.difference(cands.columns)
    if missing:
        raise ValueError(f"candidate table missing HMM columns: {sorted(missing)}")
    work = cands.copy()
    if not prob_col or prob_col not in work.columns:
        prob_col = None
    cfg = dict(hmm_cfg)
    cfg["_track_xy"] = xy
    grouped = {int(p): g.sort_values("arc_id", kind="stable")
               for p, g in work.groupby("point_idx", sort=True)}
    ordered = sorted(p for p in grouped if 0 <= p < len(xy))
    run_points, run_groups, run_segment, active = [], [], None, None

    def flush():
        if run_points:
            out[np.asarray(run_points, dtype=int)] = _decode_chunk(
                run_groups, run_points, router, cfg, prob_col
            )

    beta = max(float(cfg.get("beta_m", 5.0)), 1e-6)
    factor, extra = float(cfg.get("max_route_factor", 4.0)), float(cfg.get("max_route_extra_m", 50.0))
    for point in ordered:
        group = grouped[point]
        segment = int(group["segment_id"].iloc[0])
        if run_points and (point != run_points[-1] + 1 or segment != run_segment):
            flush()
            run_points, run_groups, active = [], [], None
        if not run_points:
            run_segment = segment
            run_points, run_groups = [point], [group]
            active = np.ones(len(group), dtype=bool)
            continue
        previous = run_groups[-1]
        straight = float(np.linalg.norm(xy[point] - xy[run_points[-1]]))
        bound = max(0.0, factor * straight + extra)
        reachable = np.zeros(len(group), dtype=bool)
        prev_rows = previous.reset_index(drop=True)
        curr_rows = group.reset_index(drop=True)
        for k, prev in prev_rows.iterrows():
            if not active[k]:
                continue
            for j, curr in curr_rows.iterrows():
                if np.isfinite(router.route_distance(
                    int(prev["arc_id"]), float(prev["proj_frac"]),
                    int(curr["arc_id"]), float(curr["proj_frac"]), bound
                )):
                    reachable[j] = True
        if not reachable.any():
            flush()
            run_points, run_groups = [point], [group]
            run_segment, active = segment, np.ones(len(group), dtype=bool)
        else:
            run_points.append(point)
            run_groups.append(group)
            active = reachable
    flush()
    return out
