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
        nxt_scores = np.full(len(states[i]), -np.inf)
        back = np.full(len(states[i]), -1, dtype=np.int32)
        for j, arc in enumerate(states[i]):
            best, best_k = -np.inf, -1
            for k, prev_arc in enumerate(states[i - 1]):
                if not np.isfinite(scores[k]):
                    continue
                rd = router.route_distance(int(prev_arc), fracs[i - 1][k],
                                           int(arc), fracs[i][j], bound)
                if np.isfinite(rd):
                    value = scores[k] - abs(rd - straight) / beta
                    if value > best:
                        best, best_k = value, k
            if best_k >= 0:
                nxt_scores[j], back[j] = best + emissions[i][j], best_k
        if not np.isfinite(nxt_scores).any():
            # Caller splits at this dead end; reaching here indicates a numerical edge case.
            raise RuntimeError("Viterbi chunk contains an unreachable transition boundary")
        scores = nxt_scores
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

    Runs split at segment_id changes, missing candidate observations, and dead ends. A dead
    end closes the previous Viterbi run and starts a new run at the current observation.
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
    run_points, run_groups, run_segment = [], [], None

    def flush():
        if run_points:
            out[np.asarray(run_points, dtype=int)] = _decode_chunk(
                run_groups, run_points, router, cfg, prob_col
            )

    for point in ordered:
        group = grouped[point]
        segment = int(group["segment_id"].iloc[0])
        if run_points and (point != run_points[-1] + 1 or segment != run_segment):
            flush()
            run_points, run_groups = [], []
        if not run_points:
            run_segment = segment
            run_points.append(point)
            run_groups.append(group)
            continue
        # Detect whether any legal transition reaches the current candidate layer.
        prev_group = run_groups[-1]
        straight = float(np.linalg.norm(xy[point] - xy[run_points[-1]]))
        bound = max(0.0, float(cfg.get("max_route_factor", 4.0)) * straight
                    + float(cfg.get("max_route_extra_m", 50.0)))
        reachable = False
        for _, prev in prev_group.iterrows():
            for _, curr in group.iterrows():
                if np.isfinite(router.route_distance(
                    int(prev["arc_id"]), float(prev["proj_frac"]),
                    int(curr["arc_id"]), float(curr["proj_frac"]), bound
                )):
                    reachable = True
                    break
            if reachable:
                break
        if not reachable:
            flush()
            run_points, run_groups = [point], [group]
            run_segment = segment
        else:
            run_points.append(point)
            run_groups.append(group)
    flush()
    return out
