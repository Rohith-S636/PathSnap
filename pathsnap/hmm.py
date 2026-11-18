"""HMM + Viterbi decoding over candidate arcs.  Owner: P3

Hidden states  : candidate arcs of each GPS point (rows of the candidate table).
Emission       : from the classifier probability (CFG hmm.emission = classifier), optionally
                 combined with a Gaussian distance term (sigma_m). Work in LOG space.
Transition     : exp(-|route_dist - straight_dist| / beta_m) via Router; unreachable -> -inf.
Gaps           : decode each segment_id independently (track split at gaps > gaps.split_s).
Dead ends      : if every transition into step i is -inf, break and restart decoding at i
                 (log it — report how often this happens).
No candidates  : point gets arc -1; decoding continues from the next point with candidates.

Start with emission = distance so you can develop BEFORE P2 ships a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .routing import Router


def viterbi_decode(cands: pd.DataFrame, prob_col: str, router: Router, track_xy: np.ndarray, hmm_cfg: dict) -> np.ndarray:
    """Returns (N,) int: decoded arc_id per GPS point, -1 where no candidate."""
    raise NotImplementedError
