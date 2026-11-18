"""Compare classifier-only and HMM predictions; tune decoder settings on student-test trips.

Usage: python -m scripts.eval_hmm [--trips ID ...]
Only dataset/train and dataset/test are read. Evaluator data is never used.
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.config import CFG
from pathsnap.io import list_trips
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.evaluate import evaluate_prediction


def _mean_objective(frame: pd.DataFrame) -> float:
    # Route quality is primary; point accuracy and continuity are tie-breakers.
    return float(frame["route_f1"].mean() + 0.25 * frame["top1_acc_hmm"].mean()
                 + 0.10 * frame["route_continuity_hmm"].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="small parameter grid for smoke testing")
    args = parser.parse_args()
    test_dir = CFG["paths"]["test_dir"]
    trips = list_trips(test_dir)
    if args.trips:
        wanted = set(args.trips)
        trips = [tid for tid in trips if tid in wanted]
    if not trips:
        raise SystemExit(f"No student-test trips found in {test_dir}")

    artifacts = Path(CFG["paths"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    results = Path(CFG["paths"]["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    base = PathSnapPredictor(str(artifacts))
    has_model = base.model is not None
    emissions = ["distance", "classifier", "product"] if has_model else ["distance"]
    sigmas = [8.0, 10.0, 15.0] if not args.quick else [10.0]
    betas = [2.0, 5.0, 10.0] if not args.quick else [5.0]
    factors = [2.0, 4.0, 8.0] if not args.quick else [4.0]
    configs = []
    for emission, sigma, beta, factor in itertools.product(emissions, sigmas, betas, factors):
        configs.append({"emission": emission, "sigma_m": sigma, "beta_m": beta,
                        "max_route_factor": factor, "max_route_extra_m": 50.0,
                        "gap_split_s": CFG["gaps"]["split_s"], "route_fill_max_m": 2000.0})

    search_rows, best_cfg, best_score = [], None, -np.inf
    for index, config in enumerate(configs, 1):
        predictor = PathSnapPredictor(str(artifacts))
        predictor.hmm_cfg.update(config)
        rows = []
        for trip_id in trips:
            result, net = predictor.predict(test_dir, trip_id)
            metrics = evaluate_prediction(result, test_dir, net)
            rows.append(metrics)
        frame = pd.DataFrame(rows)
        score = _mean_objective(frame)
        search_rows.append({**config, "objective": score, "mean_route_f1": frame["route_f1"].mean(),
                            "mean_top1_hmm": frame["top1_acc_hmm"].mean(),
                            "mean_continuity_hmm": frame["route_continuity_hmm"].mean(),
                            "n_trips": len(frame)})
        if score > best_score:
            best_score, best_cfg = score, config
        print(f"[{index}/{len(configs)}] {config} -> objective={score:.4f}")

    # Re-evaluate classifier-only vs the selected HMM configuration and persist per-trip evidence.
    predictor = PathSnapPredictor(str(artifacts))
    predictor.hmm_cfg.update(best_cfg)
    comparison = []
    for trip_id in trips:
        result, net = predictor.predict(test_dir, trip_id)
        comparison.append({"trip_id": trip_id, **evaluate_prediction(result, test_dir, net),
                           **{f"time_{k}_ms": v for k, v in result.timings_ms.items()}})
    pd.DataFrame(search_rows).to_csv(results / "hmm_param_search.csv", index=False)
    pd.DataFrame(comparison).to_csv(results / "hmm_comparison.csv", index=False)
    (artifacts / "hmm_config.json").write_text(json.dumps(best_cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Selected HMM configuration: {best_cfg}")
    print(f"Saved {results / 'hmm_param_search.csv'} and {results / 'hmm_comparison.csv'}")
    print(f"Saved {artifacts / 'hmm_config.json'}")
    if not has_model:
        print("NOTE: P2 model artifact absent; this is the distance-only baseline, not classifier evaluation.")


if __name__ == "__main__":
    main()
