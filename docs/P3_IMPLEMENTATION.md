# P3 implementation notes

## Scope and interfaces

P3 owns `pathsnap/routing.py`, `pathsnap/hmm.py`, `pathsnap/pipeline.py`,
`pathsnap/evaluate.py`, and `scripts/eval_hmm.py`. Public call signatures in
`docs/CONTRACTS.md` are preserved.

## Directed routing

`Router` runs bounded Dijkstra over `RoadNetwork.out_arcs`; it never treats the road
network as undirected. A transition from a fractional position on arc A to a fractional
position on arc B is:

`(1 - frac_A) * length(A) + shortest_path(to(A), from(B)) + frac_B * length(B)`.

A same-arc forward movement is handled directly. Invalid IDs, unreachable paths and paths
beyond the caller's distance bound return infinity. Dijkstra distance/predecessor trees are
cached in a bounded LRU cache by source node and distance bound. `shortest_arc_path` returns
only intermediate arcs, excluding the selected endpoint arcs.

## HMM / Viterbi

Each candidate arc is a hidden state for its GPS observation. Emissions are calculated in
log space. `distance` uses a Gaussian distance likelihood; `classifier` uses the supplied
positive-class probability; `product` combines classifier log probability and the Gaussian
distance term. Probabilities are clipped away from zero before taking logarithms.

For consecutive observations, a transition is allowed only when directed network routing
returns a finite distance within `max_route_factor * straight_distance + max_route_extra_m`.
Its log score is `-abs(route_distance - straight_distance) / beta_m`. Viterbi maximizes the
sum of log emission and transition scores.

Decoding runs independently across `segment_id` boundaries, missing-candidate observations,
and dead ends. A point with no candidate remains -1. At a dead end, the decoder closes the
previous run and restarts at the current observation rather than inventing a connection.

## Prediction pipeline and P2 hand-off

`PathSnapPredictor` loads `model.joblib`, `model_meta.json` and
`hmm_config.json` when available. Candidate generation settings and feature columns are
read from model metadata, not silently inferred from the current model. The classifier must
expose `predict_proba`; positive class 1 becomes the candidate probability.

Until P2 supplies a trained model and a working `features.add_features`, the pipeline uses
a distance-only baseline and forces distance emissions. It does not claim classifier
performance. Once P2 is merged and artifacts are exported, the same pipeline calls the P2
feature function and feeds classifier probabilities into Viterbi.

The prediction path calls `load_trip(..., with_route=False)`. Reference routes are loaded
only by `evaluate_prediction`, which returns numeric metrics and never adds reference
geometry to the prediction result.

## Evaluation and tuning

Run `python -m scripts.eval_hmm` after the student-testing partition is available locally.
It writes per-trip classifier-only/HMM metrics to `results/hmm_comparison.csv`, the tested
parameter grid to `results/hmm_param_search.csv`, and the selected decoder settings to
`artifacts/hmm_config.json`. The search uses student-testing trips only; it never reads the
reserved evaluator partition. Run with `--quick` for a one-configuration smoke test or
`--trips <id> ...` for a restricted test.

The synthetic tests in `tests/test_p3_routing_hmm.py` cover directed/fractional routing,
distance bounds, connected Viterbi decoding, missing candidates and segment boundaries.
Run `pytest -q tests/test_p3_routing_hmm.py` and then `pytest -q`.

## Current dependency boundary

The repository's current main branch still contains P2 stubs in `features.py` and
`models.py`, and does not contain the local dataset or exported model artifacts. Therefore
the distance-only P3 path and synthetic tests can be implemented independently, but the
classifier-backed comparison, parameter search over classifier emissions, and real-trip
results must be rerun after P2 artifacts and local data are present. No experimental metrics
are pre-filled or fabricated here.
