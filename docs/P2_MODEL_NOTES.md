# P2 model notes: Phase 1 status

## What's ready
| File | What it does |
|---|---|
| `pathsnap/features.py` | 14 features per candidate. `build_dataset` adds them automatically. |
| `pathsnap/metrics.py` | The one shared definition of every metric: candidate metrics, top-1, miss rate, route continuity, route overlap |
| `pathsnap/models.py` | 6 models from 6 families, with their initial hyperparameters and training objectives |
| `scripts/train_models.py` | GroupKFold(5) CV on train, then refit and evaluate on student test. Writes the CSVs and saves the models. |
| `tests/test_p2_models.py` | Tests for metrics, features and all 6 models |

```bash
python -m scripts.build_dataset                  # adds features (~10 s)
python -m scripts.train_models --models logreg   # ~30 s
python -m scripts.train_models                   # all 6 models (Phase 2)
```

## Features (and why)
| Group | Features | Why |
|---|---|---|
| Distance | `dist_m`, `dist_minus_min_m`, `dist_ratio_min`, `rank`, `n_candidates` | The true road is usually the closest one, but in dense areas *relative* distance matters more |
| Direction | `heading_diff_deg`, `heading_align`, `speed_mps` | Two-way streets are two identical arcs, and only direction separates them. Heading uses a ±3 s window. Alignment fades out when the vehicle is almost stopped. |
| Consistency | `nbr_dist_mean_m`, `nbr_dist_max_m` | The correct road stays close to the neighbouring ±3 fixes too, while a crossing road is only close at the junction |
| Road context | `arc_len_m`, `endpoint_dist_m`, `is_oneway`, `to_out_degree` | Junctions and short arcs are where mistakes happen |

Median values on train (correct vs wrong candidate):
- heading difference: 6° vs 103°
- distance beyond the best candidate: 0 m vs 13 m
- neighbour distance: 8 m vs 26 m

## Phase 1 results: baseline (logistic regression, all features)
| Metric | Value | Partition |
|---|---|---|
| CV macro-F1 | 0.905 ± 0.005 (5 folds) | train (78 trips, GroupKFold by trip) |
| CV top-1 accuracy | 0.882 | train |
| Test macro-F1 / log loss | 0.900 / 0.135 | student test (10 trips) |
| **Test top-1 accuracy** | **0.864** | student test |
| Test route continuity (classifier only) | 0.967 | student test |
| Test route F1 (arc set) | 0.854 | student test |
| Nearest-road rule, top-1 (non-ML reference) | 0.536 | student test |

The nearest-road rule is close to a coin flip on two-way streets (both arcs are equally close). The direction features are what move accuracy from 54% to 86%.

All 6 models already train and predict end to end. They were smoke-tested on 50k rows (results not saved). Expect the tree and boosting models to beat logistic regression.

## Decisions to be able to defend
- **No class re-weighting** (16% positives). Re-weighting distorts the probabilities, which are the HMM's emission inputs. Top-1 depends only on the ranking within each point.
- **GroupKFold by trip.** All candidates of a trip stay in the same fold, otherwise neighbouring fixes would leak between training and validation.
- **KNN is trained on a 200k-row sample**, because its prediction cost grows with training size. This is recorded in `n_train_rows`.
- **Only the MLP has an epoch-wise loss curve** (`results/training_history/`). The rules forbid artificial curves for the other models.

## Phase 2 to-do (P2)
1. Run all 6 models with CV: `python -m scripts.train_models`.
2. Plots in `results/plots/`, made from the CSVs:
   - CV macro-F1 bars with standard deviation
   - test top-1 and macro-F1 bars
   - log loss bars
   - confusion matrices (from the tn/fp/fn/tp columns)
   - MLP loss curve
3. Choose the model, using CV, probability quality, speed and size. Write down the reasoning.
4. `scripts/bayes_opt.py`: Optuna TPE with GroupKFold on train only, at least 50 trials. Save the trials and best parameters.
5. Export `artifacts/model.joblib` and `model_meta.json` (format in `docs/CONTRACTS.md` §5). Write the tuned vs untuned table.
