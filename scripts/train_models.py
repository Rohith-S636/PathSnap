"""Train + compare >= 5 models.  Owner: P2
GroupKFold(trip_id) CV on candidates_train -> CV macro-F1 (mean ± std); refit on all train;
evaluate on candidates_test -> macro-F1, precision, recall, log loss, top-1, continuity (classifier-only),
train/inference time. Writes results/model_comparison.csv, results/cv_folds.csv,
results/plots/*.png (CV F1 bars, test top1/F1 bars, log loss bars, confusion matrices, MLP loss curve),
and artifacts/untuned/<model>.joblib. Fixed seed from CFG.
"""
raise NotImplementedError
