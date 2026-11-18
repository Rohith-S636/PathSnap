"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.  Owner: P2
Objective: GroupKFold CV macro-F1 (or log loss — justify) on TRAIN ONLY. Persist the study
(results/optuna/study.db or trials.csv), best params -> results/optuna/best_params.json,
refit -> artifacts/model.joblib, tuned vs untuned table -> results/tuned_vs_untuned.csv.
"""
raise NotImplementedError
