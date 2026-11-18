"""Model zoo: >= 5 models from different families.  Owner: P2

Every model is an sklearn-compatible estimator (wrapped in a Pipeline with its scaler if needed)
exposing predict_proba -> P(candidate is correct arc). These probabilities become HMM emissions,
so probability quality (log loss) matters, not just F1.

Suggested zoo (6, one spare in case one is weak/slow):
  logreg   - Linear/probabilistic   LogisticRegression (+StandardScaler)
  gnb      - Probabilistic          GaussianNB
  knn      - Instance-based         KNeighborsClassifier(weights="distance") (+StandardScaler)
  rf       - Ensemble (bagging)     RandomForestClassifier
  lgbm     - Boosting               LGBMClassifier
  mlp      - Neural network         MLPClassifier (+StandardScaler) -> has loss_curve_ for epoch plots
Class imbalance (1 positive vs ~k-1 negatives per point): consider class_weight / is_unbalance.
"""
from __future__ import annotations

MODEL_FAMILIES = {
    "logreg": "Linear / probabilistic",
    "gnb": "Probabilistic (generative)",
    "knn": "Instance-based",
    "rf": "Ensemble (bagging)",
    "lgbm": "Boosting",
    "mlp": "Neural network",
}


def build_model(name: str, params: dict | None = None, seed: int = 42):
    """Return an unfitted sklearn estimator/Pipeline for `name`, with `params` overriding defaults."""
    raise NotImplementedError
