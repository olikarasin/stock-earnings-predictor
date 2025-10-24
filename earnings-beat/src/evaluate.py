from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluateCLIConfig:
    features_path: Path
    model_path: Path


def evaluate_from_file(config: EvaluateCLIConfig):
    import pandas as pd
    from sklearn.metrics import classification_report, roc_auc_score
    import joblib

    from .beat_predictor import prepare_features_targets

    df = pd.read_parquet(config.features_path)
    X, y = prepare_features_targets(df)
    bundle = joblib.load(config.model_path)
    model = bundle["model"]

    try:
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
    except Exception:
        y_pred = model.predict(X)
        y_proba = None
        auc = roc_auc_score(y, y_pred)

    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=False)
    return {"auc": float(auc), "report": report}


