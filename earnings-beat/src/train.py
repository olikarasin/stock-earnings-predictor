from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainCLIConfig:
    features_path: Path
    model_path: Path
    model_type: str = "lightgbm"
    test_size: float = 0.2
    random_state: int = 42


def train_from_file(config: TrainCLIConfig):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib

    from .beat_predictor import prepare_features_targets, TrainConfig, train_model

    df = pd.read_parquet(config.features_path)
    X, y = prepare_features_targets(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    model = train_model(X_train, y_train, TrainConfig(model_type=config.model_type, random_state=config.random_state))
    # simple validation metric
    try:
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred_proba)
    except Exception:
        # Some models may not expose predict_proba consistently
        y_pred = model.predict(X_valid)
        auc = roc_auc_score(y_valid, y_pred)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_type": config.model_type, "auc_valid": float(auc)}, config.model_path)
    return float(auc)


