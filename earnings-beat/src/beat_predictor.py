from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


ModelType = Literal["sklearn", "xgboost", "lightgbm"]


@dataclass
class TrainConfig:
    model_type: ModelType = "lightgbm"
    random_state: int = 42


def train_model(X, y, config: TrainConfig):
    """Train a simple binary classifier depending on model_type.

    Heavy imports are inside to keep import-time light.
    """
    if config.model_type == "sklearn":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=200, random_state=config.random_state)
        model.fit(X, y)
        return model

    if config.model_type == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(X, y)
        return model

    if config.model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.random_state,
        )
        model.fit(X, y)
        return model

    raise ValueError(f"Unsupported model_type: {config.model_type}")


def prepare_features_targets(df) -> Tuple[object, object]:
    """Split a DataFrame into features X and target y."""
    import pandas as pd

    if "target" not in df.columns:
        raise ValueError("Features DataFrame must contain 'target' column")
    feature_cols = [c for c in df.columns if c not in {"Date", "target"}]
    X = df[feature_cols]
    y = df["target"].astype(int)
    return X, y


