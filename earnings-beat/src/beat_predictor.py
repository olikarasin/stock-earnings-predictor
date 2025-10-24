from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Optional


# Retain legacy training helpers for compatibility in other modules
ModelType = Literal["sklearn", "xgboost", "lightgbm"]


@dataclass
class TrainConfig:
    model_type: ModelType = "lightgbm"
    random_state: int = 42


def train_model(X, y, config: TrainConfig):
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
    import pandas as pd

    if "target" not in df.columns:
        raise ValueError("Features DataFrame must contain 'target' column")
    feature_cols = [c for c in df.columns if c not in {"Date", "target"}]
    X = df[feature_cols]
    y = df["target"].astype(int)
    return X, y


# -----------------------------
# User-facing CLI for single-ticker prediction
# -----------------------------

def _load_artifacts(models_dir: Path):
    import json
    import joblib

    model = joblib.load(models_dir / "beat_model.pkl")
    calibrator = joblib.load(models_dir / "calibrator.pkl")
    spec_path = models_dir / "feature_spec.json"
    features: list[str]
    medians: dict[str, float] = {}
    if spec_path.exists():
        try:
            data = json.loads(spec_path.read_text())
            if isinstance(data, dict):
                features = list(data.get("features", []))
                med = data.get("medians")
                if isinstance(med, dict):
                    medians = {str(k): float(v) for k, v in med.items()}
            elif isinstance(data, list):
                features = [str(x) for x in data]
            else:
                features = []
        except Exception:
            features = []
    else:
        features = []
    # Optional separate medians file
    med_path = models_dir / "feature_medians.json"
    if med_path.exists():
        try:
            med_data = json.loads(med_path.read_text())
            if isinstance(med_data, dict):
                medians.update({str(k): float(v) for k, v in med_data.items()})
        except Exception:
            pass
    return model, calibrator, features, medians


def _median_impute_row(df_row, features: list[str], medians: dict[str, float]):
    import pandas as pd
    import numpy as np

    # Ensure all columns exist
    X = pd.DataFrame([{}])
    for c in features:
        X[c] = df_row.get(c, np.nan)
    # Impute using provided medians; default 0.0 if missing
    for c in features:
        val = X.at[0, c]
        if pd.isna(val):
            X.at[0, c] = float(medians.get(c, 0.0))
        else:
            try:
                X.at[0, c] = float(val)
            except Exception:
                X.at[0, c] = float(medians.get(c, 0.0))
    return X


def _compute_top_drivers(model, X_row, features: list[str], medians: dict[str, float]) -> list[tuple[str, float]]:
    import numpy as np

    # Try SHAP if available
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_row)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        sv = np.array(vals)[0]
        idx = np.argsort(-np.abs(sv))[:3]
        return [(features[i], float(sv[i])) for i in idx]
    except Exception:
        pass

    # Fallback: standardized z-scores using medians (approximate)
    x = X_row.values.astype(float)[0]
    z = []
    for i, c in enumerate(features):
        m = float(medians.get(c, 0.0))
        denom = abs(m) if abs(m) > 1e-9 else 1.0
        z.append((c, (x[i] - m) / denom))
    z.sort(key=lambda t: -abs(t[1]))
    return z[:3]


def _format_verdict(p: float) -> str:
    if p >= 0.65:
        return "Likely"
    if 0.45 <= p <= 0.64:
        return "Unsure"
    if p <= 0.44:
        return "Unlikely"
    return "Unsure"


def _predict_for_ticker(ticker: str) -> int:
    import joblib
    import pandas as pd

    from .features import build_features_for_inference

    models_dir = Path(__file__).resolve().parents[1] / "models"
    model, calibrator, features, medians = _load_artifacts(models_dir)
    if not features:
        print("Feature spec not found; please train the model first.")
        return 1

    X_raw, notes = build_features_for_inference(ticker)
    row = X_raw.iloc[0]
    X = _median_impute_row(row, features, medians)

    # Predict
    try:
        p_raw = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        p_raw = float(model.predict(X)[0])
    try:
        p = float(calibrator.predict_proba(X)[:, 1][0])
    except Exception:
        p = p_raw

    verdict = _format_verdict(p)
    drivers = _compute_top_drivers(model, X, features, medians)

    # Print output
    print(f"Ticker: {ticker}")
    print(f"Beat probability: {p:.0%}")
    print(f"Verdict: {verdict}")
    if drivers:
        tops = ", ".join([f"{name} ({abs(val):.2f})" for name, val in drivers])
        print(f"Top drivers: {tops}")
    upcoming = notes.get("upcoming_earnings_date") if isinstance(notes, dict) else None
    if upcoming:
        print(f"Notes: upcoming ER date {upcoming}")
    # PT drift counts
    try:
        pt_up = X_raw.iloc[0].get("pt_up_90d")
        pt_down = X_raw.iloc[0].get("pt_down_90d")
        if pd.notna(pt_up) or pd.notna(pt_down):
            print(f"Notes: PT drift 90d up={pt_up}, down={pt_down}")
    except Exception:
        pass
    return 0


def build_parser():
    import argparse
    p = argparse.ArgumentParser(prog="beat-predictor")
    p.add_argument("--ticker", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ticker = args.ticker
    if not ticker:
        try:
            ticker = input("Enter ticker (e.g., AAPL): ").strip().upper()
        except EOFError:
            ticker = None
    if not ticker:
        print("No ticker provided.")
        return 1
    return _predict_for_ticker(ticker)


if __name__ == "__main__":
    raise SystemExit(main())


