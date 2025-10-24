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

    # Train with graceful fallback if LightGBM is unavailable on macOS (libomp)
    try:
        model = train_model(
            X_train,
            y_train,
            TrainConfig(model_type=config.model_type, random_state=config.random_state),
        )
    except Exception as e:
        if config.model_type == "lightgbm":
            print(
                "LightGBM unavailable. To enable, install OpenMP: 'brew install libomp' then 'pip install --force-reinstall lightgbm'.\n"
                "Proceeding with RandomForest fallback."
            )
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=400, random_state=config.random_state, class_weight="balanced"
            )
            model.fit(X_train, y_train)
        else:
            raise
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


# -----------------------------
# New training pipeline (dataset build + models + calibration)
# -----------------------------

def _default_universe() -> list[str]:
    return [
        "AAPL","MSFT","AMZN","GOOGL","NVDA","META","TSLA","JPM","BAC","WFC",
        "V","MA","UNH","JNJ","PG","XOM","CVX","PFE","KO","PEP",
        "INTC","AMD","NFLX","ORCL","CSCO","DIS","HD","MCD","ABBV","COST",
    ]


def _read_tickers_file(path: Optional[Path]) -> list[str]:
    if not path:
        return _default_universe()
    try:
        text = Path(path).read_text()
    except Exception:
        return _default_universe()
    raw = [t.strip().upper() for part in text.splitlines() for t in part.replace(","," ").split()]
    tickers = [t for t in raw if t and not t.startswith("#")]
    return tickers or _default_universe()


def _time_split(df, date_col: str = "event_date"):
    import numpy as np
    import pandas as pd

    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    train = df.iloc[:i1]
    valid = df.iloc[i1:i2]
    test = df.iloc[i2:]
    return train, valid, test


def _precision_at_top_kpct(y_true, y_proba, k_pct: float = 0.10) -> float:
    import numpy as np

    n = len(y_true)
    k = max(1, int(np.ceil(n * k_pct)))
    order = np.argsort(-y_proba)
    top_idx = order[:k]
    return float(np.mean(y_true[top_idx])) if k > 0 else float("nan")


def _evaluate_metrics(y_true, y_proba) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")
    brier = float(np.mean((y_true - y_proba) ** 2))
    p_at_10 = _precision_at_top_kpct(y_true, y_proba, 0.10)
    return {"auc": float(auc), "brier": float(brier), "precision_at_10pct": float(p_at_10)}


def train_pipeline(tickers_file: Optional[Path] = None, years: int = 8, models_dir: Path | None = None, random_state: int = 42) -> dict:
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier

    from .features import build_labeled_events, get_feature_columns

    models_dir = models_dir or (Path(__file__).resolve().parents[1] / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    tickers = _read_tickers_file(tickers_file)
    print(f"Building labeled dataset for {len(tickers)} tickers...")
    df = build_labeled_events(tickers, lookback_years=years)
    if df is None or df.empty:
        raise RuntimeError("No labeled events built; cannot train")

    feature_cols = get_feature_columns()
    # Train/valid/test split by time
    train_df, valid_df, test_df = _time_split(df, date_col="event_date")

    # Impute using TRAIN medians to avoid leakage
    def impute_with_train_medians(train_df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        X_other = other[feature_cols].copy()
        medians = train_df[feature_cols].median(numeric_only=True)
        for c in feature_cols:
            X_other[c] = pd.to_numeric(X_other[c], errors="coerce").fillna(float(medians.get(c, 0.0)))
        return X_other

    X_train = impute_with_train_medians(train_df, train_df)
    y_train = train_df["beat"].astype(int).values
    X_valid = impute_with_train_medians(train_df, valid_df)
    y_valid = valid_df["beat"].astype(int).values
    X_test = impute_with_train_medians(train_df, test_df)
    y_test = test_df["beat"].astype(int).values

    # Models
    # Train LightGBM, fallback to RandomForest if unavailable
    try:
        lgbm = LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            class_weight="balanced",
        )
        lgbm.fit(X_train, y_train)
    except Exception:
        print(
            "LightGBM unavailable. To enable, install OpenMP: 'brew install libomp' then 'pip install --force-reinstall lightgbm'.\n"
            "Falling back to RandomForest for primary model."
        )
        from sklearn.ensemble import RandomForestClassifier

        lgbm = RandomForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )
        lgbm.fit(X_train, y_train)

    logreg = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", class_weight="balanced")
    logreg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)

    # Calibrate LGBM on valid
    valid_n = len(y_valid)
    pos_valid = int(np.sum(y_valid))
    method = "isotonic" if valid_n >= 1000 and 0 < pos_valid < valid_n else "sigmoid"
    calibrator = CalibratedClassifierCV(base_estimator=lgbm, method=method, cv="prefit")
    calibrator.fit(X_valid, y_valid)

    # Metrics on test (use calibrated probabilities for LGBM)
    from sklearn.metrics import roc_auc_score

    proba_lgbm = calibrator.predict_proba(X_test)[:, 1]
    proba_lr = logreg.predict_proba(X_test)[:, 1]
    proba_rf = rf.predict_proba(X_test)[:, 1]
    metrics_lgbm = _evaluate_metrics(y_test, proba_lgbm)
    metrics_lr = _evaluate_metrics(y_test, proba_lr)
    metrics_rf = _evaluate_metrics(y_test, proba_rf)

    print("Test metrics (LGBM, calibrated):", metrics_lgbm)
    print("Test metrics (LogReg):", metrics_lr)
    print("Test metrics (RandomForest):", metrics_rf)

    # Persist artifacts
    joblib.dump(lgbm, models_dir / "beat_model.pkl")
    joblib.dump(calibrator, models_dir / "calibrator.pkl")
    # Feature spec retains order
    (models_dir / "feature_spec.json").write_text(json.dumps(feature_cols))
    # Feature stats for lightweight driver explanation
    q = train_df[feature_cols].quantile([0.25, 0.75], numeric_only=True)
    q25 = q.loc[0.25]
    q75 = q.loc[0.75]
    med = train_df[feature_cols].median(numeric_only=True)
    feature_stats = {
        "median": {c: float(med.get(c, 0.0)) for c in feature_cols},
        "iqr": {c: float(max(q75.get(c, 0.0) - q25.get(c, 0.0), 0.0)) for c in feature_cols},
    }
    (models_dir / "feature_stats.json").write_text(json.dumps(feature_stats))
    (models_dir / "metrics.json").write_text(json.dumps({
        "lgbm_calibrated": metrics_lgbm,
        "logreg": metrics_lr,
        "random_forest": metrics_rf,
    }, indent=2))

    return {
        "lgbm": metrics_lgbm,
        "logreg": metrics_lr,
        "random_forest": metrics_rf,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
    }


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(prog="earnings-beat-train")
    p.add_argument("--tickers-file", type=Path, default=None)
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--random-state", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = train_pipeline(
            tickers_file=args.tickers_file,
            years=args.years,
            models_dir=args.models_dir,
            random_state=args.random_state,
        )
        print("Train/Valid/Test sizes:", result["n_train"], result["n_valid"], result["n_test"])
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


