from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earnings-beat")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch price data via yfinance")
    p_fetch.add_argument("--ticker", required=True)
    p_fetch.add_argument("--period", default="1y")
    p_fetch.add_argument("--interval", default="1d")
    p_fetch.add_argument("--output", type=Path, required=True)

    p_features = sub.add_parser("features", help="Build features from price data")
    p_features.add_argument("--input", type=Path, required=True)
    p_features.add_argument("--output", type=Path, required=True)
    p_features.add_argument("--window-short", type=int, default=5)
    p_features.add_argument("--window-long", type=int, default=20)

    p_train = sub.add_parser("train", help="Train a model from a features file")
    p_train.add_argument("--features", type=Path, required=True)
    p_train.add_argument("--model", type=Path, required=True)
    p_train.add_argument("--model-type", default="lightgbm", choices=["sklearn", "xgboost", "lightgbm"])
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--random-state", type=int, default=42)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model on a features file")
    p_eval.add_argument("--features", type=Path, required=True)
    p_eval.add_argument("--model", type=Path, required=True)

    p_pred = sub.add_parser("predict", help="Predict using a trained model")
    p_pred.add_argument("--features", type=Path, required=True)
    p_pred.add_argument("--model", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        from .fetch import FetchConfig, fetch_prices

        fetch_prices(
            FetchConfig(
                ticker=args.ticker,
                period=args.period,
                interval=args.interval,
                output_path=args.output,
            )
        )
        return 0

    if args.command == "features":
        from .features import FeatureConfig, build_features

        build_features(
            FeatureConfig(
                input_path=args.input,
                output_path=args.output,
                window_short=args.window_short,
                window_long=args.window_long,
            )
        )
        return 0

    if args.command == "train":
        from .train import TrainCLIConfig, train_from_file

        auc = train_from_file(
            TrainCLIConfig(
                features_path=args.features,
                model_path=args.model,
                model_type=args.model_type,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        )
        print(f"Validation AUC: {auc:.4f}")
        return 0

    if args.command == "evaluate":
        from .evaluate import EvaluateCLIConfig, evaluate_from_file

        result = evaluate_from_file(
            EvaluateCLIConfig(features_path=args.features, model_path=args.model)
        )
        print(f"AUC: {result['auc']:.4f}")
        print(result["report"])
        return 0

    if args.command == "predict":
        import pandas as pd
        import joblib

        df = pd.read_parquet(args.features)
        # simple feature selection like in training
        from .beat_predictor import prepare_features_targets

        X, _ = prepare_features_targets(df)
        bundle = joblib.load(args.model)
        model = bundle["model"]
        try:
            proba = model.predict_proba(X)[:, 1]
            print(proba)
        except Exception:
            pred = model.predict(X)
            print(pred)
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


