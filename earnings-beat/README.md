## earnings-beat

Earnings-beat is a lightweight, modular starter project for building and evaluating models that predict whether a company will "beat" earnings expectations. It provides minimal data fetching via Yahoo Finance, simple feature generation from price history, and a pluggable model interface that supports common libraries. This template is designed for experimentation and extension, not production trading systems.

### Quickstart
1) Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Fetch price data:
```bash
python -m src.cli fetch --ticker AAPL --period 2y --interval 1d --output data/raw/AAPL_prices.parquet
```

3) Generate features from prices:
```bash
python -m src.cli features --input data/raw/AAPL_prices.parquet --output data/features/AAPL_features.parquet
```

4) Train a model (expects a column named "target" in the features file):
```bash
python -m src.cli train --features data/features/AAPL_features.parquet --model models/beat_predictor.joblib --model-type lightgbm
```

5) Evaluate a trained model:
```bash
python -m src.cli evaluate --features data/features/AAPL_features.parquet --model models/beat_predictor.joblib
```

6) Predict on a features file:
```bash
python -m src.cli predict --features data/features/AAPL_features.parquet --model models/beat_predictor.joblib
```

7) Full training pipeline and single-ticker inference:
```bash
python -m src.train --tickers-file tickers.txt --years 8
python -m src.beat_predictor --ticker AAPL
```

### Metrics
- AUC: Area Under the ROC Curve; threshold-agnostic ranking quality of positive vs negative events.
- Brier: Mean squared error between predicted probability and actual outcome (0/1). Lower is better.
- Precision@Top10pct: Precision computed on the top 10% highest-probability predictions.

### Data limitations
- Yahoo Finance data can have gaps, revisions, and schema changes. Network requests may fail or rate-limit.
- Financial statement mapping may vary by ticker (e.g., revenue line names). The code attempts robust fallbacks.
- Recommendations feeds can be sparse and textual; the price-target detection is heuristic.

### Test matrix (example)
This repo is compatible with a simple GitHub Actions-like matrix (not included here):
```
python-version: ["3.10", "3.11", "3.12"]
os: [ubuntu-latest, macos-latest]
steps:
  - run: pip install -r requirements.txt pytest
  - run: pytest -m "not network"  # fast unit tests without network
  - run: pytest -m network --maxfail=1 -q || true  # allow flaky network tests
```

### Disclaimer
This project is for educational and research purposes only. It is not an investment product. Past performance does not guarantee future results. Nothing in this repository constitutes financial advice. Use at your own risk.


