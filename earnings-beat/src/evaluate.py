from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


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


# -----------------------------
# Evaluation helpers and plotting
# -----------------------------

def precision_at_k(y_true, y_prob, k: float = 0.1) -> float:
    """Return precision among top-k fraction by predicted probability.

    k is a fraction in (0,1]; when len(y_true)*k < 1, selects at least 1.
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = float(k)
    if k <= 0:
        k = 0.1
    top_n = max(1, int(np.ceil(n * k)))
    order = np.argsort(-y_prob)
    idx = order[:top_n]
    return float(np.mean(y_true[idx]))


def calibration_report(y_true, y_prob, n_bins: int = 10) -> Dict[str, object]:
    """Return a simple calibration summary with per-bin mean prob and accuracy.

    Returns dict with keys:
    - bin_means: list of mean predicted probabilities in each bin
    - bin_accs: list of empirical positive rates in each bin
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bins: 1..n_bins via digitize on right-open intervals except last bin closed
    # Use right=False so [0, e1) -> bin 1, ..., [e_{n-1}, 1] -> bin n
    bins = np.digitize(y_prob, edges[1:-1], right=False)
    # bins in 0..n_bins-1
    bin_means = []
    bin_accs = []
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            bin_means.append(float("nan"))
            bin_accs.append(float("nan"))
            continue
        bin_means.append(float(np.mean(y_prob[mask])))
        bin_accs.append(float(np.mean(y_true[mask])))
    return {"bin_means": bin_means, "bin_accs": bin_accs}


def plot_roc_curve(y_true, y_prob, out_path: Path) -> None:
    """Save a ROC curve plot to out_path using matplotlib."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reliability_diagram(y_true, y_prob, out_path: Path, n_bins: int = 10) -> None:
    """Save a reliability (calibration) plot to out_path using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    report = calibration_report(y_true, y_prob, n_bins=n_bins)
    x = np.array(report["bin_means"], dtype=float)
    y = np.array(report["bin_accs"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    # Only plot points that are not NaN
    m = ~(np.isnan(x) | np.isnan(y))
    ax.plot(x[m], y[m], marker="o")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_default_predictions(models_dir: Path) -> tuple[object, object] | None:
    """Load y_true.npy and y_prob.npy from models_dir if present."""
    import numpy as np

    y_true_path = models_dir / "y_true.npy"
    y_prob_path = models_dir / "y_prob.npy"
    if y_true_path.exists() and y_prob_path.exists():
        try:
            y_true = np.load(y_true_path)
            y_prob = np.load(y_prob_path)
            return y_true, y_prob
        except Exception:
            return None
    return None


def _main_plots_from_npy() -> int:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    maybe = _load_default_predictions(models_dir)
    if not maybe:
        return 0
    y_true, y_prob = maybe
    try:
        plot_roc_curve(y_true, y_prob, models_dir / "roc_curve.png")
        plot_reliability_diagram(y_true, y_prob, models_dir / "reliability_diagram.png")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(_main_plots_from_npy())
