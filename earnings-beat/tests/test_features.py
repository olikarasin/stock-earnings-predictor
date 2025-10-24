import pandas as pd
from pathlib import Path
import numpy as np
import pytest
import socket


def _has_network(host: str = "query1.finance.yahoo.com", port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def test_import_features_module():
    import src.features as features  # noqa: F401


def test_build_features_from_dataframe_roundtrip(tmp_path: Path):
    from src.features import FeatureConfig, build_features

    dates = pd.date_range("2022-01-01", periods=30, freq="D")
    close = pd.Series(np.linspace(100, 110, 30))
    df = pd.DataFrame({"Date": dates, "Close": close})
    input_path = tmp_path / "prices.parquet"
    df.to_parquet(input_path)

    output_path = tmp_path / "features.parquet"
    cfg = FeatureConfig(input_path=input_path, output_path=output_path)
    out = build_features(cfg)

    assert {"return_1d", "ma_short", "ma_long", "ma_ratio", "target"}.issubset(out.columns)
    assert output_path.exists()


@pytest.mark.network
def test_build_features_inference():
    if not _has_network():
        pytest.skip("offline: skipping network test")
    from src.features import build_features_for_inference, get_feature_columns

    X, notes = build_features_for_inference("AAPL")
    assert X.shape[0] == 1
    required = set(get_feature_columns())
    assert required.issubset(set(X.columns))


