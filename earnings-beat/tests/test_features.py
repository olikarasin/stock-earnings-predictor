import pandas as pd
from pathlib import Path
import numpy as np


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


