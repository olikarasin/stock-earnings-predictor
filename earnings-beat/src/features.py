from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FeatureConfig:
    input_path: Path
    output_path: Optional[Path] = None
    window_short: int = 5
    window_long: int = 20


def build_features(config: FeatureConfig):
    """Create simple rolling-window features from a prices parquet file.

    This expects columns including "Date" (or index), "Close", and will
    produce moving averages and daily returns. Heavy imports are internal.
    """
    import pandas as pd
    import numpy as np

    df: pd.DataFrame = pd.read_parquet(config.input_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        df = df.sort_index().reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    if "Close" not in df.columns:
        raise ValueError("Input data must contain a 'Close' column")

    df["return_1d"] = df["Close"].pct_change()
    df["ma_short"] = df["Close"].rolling(window=config.window_short, min_periods=1).mean()
    df["ma_long"] = df["Close"].rolling(window=config.window_long, min_periods=1).mean()
    df["ma_ratio"] = np.where(df["ma_long"] != 0, df["ma_short"] / df["ma_long"], 0.0)

    # Placeholder simple target: future 5-day return > 0 indicates "beat" proxy
    df["target"] = df["Close"].pct_change(periods=5).shift(-5)
    df["target"] = (df["target"] > 0).astype(int)

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    return df


