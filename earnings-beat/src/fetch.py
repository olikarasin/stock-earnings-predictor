from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FetchConfig:
    ticker: str
    period: str = "1y"
    interval: str = "1d"
    output_path: Optional[Path] = None


def fetch_prices(config: FetchConfig):
    """Fetch historical price data using yfinance and return a DataFrame.

    Import of heavy third-party libraries happens inside the function to keep
    module import light and avoid side effects during test discovery.
    """
    import yfinance as yf  # local import to keep top-level imports clean
    import pandas as pd

    ticker = yf.Ticker(config.ticker)
    df: pd.DataFrame = ticker.history(period=config.period, interval=config.interval)
    if df.empty:
        raise ValueError(f"No data returned for {config.ticker}")
    df = df.reset_index()

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
    return df


