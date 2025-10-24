from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import time
import json


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_TTL_HOURS = 24


def _is_fresh(path: Path, ttl_hours: int = DEFAULT_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= ttl_hours * 3600


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cache_path_for_prices(ticker: str, period: str, interval: str) -> Path:
    safe = ticker.upper().replace("/", "-")
    return RAW_DATA_DIR / f"{safe}_prices_{period}_{interval}.parquet"


def _cache_path_for_suffix(ticker: str, suffix: str, ext: str = "parquet") -> Path:
    safe = ticker.upper().replace("/", "-")
    return RAW_DATA_DIR / f"{safe}_{suffix}.{ext}"


@dataclass
class FetchConfig:
    ticker: str
    period: str = "1y"
    interval: str = "1d"
    output_path: Optional[Path] = None


def fetch_prices(config: FetchConfig):
    """Backward-compatible wrapper used by CLI to fetch price history.

    Delegates to get_price_history and optionally writes to the given output path.
    """
    df = get_price_history(config.ticker, period=config.period, interval=config.interval)
    if config.output_path is not None:
        _ensure_dir(Path(config.output_path))
        try:
            # write as parquet; caller controls extension by path
            import pandas as pd  # noqa: F401 - type only

            df.to_parquet(Path(config.output_path))
        except Exception:
            # best-effort write; ignore failures
            pass
    return df


def get_price_history(ticker: str, period: str = "3y", interval: str = "1d") -> "pd.DataFrame":
    """Return historical prices for a ticker using yfinance.

    Results are cached under data/raw as parquet with a 24h TTL.
    Columns typically include Date (index), Open, High, Low, Close, Volume, etc.
    On missing data or errors, returns an empty DataFrame.
    """
    import pandas as pd
    try:
        cache_path = _cache_path_for_prices(ticker, period, interval)
        if _is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        import yfinance as yf

        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        _ensure_dir(cache_path)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


def get_quarterly_financials(ticker: str) -> "pd.DataFrame":
    """Return quarterly financials (rows are line items, columns are quarter-end dates).

    Cached to parquet for 24h. Returns an empty DataFrame on errors or unavailability.
    """
    import pandas as pd
    cache_path = _cache_path_for_suffix(ticker, "quarterly_financials")
    try:
        if _is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        import yfinance as yf

        qf = yf.Ticker(ticker).quarterly_financials
        if qf is None:
            return pd.DataFrame()
        df = qf.copy()
        # yfinance returns items as index already; ensure proper type
        _ensure_dir(cache_path)
        try:
            df.to_parquet(cache_path)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


def get_earnings_history(ticker: str) -> "pd.DataFrame":
    """Return past earnings events with actual EPS and estimate EPS as a DataFrame.

    Cached to parquet for 24h. Returns an empty DataFrame on errors or missing data.
    """
    import pandas as pd
    cache_path = _cache_path_for_suffix(ticker, "earnings_history")
    try:
        if _is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        import yfinance as yf

        tk = yf.Ticker(ticker)
        data = None
        # yfinance exposes get_earnings_history() on recent versions
        try:
            data = tk.get_earnings_history()
        except Exception:
            data = None
        df = pd.DataFrame(data) if data else pd.DataFrame()
        # Fallback: get_earnings_dates has columns like 'Earnings Date','Reported EPS','EPS Estimate'
        if df is None or df.empty:
            try:
                ed = tk.get_earnings_dates(limit=1000)
                if ed is not None and not ed.empty:
                    df = ed.reset_index(drop=True)
            except Exception:
                df = pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        _ensure_dir(cache_path)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


def get_earnings_trend(ticker: str) -> Dict:
    """Return the earningsTrend JSON payload if available.

    Returns an empty dict on errors or missing data. Cached to JSON for 24h.
    """
    cache_path = _cache_path_for_suffix(ticker, "earnings_trend", ext="json")
    try:
        if _is_fresh(cache_path):
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                return {}

        import yfinance as yf

        tk = yf.Ticker(ticker)
        payload = {}
        try:
            payload = tk.get_earnings_trend() or {}
        except Exception:
            payload = {}
        _ensure_dir(cache_path)
        try:
            cache_path.write_text(json.dumps(payload))
        except Exception:
            pass
        return payload or {}
    except Exception:
        return {}


def get_recommendations(ticker: str) -> "pd.DataFrame":
    """Return analyst recommendation actions table from yfinance.

    Cached to parquet for 24h. Returns an empty DataFrame on errors or unavailability.
    """
    import pandas as pd
    cache_path = _cache_path_for_suffix(ticker, "recommendations")
    try:
        if _is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        import yfinance as yf

        rec = yf.Ticker(ticker).recommendations
        if rec is None:
            return pd.DataFrame()
        df = rec.reset_index() if hasattr(rec, "reset_index") else rec
        _ensure_dir(cache_path)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


def get_beta_estimate(ticker: str) -> float | None:
    """Return beta if available from ticker info; otherwise estimate via SPY daily returns.

    Estimation method: beta = Cov(r_ticker, r_spy) / Var(r_spy) using daily Close returns
    over the available cached price histories. Returns None on errors.
    """
    try:
        import yfinance as yf

        info_beta = None
        try:
            info = yf.Ticker(ticker).info or {}
            info_beta = info.get("beta")
        except Exception:
            info_beta = None
        if isinstance(info_beta, (int, float)):
            return float(info_beta)

        # Fallback: compute from returns using cached price data
        import pandas as pd
        import numpy as np

        df_t = get_price_history(ticker, period="3y", interval="1d")
        df_s = get_price_history("SPY", period="3y", interval="1d")
        if df_t is None or df_s is None or df_t.empty or df_s.empty:
            return None
        # Expect Date column present after reset_index
        if "Date" not in df_t.columns:
            return None
        if "Date" not in df_s.columns:
            return None
        t = df_t[["Date", "Close"]].dropna()
        s = df_s[["Date", "Close"]].dropna()
        t["Date"] = pd.to_datetime(t["Date"]).dt.tz_localize(None)
        s["Date"] = pd.to_datetime(s["Date"]).dt.tz_localize(None)
        t = t.set_index("Date").sort_index()
        s = s.set_index("Date").sort_index()
        rets = pd.DataFrame({
            "t": t["Close"].pct_change(),
            "s": s["Close"].pct_change(),
        }).dropna()
        if rets.empty or rets["s"].var() == 0:
            return None
        cov = float(rets.cov().loc["t", "s"])
        var_s = float(rets["s"].var())
        beta = cov / var_s if var_s != 0 else None
        return float(beta) if beta is not None else None
    except Exception:
        return None



