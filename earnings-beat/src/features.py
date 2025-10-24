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


def build_features_for_inference(ticker: str) -> tuple["pd.DataFrame", dict]:
    """Build a single-row feature vector for a ticker using only information up to now.

    Features produced (columns):
    - sales_ttm_growth
    - sales_qoq_q1, sales_qoq_q2
    - earnings_ttm_growth
    - earnings_qoq_q1, earnings_qoq_q2
    - eps_surprise_mean_4, eps_surprise_std_4, eps_surprise_last
    - eps_revision_up_1q, eps_revision_down_1q, eps_revision_ratio_1q
    - pt_up_90d, pt_down_90d, pt_net_90d, pt_avg_delta_90d
    - return_1m, return_3m
    - vol_20d
    - beta

    Returns a (X, notes) tuple where X is a 1-row DataFrame and notes is a dict
    with keys like 'upcoming_earnings_date' and 'data_warnings'. Some features
    may be NaN if unavailable.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, timezone

    from .fetch import (
        get_quarterly_financials,
        get_earnings_history,
        get_earnings_trend,
        get_recommendations,
        get_price_history,
        get_beta_estimate,
    )

    warnings: list[str] = []

    # Helper: safe division
    def sdiv(num, den):
        try:
            den = float(den)
            if den == 0 or np.isnan(den):
                return np.nan
            return float(num) / den
        except Exception:
            return np.nan

    # 1) Quarterly financials: Total Revenue and Net Income
    qf = get_quarterly_financials(ticker)
    sales_ttm_growth = np.nan
    earnings_ttm_growth = np.nan
    sales_qoq_q1 = np.nan
    sales_qoq_q2 = np.nan
    earnings_qoq_q1 = np.nan
    earnings_qoq_q2 = np.nan
    try:
        if qf is not None and not qf.empty:
            # Rows are items, columns are quarter-end timestamps
            # Normalize item names
            def find_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
                lower_index = {str(i).lower(): i for i in df.index}
                for c in candidates:
                    if c.lower() in lower_index:
                        return df.loc[lower_index[c.lower()]]
                return None

            sales_row = find_row(qf, ["Total Revenue", "Total revenue", "total revenue", "Revenue"])
            earn_row = find_row(qf, ["Net Income", "Net income", "net income", "NetIncome"])

            def compute_ttm_growth(series: pd.Series) -> float:
                s = pd.to_numeric(series, errors="coerce").dropna()
                # Ensure columns chronological: use the order in series
                values = s.values.astype(float)
                if values.size < 8:
                    return np.nan
                recent = values[-4:].sum()
                prior = values[-8:-4].sum()
                return sdiv(recent - prior, abs(prior))

            def compute_qoq_last2(series: pd.Series) -> tuple[float, float]:
                s = pd.to_numeric(series, errors="coerce").dropna()
                values = s.values.astype(float)
                if values.size < 3:
                    return (np.nan, np.nan)
                # QoQ for latest vs prev, and prev vs prev2
                q1 = sdiv(values[-1] - values[-2], abs(values[-2]))
                q2 = sdiv(values[-2] - values[-3], abs(values[-3]))
                return (q1, q2)

            if sales_row is not None:
                sales_ttm_growth = compute_ttm_growth(sales_row)
                sales_qoq_q1, sales_qoq_q2 = compute_qoq_last2(sales_row)
            else:
                warnings.append("quarterly financials: missing Total Revenue")
            if earn_row is not None:
                earnings_ttm_growth = compute_ttm_growth(earn_row)
                earnings_qoq_q1, earnings_qoq_q2 = compute_qoq_last2(earn_row)
            else:
                warnings.append("quarterly financials: missing Net Income")
        else:
            warnings.append("quarterly financials unavailable")
    except Exception:
        warnings.append("error processing quarterly financials")

    # 2) Earnings history: compute EPS surprises
    eps_surprise_mean_4 = np.nan
    eps_surprise_std_4 = np.nan
    eps_surprise_last = np.nan
    try:
        eh = get_earnings_history(ticker)
        if eh is not None and not eh.empty:
            df_e = eh.copy()
            # Try to identify actual and estimate columns
            cols = {c.lower(): c for c in df_e.columns}
            actual_col = cols.get("epsactual") or cols.get("actual") or cols.get("eps_actual")
            est_col = cols.get("epsestimate") or cols.get("estimate") or cols.get("eps_estimate")
            # Identify ordering column
            date_col = None
            for k in ["startdatetime", "reportdate", "date", "period", "quarter"]:
                if k in cols:
                    date_col = cols[k]
                    break
            if actual_col and est_col:
                df_e = df_e[[actual_col, est_col] + ([date_col] if date_col else [])].copy()
                df_e["_surprise"] = (pd.to_numeric(df_e[actual_col], errors="coerce") - pd.to_numeric(df_e[est_col], errors="coerce")) / (
                    pd.to_numeric(df_e[est_col], errors="coerce").abs()
                )
                if date_col:
                    try:
                        df_e = df_e.sort_values(date_col)
                    except Exception:
                        pass
                surp = df_e["_surprise"].dropna().values
                if surp.size > 0:
                    eps_surprise_last = float(surp[-1])
                if surp.size >= 1:
                    last4 = surp[-4:]
                    eps_surprise_mean_4 = float(np.mean(last4)) if last4.size > 0 else np.nan
                    eps_surprise_std_4 = float(np.std(last4, ddof=0)) if last4.size > 1 else np.nan
            else:
                warnings.append("earnings history missing actual/estimate EPS columns")
        else:
            warnings.append("earnings history unavailable")
    except Exception:
        warnings.append("error processing earnings history")

    # 3) Earnings trend: EPS revisions for next quarter
    eps_revision_up_1q = np.nan
    eps_revision_down_1q = np.nan
    eps_revision_ratio_1q = np.nan
    try:
        et = get_earnings_trend(ticker) or {}
        # Structure may be { 'earningsTrend': { 'trend': [ {...} ] } } or { 'trend': [ {...} ] }
        trend_list = None
        if isinstance(et, dict):
            if "earningsTrend" in et and isinstance(et["earningsTrend"], dict):
                trend_list = et["earningsTrend"].get("trend")
            if trend_list is None:
                trend_list = et.get("trend")
        if isinstance(trend_list, list):
            # Find next quarter
            target = None
            for item in trend_list:
                period = (item or {}).get("period")
                if period in {"+1q", "0q", "currentQuarter"}:
                    target = item
                    break
            if target and isinstance(target, dict):
                rev = target.get("epsRevisions") or {}
                up = 0
                down = 0
                for k, v in (rev or {}).items():
                    if not isinstance(v, (int, float)):
                        continue
                    if "up" in k.lower():
                        up += int(v)
                    if "down" in k.lower():
                        down += int(v)
                eps_revision_up_1q = float(up)
                eps_revision_down_1q = float(down)
                denom = up + down
                eps_revision_ratio_1q = (up / denom) if denom > 0 else np.nan
            else:
                warnings.append("earnings trend missing +1q period")
        else:
            warnings.append("earnings trend unavailable")
    except Exception:
        warnings.append("error processing earnings trend")

    # 4) Recommendations: price target changes in last 90 days
    pt_up_90d = 0.0
    pt_down_90d = 0.0
    pt_net_90d = 0.0
    pt_avg_delta_90d = np.nan
    try:
        rec = get_recommendations(ticker)
        if rec is not None and not rec.empty:
            df_r = rec.copy()
            # Normalize date
            if "Date" in df_r.columns:
                dt = pd.to_datetime(df_r["Date"], errors="coerce")
            else:
                # Try index
                try:
                    dt = pd.to_datetime(df_r.index, errors="coerce")
                except Exception:
                    dt = pd.Series([pd.NaT] * len(df_r))
            df_r["_date"] = dt
            cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=90))
            recent = df_r[df_r["_date"] >= cutoff]
            if not recent.empty:
                # Action analysis
                action_col = None
                for c in ["Action", "action", "Type", "type"]:
                    if c in recent.columns:
                        action_col = c
                        break
                up_count = 0
                down_count = 0
                if action_col:
                    acts = recent[action_col].astype(str).str.lower()
                    # price target mentions
                    mask_pt = acts.str.contains("price target") | acts.str.contains(r"\bpt\b")
                    if mask_pt.any():
                        up_words = ["raise", "raised", "increase", "increased", "hike", "boost", "up"]
                        down_words = ["cut", "lower", "lowered", "decrease", "reduced", "down"]
                        up_count = int(acts[mask_pt].apply(lambda s: any(w in s for w in up_words)).sum())
                        down_count = int(acts[mask_pt].apply(lambda s: any(w in s for w in down_words)).sum())
                        # Try to compute average delta if numeric columns exist
                        num_cols = [
                            ("newPriceTarget", "oldPriceTarget"),
                            ("Price Target New", "Price Target Old"),
                            ("new_pt", "old_pt"),
                            ("newTarget", "oldTarget"),
                        ]
                        deltas = None
                        for a, b in num_cols:
                            if a in recent.columns and b in recent.columns:
                                a_num = pd.to_numeric(recent[a], errors="coerce")
                                b_num = pd.to_numeric(recent[b], errors="coerce")
                                dd = (a_num - b_num).dropna()
                                if not dd.empty:
                                    deltas = dd
                                    break
                        if deltas is not None and not deltas.empty:
                            pt_avg_delta_90d = float(deltas.mean())
                        else:
                            pt_avg_delta_90d = np.nan
                    else:
                        # Fallback: use upgrades/downgrades as proxy; avg delta not measurable
                        up_count = int(acts.str.contains("upgrade").sum())
                        down_count = int(acts.str.contains("downgrade").sum())
                        pt_avg_delta_90d = np.nan
                pt_up_90d = float(up_count)
                pt_down_90d = float(down_count)
                pt_net_90d = float(up_count - down_count)
            else:
                # No recent records
                pt_up_90d = 0.0
                pt_down_90d = 0.0
                pt_net_90d = 0.0
                pt_avg_delta_90d = np.nan
        else:
            warnings.append("recommendations unavailable")
    except Exception:
        warnings.append("error processing recommendations")

    # 5) Daily prices: 1m/3m returns and 20d realized vol
    return_1m = np.nan
    return_3m = np.nan
    vol_20d = np.nan
    try:
        px = get_price_history(ticker, period="6mo", interval="1d")
        if px is not None and not px.empty and "Close" in px.columns:
            s = pd.to_numeric(px["Close"], errors="coerce").dropna()
            if s.size >= 2:
                # daily returns
                rets = s.pct_change().dropna()
                if rets.size >= 20:
                    vol_20d = float(rets[-20:].std())
            # 21 trading days ~ 1m; 63 ~ 3m
            def lookback_ret(series: pd.Series, n: int) -> float:
                if series.size > n:
                    a = float(series.iloc[-1])
                    b = float(series.iloc[-1 - n])
                    if b != 0:
                        return (a / b) - 1.0
                return np.nan

            return_1m = lookback_ret(s, 21)
            return_3m = lookback_ret(s, 63)
        else:
            warnings.append("price history unavailable for returns/vol")
    except Exception:
        warnings.append("error processing price history")

    # 6) Beta
    beta = np.nan
    try:
        b = get_beta_estimate(ticker)
        beta = float(b) if b is not None else np.nan
    except Exception:
        warnings.append("error computing beta")

    # Upcoming earnings date (best-effort)
    upcoming_earnings_date = None
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        # Try calendar first
        cal = None
        try:
            cal = tk.calendar
        except Exception:
            cal = None
        if cal is not None and hasattr(cal, "index") and not cal.empty:
            for idx in cal.index:
                if str(idx).lower().startswith("earnings date"):
                    val = cal.loc[idx].iloc[0]
                    try:
                        upcoming_earnings_date = pd.to_datetime(val).isoformat()
                        break
                    except Exception:
                        pass
        if upcoming_earnings_date is None:
            # Try get_earnings_dates with limit 1 and upcoming=True if supported
            try:
                ed = tk.get_earnings_dates(limit=1)
                if ed is not None and not ed.empty:
                    dtcol = None
                    for c in ed.columns:
                        if "earnings date" in c.lower() or "reporteddate" in c.lower() or "date" in c.lower():
                            dtcol = c
                            break
                    if dtcol:
                        val = ed.iloc[0][dtcol]
                        upcoming_earnings_date = pd.to_datetime(val).isoformat()
            except Exception:
                pass
    except Exception:
        # ignore failure
        pass

    data = {
        "sales_ttm_growth": sales_ttm_growth,
        "sales_qoq_q1": sales_qoq_q1,
        "sales_qoq_q2": sales_qoq_q2,
        "earnings_ttm_growth": earnings_ttm_growth,
        "earnings_qoq_q1": earnings_qoq_q1,
        "earnings_qoq_q2": earnings_qoq_q2,
        "eps_surprise_mean_4": eps_surprise_mean_4,
        "eps_surprise_std_4": eps_surprise_std_4,
        "eps_surprise_last": eps_surprise_last,
        "eps_revision_up_1q": eps_revision_up_1q,
        "eps_revision_down_1q": eps_revision_down_1q,
        "eps_revision_ratio_1q": eps_revision_ratio_1q,
        "pt_up_90d": pt_up_90d,
        "pt_down_90d": pt_down_90d,
        "pt_net_90d": pt_net_90d,
        "pt_avg_delta_90d": pt_avg_delta_90d,
        "return_1m": return_1m,
        "return_3m": return_3m,
        "vol_20d": vol_20d,
        "beta": beta,
    }

    X = pd.DataFrame([data])
    notes = {"upcoming_earnings_date": upcoming_earnings_date, "data_warnings": warnings}
    return X, notes


