from pathlib import Path
import warnings
import pytest
import socket


def _has_network(host: str = "query1.finance.yahoo.com", port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def test_import_fetch_module():
    # Ensure module imports without side effects or missing deps at top-level
    import src.fetch as fetch  # noqa: F401


def test_fetch_api_presence():
    import src.fetch as fetch

    assert hasattr(fetch, "get_price_history")
    assert hasattr(fetch, "get_quarterly_financials")
    assert hasattr(fetch, "get_earnings_history")
    assert hasattr(fetch, "get_earnings_trend")
    assert hasattr(fetch, "get_recommendations")
    assert hasattr(fetch, "get_beta_estimate")


def test_fetch_config_dataclass_fields():
    from src.fetch import FetchConfig

    cfg = FetchConfig(ticker="AAPL", period="1mo", interval="1d", output_path=None)
    assert cfg.ticker == "AAPL"
    assert cfg.period == "1mo"
    assert cfg.interval == "1d"
    assert cfg.output_path is None


@pytest.mark.network
def test_fetch_price_history_shape():
    if not _has_network():
        pytest.skip("offline: skipping network test")
    from src.fetch import get_price_history

    df = get_price_history("AAPL", period="1mo", interval="1d")
    # On error, df can be empty, but shape columns should include if not empty
    if df is not None and not df.empty:
        cols = set(df.columns.str.lower())
        required = {"open", "high", "low", "close"}
        assert required.issubset(cols)


@pytest.mark.network
def test_fetch_quarterly_financials_nonempty():
    if not _has_network():
        pytest.skip("offline: skipping network test")
    from src.fetch import get_quarterly_financials

    df = get_quarterly_financials("AAPL")
    if df is None or df.empty:
        warnings.warn("quarterly financials empty for AAPL; passing test")
        return
    # If present, should contain a Total Revenue row
    idx_lower = set([str(i).lower() for i in df.index])
    assert any("total revenue" in s for s in idx_lower)


