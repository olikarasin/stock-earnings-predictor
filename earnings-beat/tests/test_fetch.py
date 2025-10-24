from pathlib import Path


def test_import_fetch_module():
    # Ensure module imports without side effects or missing deps at top-level
    import src.fetch as fetch  # noqa: F401


def test_fetch_config_dataclass_fields():
    from src.fetch import FetchConfig

    cfg = FetchConfig(ticker="AAPL", period="1mo", interval="1d", output_path=None)
    assert cfg.ticker == "AAPL"
    assert cfg.period == "1mo"
    assert cfg.interval == "1d"
    assert cfg.output_path is None


