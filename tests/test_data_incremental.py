"""Tests for incremental OHLCV download (append only-new rows to cache)."""

import numpy as np
import pandas as pd

from calibmiss.data import download_incremental


def _frame(dates_start: str, n_days: int, tickers: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range(dates_start, periods=n_days)
    frames = []
    rng = np.random.default_rng(0)
    for t in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        frames.append(
            pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 9_000_000, n_days).astype(float),
                },
                index=dates,
            ).assign(ticker=t)
        )
    df = pd.concat(frames)
    df = df.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["ticker", "date"]).sort_index()


def test_incremental_merges_tail_without_duplicates(tmp_path, monkeypatch) -> None:
    """With a cached sub-range, the next call fetches only the tail and merges."""
    tickers = ["AAA", "BBB"]
    full = _frame("2023-01-02", 400, tickers)
    calls: list[tuple] = []

    def fake_download(tickers_arg, start=None, end=None, **kwargs):
        calls.append((start, end))
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        return full.loc[
            (full.index.get_level_values("date") >= start_dt)
            & (full.index.get_level_values("date") <= end_dt)
        ]

    # First call caches only the early sub-range (end date bounds the result).
    first = download_incremental(
        tickers, "2023-01-02", "2023-06-30", cache_dir=tmp_path, fetch=fake_download
    )
    first_expected = full.loc[full.index.get_level_values("date") <= pd.Timestamp("2023-06-30")]
    assert len(first) == len(first_expected)

    # Second call extends the end: it must fetch just the missing tail and merge
    # it in without duplicates.
    second = download_incremental(
        tickers, "2023-01-02", "2024-04-30", cache_dir=tmp_path, fetch=fake_download
    )
    second_expected = full.loc[full.index.get_level_values("date") <= pd.Timestamp("2024-04-30")]
    assert len(second) == len(second_expected)
    assert not second.index.duplicated().any()
    # The incremental call requested a narrow tail window, not full history.
    assert calls[-1][0] > "2023-01-02"
    pd.testing.assert_frame_equal(second, second_expected)


def test_incremental_falls_back_to_full_when_no_cache(tmp_path, monkeypatch) -> None:
    """Without a cache, a full-range download is performed."""
    full = _frame("2023-01-02", 200, ["CCC"])
    seen: list[dict] = []

    def fake_download(tickers_arg, start=None, end=None, **kwargs):
        seen.append({"start": start, "end": end})
        return full

    out = download_incremental(
        ["CCC"], "2023-01-02", "2023-12-29", cache_dir=tmp_path, fetch=fake_download
    )
    assert len(out) == len(full)
    assert seen[0]["start"] == "2023-01-02"


def test_incremental_rejects_bad_data(tmp_path, monkeypatch) -> None:
    def bad_download(tickers_arg, start=None, end=None, **kwargs):
        return pd.DataFrame()

    try:
        download_incremental(
            ["AAA"], "2023-01-02", "2023-12-29", cache_dir=tmp_path, fetch=bad_download
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
