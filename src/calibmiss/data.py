"""Data download, caching, and validation for CalibMiss."""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass
class ValidationReport:
    """Report from data validation."""

    is_valid: bool
    errors: list[str]
    native_missing_rates: dict[str, float]


def _get_cache_key(tickers: list[str], start: str, end: str) -> str:
    """Generate a cache key from tickers and date range."""
    tickers_str = "_".join(sorted(tickers))
    return f"{tickers_str}_{start}_{end}"


def _compute_sha256(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of DataFrame serialized to parquet."""
    import io

    buffer = io.BytesIO()
    df.to_parquet(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def download_ohlcv(
    tickers: list[str], start: str, end: str, cache_dir: str | Path = "data/raw"
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance with caching.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Directory for cached parquet files

    Returns:
        DataFrame with MultiIndex (ticker, date) and columns:
        open, high, low, close, adj_close, volume
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _get_cache_key(tickers, start, end)

    # Check for existing cached file
    existing_files = list(cache_dir.glob(f"*_{cache_key}_sha256_*.parquet"))
    for cache_path in existing_files:
        try:
            df = pd.read_parquet(cache_path)
            expected_sha256 = cache_path.stem.rsplit("_sha256_", 1)[1]
            if _compute_sha256(df) == expected_sha256 and validate_ohlcv(df).is_valid:
                return df
        except (OSError, ValueError, KeyError, IndexError):
            continue

    # Download from Yahoo Finance
    data = _download_fetch(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data is None or data.empty:
        raise ValueError("OHLCV validation failed: OHLCV data is empty")

    df = _from_fetch(data, tickers, validate=True)

    # Save to cache with timestamp and SHA256
    cache_name = _cache_filename(df, cache_key)
    df.to_parquet(cache_dir / cache_name)

    return df


def _usable_cache(
    tickers: list[str], start: str, cache_dir: Path
) -> tuple[pd.DataFrame | None, Path | None]:
    """Return (frame, path) of a cached download covering requested ``start``.

    Cache filenames embed their original ``start``/``end``, so a request with a
    different ``end`` must still reuse a cache that covers it. Work by content:
    any valid cached frame that holds every requested ticker and starts no
    later than ``start`` is usable; the caller fetches the tail to ``end``.
    """
    if not cache_dir.is_dir():
        return None, None
    for path in sorted(cache_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
            if _compute_sha256(df) != path.stem.rsplit("_sha256_", 1)[1]:
                continue
            if not validate_ohlcv(df).is_valid:
                continue
        except (OSError, ValueError, KeyError, IndexError):
            continue
        present = set(df.index.get_level_values("ticker"))
        if not set(tickers) <= present:
            continue
        if df.index.get_level_values("date").min() <= pd.to_datetime(start):
            return df, path
    return None, None


def _download_fetch(
    tickers: list[str],
    start: str,
    end: str,
    group_by: str,
    auto_adjust: bool,
    progress: bool,
    threads: bool,
) -> pd.DataFrame:
    """Thin wrapper around ``yf.download`` with the canonical signature."""
    return yf.download(
        tickers,
        start=start,
        end=end,
        group_by=group_by,
        auto_adjust=auto_adjust,
        progress=progress,
        threads=threads,
    )


def download_incremental(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw",
    fetch: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Download OHLCV, appending only rows newer than an existing cache.

    A live system must not re-download a decade of history daily. If a valid
    cached frame covers `tickers` within `start`..`end`, this fetches only the
    tail window (cache-last-date, end) and merges it in, saving a new
    SHA256-named cache file and removing the old one. Without a usable cache it
    falls back to a full-range download.

    Args:
        tickers: Ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        cache_dir: Directory containing cached parquet files.
        fetch: Callable with the yfinance ``download`` signature
            (``tickers, start, end, **kwargs``) returning an OHLCV DataFrame.
            Defaults to ``yf.download``; inject a stub in tests.

    Returns:
        Normalized DataFrame with MultiIndex (ticker, date) and lowercase
        OHLCV columns.
    """
    cache_dir = Path(cache_dir)
    cache_key = _get_cache_key(tickers, start, end)
    fetch = fetch or _download_fetch
    cached, cached_path = _usable_cache(tickers, start, cache_dir)

    if cached is None:
        # No usable cache: full-range download. Normalize the fetch result the
        # same way "download_ohlcv" does, so this fallback also uses `fetch`.
        fetched = fetch(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        raw = _from_fetch(fetched, tickers, validate=True)
        _save_cache(raw, cache_dir, cache_key)
        return raw

    last_date = cached.index.get_level_values("date").max()
    tail = fetch(
        tickers,
        start=last_date.strftime("%Y-%m-%d"),
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if tail is None or tail.empty:
        return cached
    tail = _from_fetch(tail, tickers, validate=False)
    tail = tail.loc[tail.index.get_level_values("date") > last_date]
    merged = pd.concat([cached, tail])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    report = validate_ohlcv(merged)
    if not report.is_valid:
        raise ValueError("OHLCV validation failed: " + "; ".join(report.errors))

    _save_cache(merged, cache_dir, cache_key)
    if cached_path is not None:
        try:
            cached_path.unlink()
        except OSError:
            pass
    return merged


def _from_fetch(fetched: pd.DataFrame, tickers: list[str], validate: bool) -> pd.DataFrame:
    """Normalize a yfinance-style fetch result to (ticker, date) OHLCV columns.

    Accepts single- or multi-ticker frames, either as MultiIndex columns
    (yfinance group_by="ticker") or flat columns, and frames already indexed
    by (ticker, date).
    """
    column_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    if isinstance(fetched.columns, pd.MultiIndex):
        frames = []
        for ticker in tickers:
            if ticker in fetched.columns.get_level_values(0):
                sub = fetched[ticker].copy()
                sub.index = pd.to_datetime(sub.index)
                sub.index.name = "date"
                sub = sub.rename(columns=column_mapping)
                sub["ticker"] = ticker
                frames.append(sub.reset_index().set_index(["ticker", "date"]))
        if not frames:
            raise ValueError(f"No data found for tickers: {tickers}")
        df = pd.concat(frames)
    elif set(fetched.index.names) == {"ticker", "date"}:
        df = fetched.rename(columns=column_mapping).sort_index()
    else:
        data = fetched.copy()
        data.index = pd.to_datetime(data.index)
        data.index.name = "date"
        data = data.rename(columns=column_mapping)
        data["ticker"] = tickers[0]
        df = data.reset_index().set_index(["ticker", "date"])
    expected_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError("Fetched frame lacks expected OHLCV columns")
    df = df[expected_cols]
    df = df.sort_index()
    if validate:
        report = validate_ohlcv(df)
        if not report.is_valid:
            raise ValueError("OHLCV validation failed: " + "; ".join(report.errors))
    return df


def _cache_filename(df: pd.DataFrame, cache_key: str) -> str:
    """Canonical cache filename: ``{timestamp}_{cache_key}_sha256_{sha}.parquet``."""
    timestamp = int(time.time())
    sha256 = _compute_sha256(df)
    return f"{timestamp}_{cache_key}_sha256_{sha256}.parquet"


def _save_cache(df: pd.DataFrame, cache_dir: Path, cache_key: str) -> Path:
    """Persist `df` under the canonical SHA256-named cache filename."""
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _cache_filename(df, cache_key)
    df.to_parquet(path)
    return path


def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Validate OHLCV data quality.

    Args:
        df: DataFrame with MultiIndex (ticker, date) and OHLCV columns

    Returns:
        ValidationReport with validation results
    """
    errors = []

    if df.empty:
        errors.append("OHLCV data is empty")

    # Check for duplicate (ticker, date) rows
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        errors.append(f"Found {dup_count} duplicate (ticker, date) rows")

    # Check prices are positive
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in df.columns:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{col}' has {neg_count} negative price values")

    # Check volume is non-negative
    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"Column 'volume' has {neg_vol} negative volume values")

    # Compute native missing rates per column
    missing_rates = {}
    for col in df.columns:
        missing_rates[col] = float(df[col].isna().mean())

    is_valid = len(errors) == 0

    return ValidationReport(is_valid=is_valid, errors=errors, native_missing_rates=missing_rates)
