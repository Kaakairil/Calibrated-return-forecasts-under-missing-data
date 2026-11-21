"""Serving: turn the latest available data into next-day up-probabilities.

This is the real-time entry point CalibMiss previously lacked. A daily run:

1. appends only-new rows to the cached OHLCV frame (``append_latest_rows``),
2. recomputes point-in-time features for the newest rows
   (``latest_features``),
3. applies the fitted imputer + model + calibrator (``forecast_next_day``),
4. writes a dated prediction log and a ``latest_signal.csv``
   (``run_daily``).

All feature/calibration logic is identical to the batch pipeline; this module
only supplies the "one more day" wrapper and persistence.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibmiss.features import FEATURE_COLS


def _ticker_feature_frame(
    raw: pd.DataFrame, ticker: str, zero_volume_policy: str = "keep"
) -> pd.DataFrame:
    """Features for every row of one ticker, computed on its full history."""
    sub = raw.loc[[ticker]]
    if zero_volume_policy == "drop":
        sub = sub[sub["volume"] != 0]
    c = sub["adj_close"]
    r1 = np.log(c / c.shift(1))
    out = pd.DataFrame(index=sub.index)
    out["return_1d"] = r1
    out["return_5d"] = np.log(c / c.shift(5))
    out["return_20d"] = np.log(c / c.shift(20))
    out["volatility_5d"] = r1.rolling(5).std() * np.sqrt(252)
    out["volatility_20d"] = r1.rolling(20).std() * np.sqrt(252)
    out["volume_ratio"] = sub["volume"] / sub["volume"].rolling(20).mean()
    out["hl_range"] = (sub["high"] - sub["low"]) / sub["close"]
    out["price_vs_ma50"] = c / c.rolling(50).mean()
    return out[FEATURE_COLS]


def latest_features(
    raw: pd.DataFrame, ticker: str, zero_volume_policy: str = "keep"
) -> pd.DataFrame:
    """Point-in-time features for the single newest row of `ticker`.

    Recomputes features on the ticker's full history so rolling windows and
    lag terms match the batch pipeline, then returns only the last usable row.
    Rows whose warmup window is incomplete (NaN features) are excluded.

    Args:
        raw: OHLCV frame for all tickers.
        ticker: Ticker symbol to compute for.
        zero_volume_policy: "drop" excludes halted (volume == 0) rows first.

    Returns:
        One-row DataFrame indexed (ticker, date) with the feature columns.

    Raises:
        KeyError: If `ticker` is not present in `raw`.
    """
    if ticker not in raw.index.get_level_values("ticker"):
        raise KeyError(f"ticker {ticker!r} not in raw data")
    features = _ticker_feature_frame(raw, ticker, zero_volume_policy=zero_volume_policy)
    features = features[~features.isna().any(axis=1)]
    if features.empty:
        raise ValueError(f"ticker {ticker!r} has no rows with complete features")
    return features.tail(1)


def append_latest_rows(raw: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge `new` rows into `raw`, dropping (ticker, date) duplicates.

    Any duplicate (ticker, date) key resolves in favor of `new`.

    Args:
        raw: Existing OHLCV frame with MultiIndex (ticker, date).
        new: Additional rows (same schema) to merge in.

    Returns:
        Combined frame, sorted, without duplicate keys.
    """
    combined = pd.concat([raw, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def forecast_next_day(
    artifacts: dict[str, Any],
    raw: pd.DataFrame,
    new_rows: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    zero_volume_policy: str = "keep",
) -> pd.DataFrame:
    """Return next-day up-probabilities for the newest row of each ticker.

    Args:
        artifacts: Fitted imputer/model/calibrator mapping (see
            calibmiss.artifacts). Requires keys "imputer", "model",
            "calibrator"; an optional "scaler" is applied if present.
        raw: Full OHLCV history including the newest rows to forecast.
        new_rows: Optional; only these (ticker, date) rows are forecast. If
            given, each ticker's newest feature row is used; if None, forecasts
            the newest row of every ticker.
        tickers: Optional restriction (used when `new_rows` is None).

    Returns:
        DataFrame with columns: ticker, date (asof_date), p_raw, p_cal.

    Raises:
        KeyError: A required artifact key is missing.
    """
    missing = {"imputer", "model", "calibrator"} - set(artifacts)
    if missing:
        raise KeyError(f"artifacts missing: {sorted(missing)}")
    imputer = artifacts["imputer"]
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    scaler = artifacts.get("scaler")

    if new_rows is not None:
        tickers = sorted(new_rows.index.get_level_values("ticker").unique())
    else:
        tickers = tickers or sorted(raw.index.get_level_values("ticker").unique())

    features = pd.concat(
        [latest_features(raw, t, zero_volume_policy=zero_volume_policy) for t in tickers]
    )
    X_imp = imputer.transform(features)
    X_scale = scaler.transform(X_imp) if scaler is not None else X_imp
    p_raw = model.predict_proba(X_scale)
    p_cal = calibrator.predict_proba(p_raw)

    return pd.DataFrame(
        {
            "ticker": features.index.get_level_values("ticker"),
            "date": features.index.get_level_values("date"),
            "p_raw": p_raw,
            "p_cal": p_cal,
        }
    )


def record_observations(log_path: Path | str, observations: pd.DataFrame) -> None:
    """Back-fill realized outcomes (y_observed) into a prediction log.

    `observations` must have ticker, date/asof_date, and y_observed columns.
    Existing rows are matched by (ticker, date) and updated in place.

    Args:
        log_path: Path to a prediction-log CSV with a y_observed column.
        observations: Rows to write back.

    Raises:
        ValueError: If the log does not exist.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise ValueError(f"Prediction log {log_path} does not exist")
    log = pd.read_csv(log_path)
    date_col = "date" if "date" in log.columns else "asof_date"
    key = ["ticker", date_col]
    obs = observations.copy()
    if "date" in observations.columns:
        obs[date_col] = obs["date"]
    obs_idx = obs.set_index(key)
    log_idx = log.set_index(key)
    log_idx.loc[log_idx.index.isin(obs_idx.index), "y_observed"] = obs_idx.loc[
        obs_idx.index.isin(log_idx.index), "y_observed"
    ].to_numpy()
    log_idx.reset_index().to_csv(log_path, index=False)


def run_daily(
    artifacts: dict[str, Any],
    raw: pd.DataFrame,
    merged: pd.DataFrame,
    log_dir: Path | str = "predictions",
    tickers: list[str] | None = None,
    zero_volume_policy: str = "keep",
) -> pd.DataFrame:
    """Daily end-to-end run: forecast newest rows, persist log + latest signal.

    Args:
        artifacts: Fitted imputer/model/calibrator mapping.
        raw: Previous cached OHLCV frame.
        merged: Frame after appending today's rows (see append_latest_rows).
        log_dir: Directory for predictions.csv and latest_signal.csv.
        tickers: Optional restriction.
        zero_volume_policy: "drop" excludes halted rows before forecasting.

    Returns:
        The forecast DataFrame (also written to disk).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    forecast = forecast_next_day(
        artifacts, merged, tickers=tickers, zero_volume_policy=zero_volume_policy
    )
    now = pd.Timestamp.now(tz="UTC").isoformat()
    log_entries = pd.DataFrame(
        {
            "timestamp_utc": now,
            "ticker": forecast["ticker"],
            "date": forecast["date"],
            "p_raw": forecast["p_raw"],
            "p_cal": forecast["p_cal"],
            "y_observed": np.nan,
        }
    )

    log_path = log_dir / "predictions.csv"
    if log_path.exists():
        existing = pd.read_csv(log_path)
        key = ["ticker", "date"]
        existing_idx = existing.set_index(key)
        dup = log_entries.set_index(key).index.isin(existing_idx.index)
        log_entries = log_entries[~dup]
        combined = pd.concat([existing, log_entries]).reset_index(drop=True)
    else:
        combined = log_entries
    combined.to_csv(log_path, index=False)
    forecast.to_csv(log_dir / "latest_signal.csv", index=False)
    return forecast
