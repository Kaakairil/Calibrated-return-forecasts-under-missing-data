"""Tests for the serving module (incremental ingest, latest features, forecast)."""

import numpy as np
import pandas as pd
import pytest

from calibmiss.artifacts import save_artifacts
from calibmiss.calibration import get_calibrator
from calibmiss.config import CalibratorConfig, ImputerConfig, ModelConfig
from calibmiss.features import compute_features
from calibmiss.imputers import get_imputer
from calibmiss.models import get_model
from calibmiss.serve import (
    append_latest_rows,
    forecast_next_day,
    latest_features,
    record_observations,
    run_daily,
)


def _raw_history(n_ticks: int = 3, n_days: int = 260) -> pd.DataFrame:
    """Synthetic OHLCV with a realistic price path so features are finite."""
    rng = np.random.default_rng(3)
    frames = []
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    for t in range(n_ticks):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        open_ = close * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n_days))
        vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": vol,
            },
            index=dates,
        )
        frame["ticker"] = f"T{t}"
        frames.append(frame.reset_index())
    df = pd.concat(frames)
    df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["ticker", "date"]).sort_index()


def _fit_artifacts(tmp_path, n_days: int = 260):
    raw = _raw_history(n_ticks=3, n_days=n_days)
    X, _ = compute_features(raw)
    X = X[~X.isna().any(axis=1)]
    y = pd.Series(rng_int(X.index), index=X.index)
    imputer = get_imputer(ImputerConfig(name="median", params={}))
    imputer.fit(X)
    model = get_model(ModelConfig(name="logistic_regression", params={}))
    model.fit(imputer.transform(X), y)
    calibrator = get_calibrator(CalibratorConfig(name="platt", params={}))
    calibrator.fit(model.predict_proba(imputer.transform(X)), y.to_numpy())
    save_artifacts(
        {"imputer": imputer, "model": model, "calibrator": calibrator},
        tmp_path / "artifacts",
        {"config": {"experiment_file": "experiment.yaml"}},
    )
    return tmp_path / "artifacts", X, y


def rng_int(index):
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=len(index))


def _one_new_row(raw):
    """Append one new trading day to every ticker of `raw`, returning (merged, new)."""
    last_dates = raw.index.get_level_values("date").max()
    new_date = pd.bdate_range(last_dates, periods=2)[1]
    rng = np.random.default_rng(1)
    frames = []
    for ticker in raw.index.get_level_values("ticker").unique():
        prev = raw.loc[ticker].iloc[-1]
        close = prev["close"] * (1 + rng.normal(0, 0.01))
        frames.append(
            pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "adj_close": close,
                    "volume": 2_000_000.0,
                },
                index=pd.MultiIndex.from_tuples([(ticker, new_date)], names=["ticker", "date"]),
            )
        )
    new = pd.concat(frames)
    merged = pd.concat([raw, new]).sort_index()
    return merged, new


class TestLatestFeatures:
    def test_matches_batch_computation_newest_row(self):
        raw = _raw_history()
        batch_X, _ = compute_features(raw)
        batch_X = batch_X[~batch_X.isna().any(axis=1)]
        for ticker in ["T0", "T1"]:
            latest = latest_features(raw, ticker)
            assert len(latest) == 1
            # The newest serve row is the newest date with computable features.
            assert latest.index.get_level_values("date")[0] == raw.loc[ticker].index[-1]
            # Its features must equal batch features at that same date.
            date = latest.index.get_level_values("date")[0]
            if date in batch_X.loc[ticker].index:
                expected = batch_X.loc[ticker].loc[date]
                assert np.allclose(latest.iloc[0].to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_requires_known_ticker(self):
        raw = _raw_history()
        with pytest.raises(KeyError, match="T99"):
            latest_features(raw, "T99")


class TestAppendLatestRows:
    def test_no_duplicates_and_flags_new(self):
        raw = _raw_history()
        merged, new = _one_new_row(raw)
        assert not merged.index.duplicated().any()
        assert len(merged) == len(raw) + len(new)
        assert merged.index.get_level_values("date").max() == new.index.get_level_values("date").max()

    def test_reappend_is_idempotent(self):
        raw = _raw_history()
        merged, _ = _one_new_row(raw)
        again = append_latest_rows(raw, merged)
        assert len(again) == len(merged)
        assert not again.index.duplicated().any()


class TestForecastNextDay:
    def test_finite_probabilities_and_calibration_applied(self, tmp_path):
        artifacts, X, y = _fit_artifacts(tmp_path)
        raw = _raw_history()
        merged, new = _one_new_row(raw)
        from calibmiss.artifacts import load_artifacts

        fitted = load_artifacts(artifacts)
        forecast = forecast_next_day(fitted, merged, new)
        assert set(forecast["ticker"]) == set(merged.index.get_level_values("ticker").unique())
        for _, row in forecast.iterrows():
            assert 0.0 <= row["p_raw"] <= 1.0
            assert 0.0 <= row["p_cal"] <= 1.0
            assert np.isfinite(row["p_raw"]) and np.isfinite(row["p_cal"])
        # Platt should move at least some probabilities.
        assert not np.allclose(forecast["p_raw"].to_numpy(), forecast["p_cal"].to_numpy())


class TestPredictionLog:
    def test_append_and_observe_roundtrip(self, tmp_path):
        log = tmp_path / "predictions.csv"
        df = pd.DataFrame(
            {
                "timestamp_utc": ["2026-11-20T12:00:00Z"] * 2,
                "ticker": ["T0", "T1"],
                "asof_date": ["2026-11-20", "2026-11-20"],
                "p_raw": [0.4, 0.6],
                "p_cal": [0.41, 0.59],
                "y_observed": [np.nan, np.nan],
            }
        )
        df.to_csv(log, index=False)
        # Back-fill one observation.
        record_observations(
            log,
            pd.DataFrame({"ticker": ["T0"], "asof_date": ["2026-11-20"], "y_observed": [1]}),
        )
        updated = pd.read_csv(log)
        assert updated.loc[updated["ticker"] == "T0", "y_observed"].iloc[0] == 1.0
        assert updated.loc[updated["ticker"] == "T1", "y_observed"].iloc[0] != 1.0

    def test_run_daily_writes_latest_signal(self, tmp_path):
        raw = _raw_history()
        merged, _ = _one_new_row(raw)
        artifacts, X, y = _fit_artifacts(tmp_path)
        from calibmiss.artifacts import load_artifacts

        out_dir = tmp_path / "out"
        run_daily(
            load_artifacts(artifacts),
            raw,
            merged,
            log_dir=out_dir,
        )
        signal = pd.read_csv(out_dir / "latest_signal.csv")
        assert len(signal) == len(merged.index.get_level_values("ticker").unique())
        assert {"ticker", "p_raw", "p_cal"} <= set(signal.columns)
        assert (out_dir / "predictions.csv").exists()
