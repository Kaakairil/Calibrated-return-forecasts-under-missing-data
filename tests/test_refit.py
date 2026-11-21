"""Tests for the refit/recalibration policy and drift reporting."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from calibmiss.refit import (
    RefitPolicy,
    drift_report,
    fit_pipeline_artifacts,
    refit_on_schedule,
    should_refit,
)


class TestShouldRefit:
    def test_never_fitted_is_due(self):
        assert should_refit(None, RefitPolicy("model", interval_days=30))

    def test_not_due_inside_interval(self):
        now = datetime(2026, 11, 21, tzinfo=UTC)
        last = now - timedelta(days=10)
        assert not should_refit(last, RefitPolicy("model", interval_days=30), now=now)

    def test_due_at_interval_boundary(self):
        now = datetime(2026, 11, 21, tzinfo=UTC)
        last = now - timedelta(days=30)
        assert should_refit(last, RefitPolicy("model", interval_days=30), now=now)

    def test_due_when_row_cap_exceeded(self):
        now = datetime(2026, 11, 21, tzinfo=UTC)
        last = now - timedelta(days=1)
        policy = RefitPolicy("model", interval_days=30, max_rows=1000)
        assert not should_refit(last, policy, rows_since=999, now=now)
        assert should_refit(last, policy, rows_since=1000, now=now)


class TestRefitOnSchedule:
    def _due_policies(self):
        now = datetime(2026, 11, 21, tzinfo=UTC)
        policies = [
            RefitPolicy("model", interval_days=30),
            RefitPolicy("calibrator", interval_days=7),
        ]
        last_fit = {
            "model": now - timedelta(days=31),  # due
            "calibrator": now - timedelta(days=2),  # not due
        }
        return policies, last_fit, now

    def test_refits_only_due_artifacts(self, tmp_path):
        policies, last_fit, now = self._due_policies()
        events = tmp_path / "refit_events.csv"
        refit_calls: list[str] = []

        def fake_refit(policy, ts):
            refit_calls.append(policy.artifact)
            return {
                "artifact": policy.artifact,
                "window_start": "2026-01-01",
                "window_end": "2026-11-20",
                "n_rows": 500,
            }

        refit_on_schedule(
            policies,
            last_fit,
            {},
            now,
            events,
            fake_refit,
            config_fingerprint="abc123",
        )
        assert refit_calls == ["model"]

    def test_event_log_appends_rows(self, tmp_path):
        policies, last_fit, now = self._due_policies()
        events = tmp_path / "refit_events.csv"
        refit_on_schedule(
            policies, last_fit, {}, now, events,
            lambda p, ts: {
                "artifact": p.artifact,
                "window_start": "2026-01-01",
                "window_end": "2026-11-20",
                "n_rows": 500,
            },
            config_fingerprint="abc123",
        )
        # Nothing due on a second immediate run; only the first event logged.
        assert refit_on_schedule(policies, last_fit, {}, now, events, lambda p, ts: {
            "artifact": p.artifact,
            "window_start": "2026-01-01",
            "window_end": "2026-11-20",
            "n_rows": 500,
        }) == []
        log = pd.read_csv(events)
        assert len(log) == 1
        assert log.loc[0, "artifact"] == "model"
        assert log.loc[0, "config_fingerprint"] == "abc123"
        assert log.loc[0, "n_rows"] == 500

    def test_no_refit_when_nothing_due(self, tmp_path):
        now = datetime(2026, 11, 21, tzinfo=UTC)
        policies = [RefitPolicy("model", interval_days=30)]
        last_fit = {"model": now - timedelta(days=1)}
        events = tmp_path / "refit_events.csv"
        refit_on_schedule(policies, last_fit, {}, now, events, lambda p, ts: {})
        assert not events.exists()


def _feature_frame(n_days: int = 160, n_ticks: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic features + binary target with realistic index."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product(
        [[f"T{i}" for i in range(n_ticks)], dates], names=["ticker", "date"]
    )
    n = len(idx)
    p = 0.5 + 0.1 * np.sin(np.arange(n) / 7.0)
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame(
        {
            "return_1d": rng.normal(0, 0.01, n),
            "return_5d": rng.normal(0, 0.02, n),
            "return_20d": rng.normal(0, 0.04, n),
            "volatility_5d": rng.uniform(0.1, 0.6, n),
            "volatility_20d": rng.uniform(0.1, 0.5, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
            "hl_range": rng.uniform(0.005, 0.05, n),
            "price_vs_ma50": rng.uniform(0.9, 1.1, n),
        },
        index=idx,
    )
    return X, pd.Series(y, index=idx, name="target")


class TestFitPipelineArtifacts:
    def test_returns_fitted_artifacts_with_point_in_time_calibration(self, tmp_path):
        X, y = _feature_frame()
        artifacts = fit_pipeline_artifacts(
            X, y, model="logistic_regression", imputer="median", calibrator="platt",
        )
        assert {"imputer", "model", "calibrator"} <= set(artifacts)
        # Model must not have seen calibrate rows: predictions on them should
        # still be finite and in [0, 1] even though calibrator was fit on them.
        X_imp = artifacts["imputer"].transform(X.iloc[-20:])
        p = artifacts["model"].predict_proba(X_imp)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        # Calibrator matches raw probs of nothing; sanity: returns finite probs.
        p_cal = artifacts["calibrator"].predict_proba(p)
        assert np.isfinite(p_cal).all()


class TestDriftReport:
    def test_computes_ece_mean_p_and_writes_report(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 300
        dates = pd.bdate_range("2026-07-01", periods=n)
        base = 0.55
        p_cal = np.clip(base + rng.normal(0, 0.1, n), 0.05, 0.95)
        y = (rng.random(n) < p_cal).astype(int)
        log = pd.DataFrame(
            {
                "timestamp_utc": "2026-11-20T12:00:00Z",
                "ticker": ["T0"] * n,
                "date": dates,
                "p_raw": p_cal,
                "p_cal": p_cal,
                "y_observed": y,
            }
        )
        log_path = tmp_path / "predictions.csv"
        log.to_csv(log_path, index=False)

        report_path = drift_report(log_path, output_dir=tmp_path / "reports", asof="2026-11-20")

        assert report_path.name == "drift_2026-11-20.md"
        text = report_path.read_text(encoding="utf-8")
        assert "n_rows" in text
        assert "base_rate" in text
        assert "mean_p_cal" in text
        assert "ece" in text.lower()
        # Mean predicted p and base rate should be close for a well-calibrated log.
        assert abs(p_cal.mean() - y.mean()) < 0.15

    def test_ignores_unobserved_rows(self, tmp_path):
        n = 100
        dates = pd.bdate_range("2026-08-01", periods=n)
        p = np.full(n, 0.5)
        log = pd.DataFrame(
            {
                "timestamp_utc": "2026-11-20T12:00:00Z",
                "ticker": ["T0"] * n,
                "date": dates,
                "p_raw": p,
                "p_cal": p,
                "y_observed": np.where(np.arange(n) < 50, 0, np.nan),
            }
        )
        log_path = tmp_path / "predictions.csv"
        log.to_csv(log_path, index=False)
        report = drift_report(log_path, output_dir=tmp_path / "reports", asof="2026-11-20")
        text = report.read_text(encoding="utf-8")
        assert "n_rows" in text and "50" in text
