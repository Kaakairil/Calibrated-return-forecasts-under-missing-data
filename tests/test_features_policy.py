"""Tests for the zero-volume / no-trade-day policy in feature computation."""

import numpy as np
import pandas as pd
import pytest

from calibmiss.features import compute_features


def _raw_with_zero_volume_day() -> pd.DataFrame:
    """Two tickers, one of which has an exact zero-volume (halted) day."""
    dates = pd.bdate_range("2023-01-02", periods=80)
    idx = pd.MultiIndex.from_product([["A", "B"], dates], names=["ticker", "date"])
    close = 100 + np.arange(len(idx)) % 50
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close.astype(float),
            "adj_close": close.astype(float),
            "volume": np.full(len(idx), 5_000_000.0, dtype=float),
        },
        index=idx,
    )
    # B halts deep in the series (past warmup) so features are complete there.
    halted = dates[40]
    df.loc[("B", halted), "volume"] = 0.0
    return df


class TestDropPolicy:
    def test_zero_volume_rows_dropped_under_drop(self):
        raw = _raw_with_zero_volume_day()
        X, _ = compute_features(raw, zero_volume_policy="drop")
        halted = raw.index.get_level_values("date")[40]
        assert ("B", halted) not in X.index
        # A is unaffected.
        assert X.index.get_level_values("ticker").unique().tolist() == ["A", "B"]

    def test_zero_volume_rows_kept_under_keep(self):
        raw = _raw_with_zero_volume_day()
        X, _ = compute_features(raw, zero_volume_policy="keep")
        # Past warmup, B's halted row has complete features and is kept.
        halted = raw.index.get_level_values("date")[40]
        assert ("B", halted) in X.index

    def test_drop_matches_manual_drop_with_keep(self):
        raw = _raw_with_zero_volume_day()
        X_drop, _ = compute_features(raw, zero_volume_policy="drop")
        manual = raw.loc[raw["volume"] != 0]
        X_manual, _ = compute_features(manual, zero_volume_policy="keep")
        # The multiindex rows must coincide.
        assert X_drop.index.equals(X_manual.index)
        pd.testing.assert_frame_equal(X_drop, X_manual)

    def test_invalid_policy_rejected(self):
        raw = _raw_with_zero_volume_day()
        with pytest.raises(ValueError, match="zero_volume_policy"):
            compute_features(raw, zero_volume_policy="bogus")
