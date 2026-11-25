"""Tests for single-configuration experiment orchestration."""

import numpy as np
import pandas as pd
import pytest
from calibmiss.pipeline import run_single_config


def make_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    X = pd.DataFrame(
        rng.normal(size=(120, 4)), columns=["volatility_20d", "volume_ratio", "c", "d"]
    )
    y = pd.Series((X["volatility_20d"] + X["volume_ratio"] > 0).astype(int))
    return X, y, np.arange(60), np.arange(60, 90), np.arange(90, 120)


def make_config(rate: float = 0.1) -> dict[str, object]:
    return {
        "model": "logistic_regression",
        "imputer": "median",
        "calibrator": "platt",
        "mechanism": "MCAR",
        "rate": rate,
        "seed": 42,
    }


def test_run_single_config_returns_metrics_and_configuration() -> None:
    X, y, train, cal, test = make_data()

    result = run_single_config(X, y, train, cal, test, make_config())

    assert {"brier", "ece", "roc_auc", "reliability"} <= result.keys()
    assert result["model"] == "logistic_regression"
    assert result["rate"] == 0.1
    assert result["seed"] == 42
    assert result["fold"] == 0


def test_zero_missingness_baseline_runs() -> None:
    X, y, train, cal, test = make_data()

    result = run_single_config(X, y, train, cal, test, make_config(rate=0.0))

    assert np.isfinite(result["brier"])
    assert result["rate"] == 0.0


def test_unknown_mechanism_is_rejected_even_at_zero_rate() -> None:
    X, y, train, cal, test = make_data()
    config = make_config(rate=0.0)
    config["mechanism"] = "BOGUS"

    with pytest.raises(ValueError, match="Unknown mechanism: BOGUS"):
        run_single_config(X, y, train, cal, test, config)


def test_mnar_dispatches_through_pipeline() -> None:
    """MNAR must be runnable end-to-end, not library-only."""
    # make_data has volatility_20d/volume_ratio, the configured rule columns.
    X, y, train, cal, test = make_data()
    config = make_config()
    config["mechanism"] = "MNAR"
    result = run_single_config(X, y, train, cal, test, config)
    assert np.isfinite(result["brier"])
    assert result["mechanism"] == "MNAR"


def test_market_block_dispatches_through_pipeline() -> None:
    """MCAR-BLOCK-MARKET must be runnable end-to-end, not library-only."""
    X, y, train, cal, test = _ticker_data()
    config = make_config()
    config["mechanism"] = "MCAR-BLOCK-MARKET"
    result = run_single_config(X, y, train, cal, test, config)
    assert np.isfinite(result["brier"])
    assert result["mechanism"] == "MCAR-BLOCK-MARKET"


def test_mar_missingness_uses_configured_rule() -> None:
    X, y, train, cal, test = make_data()
    config = make_config()
    config["mechanism"] = "MAR"

    result = run_single_config(X, y, train, cal, test, config)

    assert result["mechanism"] == "MAR"
    assert np.isfinite(result["brier"])


def test_mar_rejects_non_finite_depends_on_column_before_injection() -> None:
    X, y, train, cal, test = make_data()
    X.iloc[0, X.columns.get_loc("volume_ratio")] = np.nan
    config = make_config()
    config["mechanism"] = "MAR"

    with pytest.raises(ValueError, match="volume_ratio"):
        run_single_config(X, y, train, cal, test, config)


def test_mar_pipeline_uses_train_threshold_for_all_splits(monkeypatch) -> None:
    """The MAR quantile threshold must be computed from the train split and
    reused for calibrate and test, so test-time missingness does not depend
    on the test split's own distribution."""
    import calibmiss.pipeline as pipeline_module
    from calibmiss.missingness import inject_mar as real_inject_mar

    X, y, train, cal, test = make_data()
    config = make_config()
    config["mechanism"] = "MAR"

    recorded: list[float | None] = []

    def spy(X_frame, rule, target_rate, seed, depends_on_threshold=None):
        recorded.append(depends_on_threshold)
        return real_inject_mar(X_frame, rule, target_rate, seed, depends_on_threshold)

    monkeypatch.setattr(pipeline_module, "inject_mar", spy)
    run_single_config(X, y, train, cal, test, config)

    assert len(recorded) == 3
    assert all(t is not None for t in recorded)
    assert len(set(recorded)) == 1
    expected = float(np.quantile(X.iloc[train]["volume_ratio"].to_numpy(), 0.7))
    assert recorded[0] == pytest.approx(expected)


def test_explicit_model_params_override_yaml() -> None:
    X, y, train, cal, test = make_data()
    config = make_config(rate=0.0)
    config["model_params"] = {"C": 0.01, "max_iter": 1000, "class_weight": None}

    result = run_single_config(X, y, train, cal, test, config)

    assert np.isfinite(result["brier"])


def _ticker_data(n_ticks: int = 5, rows_per_ticker: int = 60) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    idx = pd.MultiIndex.from_product(
        [range(n_ticks), range(rows_per_ticker)], names=["ticker", "date"]
    )
    X = pd.DataFrame(rng.normal(size=(n_ticks * rows_per_ticker, 3)), index=idx)
    y = pd.Series((X.iloc[:, 0] > 0).astype(int), index=idx)
    n = len(X)
    return X, y, np.arange(n // 2), np.arange(n // 2, int(n * 0.75)), np.arange(int(n * 0.75), n)


def test_mcar_block_runs_through_pipeline() -> None:
    X, y, train, cal, test = _ticker_data()
    config = make_config()
    config["mechanism"] = "MCAR-BLOCK"

    result = run_single_config(X, y, train, cal, test, config)

    assert result["mechanism"] == "MCAR-BLOCK"
    assert np.isfinite(result["brier"])


def test_seed_reproducibility_for_mcar_block() -> None:
    X, y, train, cal, test = _ticker_data()
    config = make_config()
    config["mechanism"] = "MCAR-BLOCK"
    first = run_single_config(X, y, train, cal, test, config)
    second = run_single_config(X, y, train, cal, test, config)
    assert {k: v for k, v in first.items() if "reliability" not in k} == {
        k: v for k, v in second.items() if "reliability" not in k
    }


def test_drop_imputer_drops_incomplete_rows() -> None:
    X, y, train, cal, test = _ticker_data()
    config = make_config()
    config["mechanism"] = "MCAR"
    config["imputer"] = "drop"
    config["imputer_params"] = {}

    result = run_single_config(X, y, train, cal, test, config)

    assert np.isfinite(result["brier"])
    assert result["imputer"] == "drop"


def test_split_sizes_recorded_and_drop_shrinks_eval_set() -> None:
    """Metrics rows must record post-imputation split sizes.

    `drop` removes incomplete rows from the eval splits, so its n_test is
    strictly smaller than the raw split — without the column, that selection
    was invisible in metrics.csv.
    """
    X, y, train, cal, test = _ticker_data()

    median = run_single_config(X, y, train, cal, test, make_config())
    assert {"n_train", "n_cal", "n_test"} <= median.keys()
    assert median["n_train"] == len(train)
    assert median["n_cal"] == len(cal)
    assert median["n_test"] == len(test)

    config = make_config()
    config["imputer"] = "drop"
    config["imputer_params"] = {}
    dropped = run_single_config(X, y, train, cal, test, config)
    assert 0 < dropped["n_test"] < len(test)
    assert 0 < dropped["n_cal"] < len(cal)
