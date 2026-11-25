"""Tests for scripts/run_daily.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_daily as daily_script  # noqa: E402
from calibmiss.refit import fit_pipeline_artifacts  # noqa: E402


def make_raw(n_rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    idx = pd.MultiIndex.from_product([["T0"], dates], names=["ticker", "date"])
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
        },
        index=idx,
    )


def write_raw_cache(raw: pd.DataFrame, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "1700000000_T0_2023-01-02_2024-03-01_sha256_abc.parquet"
    raw.to_parquet(path)
    return path


def _fit_artifacts(raw: pd.DataFrame, out: Path) -> Path:
    from calibmiss.features import compute_features

    X, y = compute_features(raw)
    X = X[~X.isna().any(axis=1)]
    y = y.reindex(X.index)
    fitted = fit_pipeline_artifacts(X, y)
    from calibmiss.artifacts import save_artifacts

    return save_artifacts(fitted, out, {"config": {"experiment_file": "experiment.yaml"}})


def test_run_daily_offline_writes_log_and_signal(tmp_path: Path) -> None:
    raw = make_raw()
    art_dir = _fit_artifacts(raw, tmp_path / "artifacts")
    write_raw_cache(raw, tmp_path / "raw")

    daily_script.main([
        "--artifacts-dir", str(art_dir),
        "--log-dir", str(tmp_path / "predictions"),
        "--raw-dir", str(tmp_path / "raw"),
        "--offline",
        "--raw-tail", "250",
        "--no-refit",
    ])

    signal = pd.read_csv(tmp_path / "predictions" / "latest_signal.csv")
    log = pd.read_csv(tmp_path / "predictions" / "predictions.csv")
    assert {"ticker", "date", "p_raw", "p_cal"} <= set(signal.columns)
    assert signal["p_cal"].between(0, 1).all()
    assert len(log) == 1


def test_run_daily_refits_when_never_fit(tmp_path: Path) -> None:
    raw = make_raw()
    art_dir = _fit_artifacts(raw, tmp_path / "artifacts")
    write_raw_cache(raw, tmp_path / "raw")
    # Simulate a persisted last_fit with no entries so refit is due.
    state = tmp_path / "predictions" / "last_fit.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({}), encoding="utf-8")

    daily_script.main([
        "--artifacts-dir", str(art_dir),
        "--log-dir", str(tmp_path / "predictions"),
        "--raw-dir", str(tmp_path / "raw"),
        "--offline",
        "--raw-tail", "250",
    ])

    events = pd.read_csv(tmp_path / "predictions" / "refit_events.csv")
    assert {"artifact", "timestamp_utc"} <= set(events.columns)
    # Artifacts were rewritten in place (manifest timestamp refreshed).
    manifest = json.loads((art_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "saved_at_utc" in manifest
