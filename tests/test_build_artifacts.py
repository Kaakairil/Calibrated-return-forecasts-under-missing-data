"""Tests for scripts/build_artifacts.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_artifacts  # noqa: E402

from calibmiss.artifacts import load_artifacts  # noqa: E402


def make_features_target(n_rows: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    idx = pd.MultiIndex.from_product([["T0"], dates], names=["ticker", "date"])
    X = pd.DataFrame(rng.normal(size=(n_rows, 8)), index=idx,
                     columns=["return_1d", "return_5d", "return_20d",
                              "volatility_5d", "volatility_20d",
                              "volume_ratio", "hl_range", "price_vs_ma50"])
    y = pd.Series(rng.integers(0, 2, n_rows), index=idx, name="target")
    return X, y


def write_processed(tmp_path: Path, X: pd.DataFrame, y: pd.Series) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    X.to_parquet(proc / "features.parquet")
    y.to_frame().to_parquet(proc / "target.parquet")


def test_build_artifacts_writes_joblib_and_manifest(tmp_path: Path) -> None:
    X, y = make_features_target()
    write_processed(tmp_path, X, y)
    out = tmp_path / "artifacts" / "latest"

    build_artifacts.main([
        "--features", str(tmp_path / "processed" / "features.parquet"),
        "--target", str(tmp_path / "processed" / "target.parquet"),
        "--artifacts-dir", str(out),
    ])

    artifacts = load_artifacts(out, expected_experiment="experiment.yaml")
    assert {"imputer", "model", "calibrator"} <= set(artifacts)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["experiment_file"] == "experiment.yaml"
    assert manifest["window_start"] == str(X.index.get_level_values("date").min().date())
    assert manifest["window_end"] == str(X.index.get_level_values("date").max().date())
    assert manifest["n_rows"] == len(X)


def test_build_artifacts_rejects_missing_features(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_artifacts.main([
            "--features", str(tmp_path / "missing.parquet"),
            "--artifacts-dir", str(tmp_path / "out"),
        ])
