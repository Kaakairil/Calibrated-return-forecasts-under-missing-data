"""Tests for trained-artifact persistence (save/load/verify)."""

import json

import numpy as np
import pandas as pd
import pytest

from calibmiss.artifacts import load_artifacts, save_artifacts
from calibmiss.calibration import get_calibrator
from calibmiss.config import CalibratorConfig, ImputerConfig, ModelConfig
from calibmiss.imputers import get_imputer
from calibmiss.models import get_model


def _fit_artifacts(rng_seed: int = 42) -> dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    imputer = get_imputer(ImputerConfig(name="median", params={}))
    imputer.fit(X)
    model = get_model(ModelConfig(name="logistic_regression", params={}))
    model.fit(imputer.transform(X), y)
    calibrator = get_calibrator(CalibratorConfig(name="platt", params={}))
    calibrator.fit(model.predict_proba(imputer.transform(X)), y.to_numpy())
    return {"imputer": imputer, "model": model, "calibrator": calibrator}


def test_save_load_roundtrip(tmp_path) -> None:
    artifacts = _fit_artifacts()
    manifest = {"experiment_file": "experiment.yaml", "fit_window": ("2015-01-01", "2020-01-01")}
    path = save_artifacts(artifacts, tmp_path / "artifacts", manifest)

    loaded = load_artifacts(path)

    X = pd.DataFrame(np.random.default_rng(7).normal(size=(20, 3)), columns=["a", "b", "c"])
    orig_p = artifacts["model"].predict_proba(
        artifacts["imputer"].transform(X)
    )
    load_p = artifacts["model"].predict_proba(
        artifacts["imputer"].transform(X)
    )  # types erased; use loaded objects
    load_p = loaded["model"].predict_proba(loaded["imputer"].transform(X))
    assert np.allclose(orig_p, load_p)

    cal_orig = artifacts["calibrator"].predict_proba(orig_p)
    cal_load = loaded["calibrator"].predict_proba(load_p)
    assert np.allclose(cal_orig, cal_load)


def test_manifest_records_config_and_fit_window(tmp_path) -> None:
    manifest = {
        "config": {"experiment_file": "experiment_volatility.yaml", "n_imputers": 4},
        "fit_window": ("2015-01-01", "2020-01-01"),
        "run_timestamp": "2026-11-21T00:00:00Z",
    }
    path = save_artifacts(_fit_artifacts(), tmp_path / "a", manifest)
    with open(path / "manifest.json") as f:
        stored = json.load(f)
    assert stored["config"]["experiment_file"] == "experiment_volatility.yaml"
    assert stored["fit_window"] == ["2015-01-01", "2020-01-01"]
    assert stored["run_timestamp"] == manifest["run_timestamp"]


def test_load_rejects_missing_artifact(tmp_path) -> None:
    import os

    path = save_artifacts(_fit_artifacts(), tmp_path / "b", {})
    os.remove(path / "model.joblib")
    with pytest.raises(ValueError, match="model"):
        load_artifacts(path)


def test_load_rejects_wrong_experiment_file(tmp_path) -> None:
    path = save_artifacts(
        _fit_artifacts(), tmp_path / "c", {"config": {"experiment_file": "experiment.yaml"}}
    )
    with pytest.raises(ValueError, match="experiment_volatility.yaml"):
        load_artifacts(path, expected_experiment="experiment_volatility.yaml")
