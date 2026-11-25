"""Tests for run-metadata config snapshot and results-config verification."""

import json
from pathlib import Path

import pandas as pd
from calibmiss.config import get_config
from calibmiss.run_experiment import verify_results_config, write_run_metadata


def _tiny_config():
    cfg = get_config(Path(__file__).resolve().parents[1] / "config").model_copy(deep=True)
    cfg.experiment.models = cfg.experiment.models[:1]
    cfg.experiment.imputers = cfg.experiment.imputers[:2]
    cfg.experiment.calibrators = cfg.experiment.calibrators[:1]
    cfg.experiment.missingness.mechanisms = ["MCAR", "MAR"]
    cfg.experiment.missingness.rates = [0.0, 0.1]
    cfg.experiment.missingness.seeds = [42]
    return cfg


def test_metadata_contains_config_snapshot_and_grid_dimensions(tmp_path: Path) -> None:
    cfg = _tiny_config()
    path = tmp_path / "run_metadata.json"
    write_run_metadata(
        "2026-11-11", path, config=cfg, experiment_file="experiment.yaml"
    )
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["config_snapshot"]["universe"]["tickers"][0] == "AAPL"
    assert metadata["config_snapshot"]["experiment"]["models"][0]["name"] == "logistic_regression"
    dims = metadata["grid_dimensions"]
    assert dims["models"] == ["logistic_regression"]
    assert dims["imputers"] == ["forward_fill", "median"]
    assert dims["calibrators"] == ["none"]
    assert dims["mechanisms"] == ["MCAR", "MAR"]
    assert dims["rates"] == [0.0, 0.1]
    assert dims["seeds"] == [42]
    assert dims["experiment_file"] == "experiment.yaml"
    assert dims["target"] == "direction"
    assert dims["n_mechanisms"] == 2


def _write_results(results_dir: Path, cfg, experiment_file: str = "experiment.yaml") -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    dims = {
        "models": [m.name for m in cfg.experiment.models],
        "imputers": [i.name for i in cfg.experiment.imputers],
        "calibrators": [c.name for c in cfg.experiment.calibrators],
        "mechanisms": list(cfg.experiment.missingness.mechanisms),
        "rates": list(cfg.experiment.missingness.rates),
        "seeds": list(cfg.experiment.missingness.seeds),
        "experiment_file": experiment_file,
        "target": cfg.experiment.target,
    }
    # Cover the full grid: both imputers, both mechanisms, both rates, one seed.
    metrics = pd.DataFrame(
        {
            "row_type": ["experiment", "experiment", "experiment", "experiment", "baseline"],
            "imputer": ["forward_fill", "forward_fill", "median", "median", "none"],
            "mechanism": ["MCAR", "MAR", "MCAR", "MAR", "none"],
            "model": ["logistic_regression"] * 4 + ["naive"],
            "calibrator": ["none"] * 5,
            "rate": [0.0, 0.0, 0.1, 0.1, 0.0],
            "seed": [42] * 4 + [None],
            "fold": [0] * 5,
            "brier": [0.25, 0.26, 0.3, 0.31, 0.26],
            "ece": [0.1] * 5,
            "roc_auc": [0.55, 0.54, 0.5, 0.49, 0.5],
        }
    )
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    metadata = {"download_date": "2026-11-11", "grid_dimensions": dims}
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_verify_passes_on_consistent_pair(tmp_path: Path) -> None:
    cfg = _tiny_config()
    _write_results(tmp_path, cfg)
    mismatches = verify_results_config(
        tmp_path / "metrics.csv", tmp_path / "run_metadata.json"
    )
    assert mismatches == []


def test_verify_fails_on_stale_config_pair(tmp_path: Path) -> None:
    cfg = _tiny_config()
    _write_results(tmp_path, cfg)
    # Simulate the staleness bug: metrics.csv has fewer imputers than recorded.
    metrics_path = tmp_path / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[metrics["row_type"] == "experiment", "imputer"] = "forward_fill"
    metrics.to_csv(metrics_path, index=False)

    mismatches = verify_results_config(
        metrics_path, tmp_path / "run_metadata.json"
    )
    assert any("median" in line for line in mismatches)
    assert any("has no" in line for line in mismatches)


def test_verify_without_grid_dimensions_reports_error(tmp_path: Path) -> None:
    (tmp_path / "metrics.csv").write_text("row_type,imputer,mechanism\n", encoding="utf-8")
    (tmp_path / "run_metadata.json").write_text("{}", encoding="utf-8")
    mismatches = verify_results_config(
        tmp_path / "metrics.csv", tmp_path / "run_metadata.json"
    )
    assert mismatches and "grid_dimensions" in mismatches[0]
