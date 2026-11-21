"""Run one daily live cycle: ingest, forecast, log, and check refit schedule."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibmiss.artifacts import load_artifacts, save_artifacts  # noqa: E402
from calibmiss.config import get_config  # noqa: E402
from calibmiss.data import download_incremental  # noqa: E402
from calibmiss.features import compute_features  # noqa: E402
from calibmiss.refit import (  # noqa: E402
    RefitPolicy,
    fit_pipeline_artifacts,
    refit_on_schedule,
)
from calibmiss.serve import run_daily  # noqa: E402

LAST_FIT_PATH = "last_fit.json"
DEFAULT_POLICIES = [
    RefitPolicy(artifact="model", interval_days=30),
    RefitPolicy(artifact="calibrator", interval_days=7),
    RefitPolicy(artifact="imputer", interval_days=90),
]


def _load_last_fit(log_dir: Path) -> dict[str, Any]:
    path = log_dir / LAST_FIT_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_last_fit(log_dir: Path, last_fit: dict[str, Any]) -> None:
    (log_dir / LAST_FIT_PATH).write_text(
        json.dumps(last_fit, indent=2, default=str), encoding="utf-8"
    )


def _latest_raw_cache(raw_dir: Path) -> Path:
    caches = sorted(raw_dir.glob("*_sha256_*.parquet"))
    if not caches:
        raise FileNotFoundError(f"No raw OHLCV cache in {raw_dir}; run make data first.")
    return caches[-1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "latest")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--experiment-file", default="experiment.yaml")
    parser.add_argument("--offline", action="store_true",
                        help="Skip the network fetch; forecast the cached tail only.")
    parser.add_argument("--raw-tail", type=int, default=None,
                        help="Offline mode: use only the last N raw rows per ticker.")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip the refit-schedule check.")
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.artifacts_dir, expected_experiment=args.experiment_file)

    import pandas as pd

    cfg = get_config(ROOT / "config", args.experiment_file)
    raw = pd.read_parquet(_latest_raw_cache(args.raw_dir))
    if args.offline:
        if args.raw_tail:
            raw = raw.groupby(level="ticker", sort=False).tail(args.raw_tail)
        merged = raw
    else:
        last_date = str(raw.index.get_level_values("date").max().date())
        new_rows = download_incremental(
            cfg.universe.tickers, last_date, str(pd.Timestamp.now(tz="UTC").date()),
            cache_dir=args.raw_dir,
        )
        from calibmiss.serve import append_latest_rows

        merged = append_latest_rows(raw, new_rows)

    forecast = run_daily(
        artifacts, raw, merged,
        log_dir=args.log_dir,
        zero_volume_policy=cfg.universe.zero_volume_policy,
    )
    print(forecast.to_string(index=False))

    if args.no_refit:
        return

    last_fit = _load_last_fit(args.log_dir)
    performed = refit_on_schedule(
        DEFAULT_POLICIES,
        last_fit,
        {p.artifact: 0 for p in DEFAULT_POLICIES},
        datetime.now(UTC),
        events_path=args.log_dir / "refit_events.csv",
        refit_func=lambda policy, now: _refit(merged, args, policy),
        config_fingerprint=args.experiment_file,
    )
    if performed:
        _save_last_fit(args.log_dir, last_fit)
        print(f"Refit performed for: {[r['artifact'] for r in performed]}")
    else:
        print("No refit due.")


def _refit(merged, args: argparse.Namespace, policy: RefitPolicy) -> dict[str, Any]:
    X, y = compute_features(merged)
    fitted = fit_pipeline_artifacts(X, y)
    save_artifacts(fitted, args.artifacts_dir, {
        "config": {"experiment_file": args.experiment_file},
        "window_start": str(X.index.get_level_values("date").min().date()),
        "window_end": str(X.index.get_level_values("date").max().date()),
        "n_rows": int(len(X)),
    })
    return {
        "artifact": policy.artifact,
        "window_start": str(X.index.get_level_values("date").min().date()),
        "window_end": str(X.index.get_level_values("date").max().date()),
        "n_rows": int(len(X)),
    }


if __name__ == "__main__":
    main()
