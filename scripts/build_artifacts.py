"""Fit and persist the serving artifact set from processed features."""

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibmiss.artifacts import save_artifacts  # noqa: E402
from calibmiss.refit import fit_pipeline_artifacts  # noqa: E402


def _package_versions() -> dict[str, str]:
    import importlib.metadata

    names = ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    parser.add_argument("--target", type=Path, default=ROOT / "data" / "processed" / "target.parquet")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "latest")
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--imputer", default="median")
    parser.add_argument("--calibrator", default="platt")
    parser.add_argument("--cal-frac", type=float, default=0.2)
    parser.add_argument("--experiment-file", default="experiment.yaml")
    args = parser.parse_args(argv)

    if not args.features.exists():
        raise FileNotFoundError(f"Features parquet not found: {args.features}")
    if not args.target.exists():
        raise FileNotFoundError(f"Target parquet not found: {args.target}")

    import pandas as pd

    X = pd.read_parquet(args.features)
    y = pd.read_parquet(args.target).iloc[:, 0]

    artifacts = fit_pipeline_artifacts(
        X,
        y,
        model=args.model,
        imputer=args.imputer,
        calibrator=args.calibrator,
        cal_frac=args.cal_frac,
    )

    manifest: dict[str, Any] = {
        "config": {"experiment_file": args.experiment_file},
        "model": args.model,
        "imputer": args.imputer,
        "calibrator": args.calibrator,
        "cal_frac": args.cal_frac,
        "window_start": str(X.index.get_level_values("date").min().date()),
        "window_end": str(X.index.get_level_values("date").max().date()),
        "n_rows": int(len(X)),
        "packages": _package_versions(),
    }
    out = save_artifacts(artifacts, args.artifacts_dir, manifest)
    print(f"Saved artifacts to {out} "
          f"(model={args.model}, imputer={args.imputer}, calibrator={args.calibrator}).")


if __name__ == "__main__":
    main()
