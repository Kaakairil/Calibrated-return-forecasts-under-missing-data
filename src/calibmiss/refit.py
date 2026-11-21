"""Refit / recalibration policy and drift reporting for the live system.

Production calibration must stay current: the model is retrained on a slower
cadence, the calibrator on a faster one. This module decides *when* each
artifact is due (date and row-count policy), re-trains through the same
point-in-time fitting code as the batch pipeline, logs every refit to a
``refit_events.csv`` audit trail, and produces a rolling drift report from the
prediction log so stale calibration is caught rather than silently degrading.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibmiss.calibration import get_calibrator
from calibmiss.config import CalibratorConfig, ImputerConfig, ModelConfig
from calibmiss.imputers import get_imputer
from calibmiss.metrics import compute_ece
from calibmiss.models import get_model

Record = dict[str, Any]


@dataclass(frozen=True)
class RefitPolicy:
    """When a single artifact should be refit.

    Attributes:
        artifact: Canonical name ("model", "calibrator", "imputer", ...).
        interval_days: Refit at most every N days (wall clock).
        max_rows: Also refit when this many new rows have accumulated.
    """

    artifact: str
    interval_days: int = 30
    max_rows: int | None = None


def should_refit(
    last_fit_at: datetime | str | None,
    policy: RefitPolicy,
    rows_since: int = 0,
    now: datetime | None = None,
) -> bool:
    """Return whether `policy`'s artifact is due for a refit.

    An artifact is due if it has never been fit, if at least
    ``interval_days`` have passed since ``last_fit_at``, or if more than
    ``max_rows`` new rows have accumulated.

    Args:
        last_fit_at: Last fit timestamp (naive UTC or tz-aware), or None.
        policy: Refit cadence for the artifact.
        rows_since: Number of rows observed since the last fit.
        now: Reference "now"; defaults to the current UTC time.

    Returns:
        True if a refit is due.
    """
    if last_fit_at is None:
        return True
    if isinstance(last_fit_at, str):
        last_fit_at = datetime.fromisoformat(last_fit_at.replace("Z", "+00:00"))
    if last_fit_at.tzinfo is None:
        last_fit_at = last_fit_at.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if policy.max_rows is not None and rows_since >= policy.max_rows:
        return True
    return now - last_fit_at >= timedelta(days=policy.interval_days)


def fit_pipeline_artifacts(
    X: pd.DataFrame,
    y: pd.Series,
    model: str = "logistic_regression",
    imputer: str = "median",
    calibrator: str = "platt",
    model_params: dict[str, Any] | None = None,
    imputer_params: dict[str, Any] | None = None,
    calibrator_params: dict[str, Any] | None = None,
    cal_frac: float = 0.2,
) -> dict[str, Any]:
    """Fit imputer, model and calibrator point-in-time on the latest data.

    The most recent ``cal_frac`` of rows are reserved for calibration; the
    model is fit only on the earlier rows so the calibrator sees predictions
    from a model that has not seen those rows (mirrors the walk-forward split).

    Args:
        X: Feature frame (MultiIndex (ticker, date)).
        y: Binary target aligned to X.
        model: Model name for ``get_model``.
        imputer: Imputer name for ``get_imputer``.
        calibrator: Calibrator name for ``get_calibrator``.
        model_params: Optional overrides for the model wrapper.
        imputer_params: Optional overrides for the imputer wrapper.
        calibrator_params: Optional overrides for the calibrator wrapper.
        cal_frac: Fraction of the most recent rows used for calibration.

    Returns:
        Mapping {"imputer", "model", "calibrator"} of fitted objects.
    """
    ordered = X.sort_index()
    y_ordered = y.reindex(ordered.index)
    valid = ~ordered.isna().any(axis=1) & y_ordered.notna()
    ordered_valid = ordered[valid]
    y_valid = y_ordered[valid]
    n_split = int(np.floor(len(ordered_valid) * (1 - cal_frac)))
    if n_split < 2 or n_split >= len(ordered_valid):
        raise ValueError("Not enough rows to split train/calibration")
    X_train, X_cal = ordered_valid.iloc[:n_split], ordered_valid.iloc[n_split:]
    y_train, y_cal = y_valid.iloc[:n_split], y_valid.iloc[n_split:]

    imputer_obj = get_imputer(ImputerConfig(name=imputer, params=imputer_params or {}))
    imputer_obj.fit(X_train)
    X_train_imp = imputer_obj.transform(X_train)
    X_cal_imp = imputer_obj.transform(X_cal)

    model_obj = get_model(ModelConfig(name=model, params=model_params or {}))
    model_obj.fit(X_train_imp, y_train)

    if calibrator == "none":
        calibrator_obj = get_calibrator(CalibratorConfig(name="none", params={}))
    else:
        calibrator_obj = get_calibrator(
            CalibratorConfig(name=calibrator, params=calibrator_params or {})
        )
        calibrator_obj.fit(model_obj.predict_proba(X_cal_imp), y_cal.to_numpy())

    return {"imputer": imputer_obj, "model": model_obj, "calibrator": calibrator_obj}


def refit_on_schedule(
    policies: list[RefitPolicy],
    last_fit: dict[str, datetime | str | None],
    rows_since: dict[str, int],
    now: datetime,
    events_path: Path | str,
    refit_func: Callable[[RefitPolicy, datetime], dict[str, Any]],
    config_fingerprint: str | None = None,
) -> list[Record]:
    """Run each due artifact's refit and append rows to the event log.

    Args:
        policies: Refit cadences.
        last_fit: last-fit time per artifact name.
        rows_since: Row counts observed per artifact since its last fit.
        now: Current timestamp used for both decisions and the log.
        events_path: ``refit_events.csv`` path (directory parents created).
        refit_func: ``(policy, now) -> result dict`` actually performing the
            re-fit; must return an artifact name via ``result["artifact"]``.
        config_fingerprint: Identifier of the config used for this refit.

    Returns:
        The result dicts of the refits performed (empty if none were due).
    """
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    performed: list[Record] = []
    for policy in policies:
        due = should_refit(
            last_fit.get(policy.artifact),
            policy,
            rows_since.get(policy.artifact, 0),
            now=now,
        )
        if not due:
            continue
        result = refit_func(policy, now)
        performed.append(result)
        # Advance the caller's bookkeeping so a re-run doesn't refit again.
        last_fit[policy.artifact] = now
        row = {
            "timestamp_utc": now.isoformat(),
            "artifact": policy.artifact,
            "window_start": result.get("window_start", ""),
            "window_end": result.get("window_end", ""),
            "n_rows": result.get("n_rows", 0),
            "config_fingerprint": config_fingerprint or "",
        }
        header = not events_path.exists()
        record_df = pd.DataFrame([row])
        record_df.to_csv(
            events_path, mode="a", header=header, index=False
        )
    return performed


def drift_report(
    prediction_log: Path | str,
    window_days: int = 90,
    output_dir: Path | str = "reports",
    asof: str | None = None,
) -> Path:
    """Write a compact markdown drift report over the recent prediction log.

    Args:
        prediction_log: Path to the prediction-log CSV (must have ticker,
            date, p_cal, y_observed columns).
        window_days: Look-back window for the drift statistics.
        output_dir: Directory for the report.
        asof: Report date; defaults to today (UTC).

    Returns:
        Path to the written report file.
    """
    prediction_log = Path(prediction_log)
    if not prediction_log.exists():
        raise ValueError(f"Prediction log {prediction_log} does not exist")
    log = pd.read_csv(prediction_log)
    if "y_observed" not in log.columns or "p_cal" not in log.columns:
        raise ValueError("Prediction log must have p_cal and y_observed columns")

    date_col = "date" if "date" in log.columns else "asof_date"
    log[date_col] = pd.to_datetime(log[date_col])
    asof_dt = pd.Timestamp(asof) if asof else pd.Timestamp.now(tz="UTC")
    cutoff = asof_dt - pd.Timedelta(days=window_days)
    recent = log[log[date_col] >= cutoff]
    observed = recent[recent["y_observed"].notna()]

    n_rows = int(len(recent))
    n_observed = int(len(observed))
    base_rate = (
        float(observed["y_observed"].mean()) if n_observed else float("nan")
    )
    mean_p_cal = float(observed["p_cal"].mean()) if n_observed else float("nan")
    ece = (
        compute_ece(
            observed["y_observed"].to_numpy(dtype=float).astype(int),
            observed["p_cal"].to_numpy(dtype=float),
            n_bins=10,
        )
        if n_observed
        else float("nan")
    )

    lines = [
        "# Prediction drift report",
        "",
        f"- generated_at_utc: {asof_dt.isoformat()}",
        f"- window_days: {window_days}",
        f"- asof_date: {asof_dt.date().isoformat()}",
        "",
        "## Statistics (observed rows only)",
        "",
        f"- n_rows (in window): {n_rows}",
        f"- n_rows (observed): {n_observed}",
        f"- base_rate: {base_rate:.4f}",
        f"- mean_p_cal: {mean_p_cal:.4f}",
        f"- ece: {ece:.4f}",
        "",
        "## Interpretation",
        "",
    ]
    if n_observed < 30:
        lines.append(f"- Too few observed rows ({n_observed}) for a reliable drift read.")
    elif abs(mean_p_cal - base_rate) > 0.05:
        lines.append(
            f"- mean_p_cal ({mean_p_cal:.3f}) {abs(mean_p_cal - base_rate):.3f} away from "
            f"the base rate ({base_rate:.3f}): probabilities may be miscalibrated; "
            f"recheck the calibrator."
        )
    elif ece > 0.05:
        lines.append(
            f"- ECE {ece:.3f} exceeds 0.05: consider recalibrating."
        )
    else:
        lines.append("- Calibration looks healthy: mean predicted probability tracks the base rate.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"drift_{asof_dt.date().isoformat()}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
