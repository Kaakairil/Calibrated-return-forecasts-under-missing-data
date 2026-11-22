# CalibMiss: Real-Time Readiness Preparation

**Status:** Prepared 2026-11-20. This document is the full plan (what, why, how) for converting CalibMiss from a batch calibration-backtest repo into a system that can produce and monitor daily next-day direction forecasts. It is the authoritative checklist; every section maps to an implementation step and a verification.

The repo has already been audited against three axes: point-in-time correctness, missingness realism, and results-versus-config integrity. That audit produced the gap list below. The audit also verified the committed results are internally truthful (all headline numbers re-derived from `results/metrics.csv` match the reports).

---

## Audit summary (the gaps this document closes)

### A. What is already correct (do not regress)

- **Point-in-time features** — all 8 features are lag-only (`shift(1/5/20/50)`, rolling std/mean); the direction target is next-day close vs close (`features.py`). The volatility target uses only forward prices by construction.
- **Leakage-controlled pipeline** — imputers fit on train only, calibrators on cal split only, model on train only (`pipeline.py`); MAR threshold is train-computed and reused for cal/test.
- **Split hygiene** — chronological walk-forward with purge gaps; date-axis splits keep same-date rows together (`splits.py`).
- **Baselines** — naive base rate, complete-data oracle, volatility persistence; all fit on train only.
- **Reproducibility** — fixed seeds, SHA256-cached raw data, pinned `requirements.txt`.
- **Committed numbers are truthful** — verified: naive Brier 0.2489 / ECE 0.0170 / AUC 0.500; LR-median-MCAR 0% none→Platt ECE 0.0234→0.0199; LightGBM none→Platt 0.0407→0.0194; 507/1,728 Holm-significant; 495/15,840 AUC > 0.55.

### B. Real-time / serving gaps (the core of this work)

| # | Gap | Current state | Target |
|---|---|---|---|
| 1 | No inference entry point | Nothing returns a next-day up-probability for the latest row of a ticker using a *fitted* imputer+scaler+model+calibrator. | `serve.py`: incremental append → point-in-time features → apply fitted artifacts → calibrated up-probability per ticker, persisted to a dated prediction log. |
| 2 | No artifact serialization | No trained artifact is ever saved. | `artifacts/` with joblib-serialized (imputer, scaler, model, calibrator) + a metadata manifest (config snapshot, fit window, checksum). |
| 3 | No incremental data ingest | `download_ohlcv` re-downloads the full 2010–2024 range on every call. | Cached incremental append: fetch only rows newer than the cached last date; revalidate; keep SHA256 integrity. |
| 4 | No per-ticker new-row feature computation | Features are computed batch over full history. | `features.py:latest_features(raw, ticker)` — compute only the newest row for a ticker using its full history (purge warmup NaN). |
| 5 | No refit/recalibration policy | Folds exist for evaluation only. | `refit.py`: scheduled policies — model refit monthly, Platt/weekly, drift checks (rolling ECE of p vs realized). |
| 6 | No prediction log / monitoring | No timestamped prediction store. | `predictions/` daily CSV append with `(ts, ticker, date, features_hash, p_raw, p_cal, observed_y)`; drift reports. |
| 7 | No CI / deployment wiring | No `.github/`, no automated checks. | GitHub Actions workflow: install → pytest → ruff → smoke serve test on push/PR. |
| 8 | No structured "latest signal" output | Output is study tables only. | Serve writes `latest_signal.csv` (+ optionally parquet) for downstream consumption. |

### C. Research-integrity gaps

1. **COMMITTED RESULTS vs EXPANDED CONFIG — stale (confirmed).** `config/experiment.yaml` and `config/experiment_volatility.yaml` now declare **4 imputers (incl. drop) + 3 mechanisms (MCAR-BLOCK)** — but neither `results/metrics.csv` nor `results/volatility/metrics.csv` contains drop-imputer or MCAR-BLOCK rows (verified: only MCAR/MAR, 3 imputers). README line 28 and `report.md` §design describe the expanded design with old numbers; the only caveat is one line in `report.md:66`. `run_metadata.json` records only download date + package versions — **no config snapshot**, so the drift is not machine-detectable.

2. **Missingness-realism gaps (modeling, not code bugs):**
   - No **MNAR** mechanism (missingness depending on the missing value itself) — the classic confounder.
   - **MCAR-BLOCK streaks are independent per ticker** — a market-wide halt/vendor outage hitting many tickers the same day is not modeled.
   - MAR depends on a single driver (`volume_ratio`); spread/price-level/halt-state dependencies ignored.
   - **Zero-volume / halted days** are not explicitly handled (only negative volume is rejected); a daily live system must distinguish "ticker did not trade" from "data missing".
   - **Non-synchronous closes / thin trading** across tickers on the same date are ignored.

---

## Implementation plan (in execution order)

Each step: **RED** (failing test first) → **GREEN** → verify with `pytest` + `ruff`. No commit, no push (per instruction) — work stays in the working tree until instructed otherwise.

### Step 1 — Artifact persistence (`src/calibmiss/artifacts.py`)

**Why:** step 2 (serve) and step 5 (refit) both need fitted objects loaded and saved.

- `save_artifacts(artifacts: dict[str, object], path: Path, manifest: dict) -> Path` — joblib.dump each, write `manifest.json` (config snapshot from `get_config`, data checksum, fit-window bounds, package versions, timestamp).
- `load_artifacts(path: Path) -> dict[str, object]` — load and verify against manifest fingerprint.

**Tests (`tests/test_artifacts.py`):**
- round-trips model/imputer/calibrator/scaler (predict equal before/after save/load)
- manifest records config + fit window; load fails loudly on manifest mismatch (wrong experiment file recorded)

### Step 2 — Serving module (`src/calibmiss/serve.py`)

**Why:** this is the actual "happens in real time" entry point. Takes today's data, returns tomorrow's forecast.

- `latest_features(raw, ticker, feature_cols)` — recompute features for the newest row(s) of one ticker from its full per-ticker history; drop leading warmup NaN rows.
- `append_latest_rows(raw_current, new_df)` — concat, de-dup `(ticker, date)`, revalidate, return merged + flag which rows are new.
- `forecast_next_day(artifacts, raw_current, new_df, ticker=None) -> DataFrame` — for each (new-ticker-date) row: impute → scale → predict_proba → calibrate; return `(ticker, date, p_raw, p_cal)`.
- `run_daily(...)` — orchestrates ingest → latest-features → forecast → append prediction log → write `latest_signal.csv`.
- Predictions log: `predictions/predictions.csv` append-only with `(timestamp_utc, ticker, asof_date, features_sha256, p_raw, p_cal, y_observed)`; a `record_observations(...)` that back-fills `y_observed` when next-day data arrives.

**Tests (`tests/test_serve.py`):**
- synthetic multi-ticker history → fit a small artifacts set → `latest_features` returns the newest row only, values match the batch `compute_features` newest row
- `forecast_next_day` finite probabilities in [0,1], calibration applied (p_raw != p_cal when a Platt calibrator is used)
- `append_latest_rows` does not duplicate rows and flags new rows
- prediction log append + observation back-fill round-trip

### Step 3 — Incremental data ingest (`data.py` addition + `scripts/download_ohlcv.py`)

**Why:** live systems must not re-download 15 years of data daily.

- `download_incremental(tickers, start, end, cache_dir, max_age_days=5)` — take newest cached parquet, fetch `end` back to `last_cached_date`, merge, re-save under a new SHA256 name, drop the old cache file. Uses `yfinance.Ticker.history`-style ranges (full re-download only if no cache / corrupt).
- Guard: if the call is not incremental (no cache), fall back to `download_ohlcv`.

**Tests (`tests/test_data_incremental.py`):** monkeypatch `yf.download` to return only the tail window; assert merged frame has both old and new rows, no duplicates, new SHA256 filename, old file gone.

### Step 4 — Refit / recalibration policy (`src/calibmiss/refit.py`)

**Why:** production calibration must stay current; drift must be spotted.

- `should_refit_artifact(last_fit_at, policy)` — date-based (model monthly, Platt weekly) + row-count based.
- `refit_on_schedule(...)` — refit per policy, save new artifacts, log a `refit_events.csv` row `(timestamp, artifact, window_start, window_end, n_rows, config_fingerprint)`.
- `drift_report(prediction_log, window_days=90)` — rolling ECE / mean-p vs base rate, produce a compact markdown report `reports/drift_YYYY-MM-DD.md`.

**Tests (`tests/test_refit.py`):** policy triggers on/off at boundaries; drift report computes ECE on a synthetic log; refit event log appends.

### Step 5 — Zero-trade / NaN-volume policy (`features.py` + config)

**Why:** a halted or non-trading day is not a missing-feature day; conflating them biases a live forecast.

- Config flag `universe.zero_volume_policy: "drop" | "keep"` (default **drop** for live, matching ARCA-like "did not trade" semantics). `compute_features` drops rows where `volume == 0` before feature computation when the policy is `drop`; `keep` preserves current behavior.
- Document counter: this also slightly changes validation row counts (note in `pipeline_run.md`).

**Tests (`tests/test_features_policy.py`):** zero-volume rows dropped under `drop`, kept under `keep`, and the same row's features match between policy and an explicit manual drop.

### Step 6 — Results-integrity: config snapshot in metadata (`run_experiment.py`)

**Why:** this is the machine-checkable fix for the staleness bug found in the audit.

- `write_run_metadata(...)` now persists `config_snapshot` (serialized `cfg.model_dump()`) and `grid_dimensions` (models/imputers/calibrators/mechanisms/rates/seeds counts + experiment-file name + target).
- Add `scripts/verify_results_config.py` — reads `metrics.csv` + `run_metadata.json` and errors if the mechanism/imputer columns present don't match the recorded grid dimensions.

**Tests (`tests/test_run_metadata.py`):** metadata file contains the snapshot; `verify_results_config.py` passes on a synthetic consistent pair and fails on a synthetic stale pair.

### Step 7 — Missingness realism (research extension)

**Why:** close the C.2 gaps. This is science, not plumbing.

- **MNAR**: `inject_mnar(X, rate, seed, target_feature)` — missing probability increases with the value of the *target feature itself* (e.g., missing `volatility_20d` when volatility is extreme), calibrated to the requested rate. Fit-parameters estimated on train. Mirror `inject_mar`'s threshold API.
- **Market-correlated MCAR-BLOCK**: `inject_mcar_block_market(X, rate, seed)` — a shared date mask selects global outage days; all tickers missing on those days; per-ticker streak blocks remain independent outside them.
- Register both in `missingness.yaml`-style config and in `get_*` switch; add to the mechanism grids as opt-in (`mechanisms: [..., "MNAR"]`) — direction grid can be rerun with them in the same pass as Step 6's integrity rerun.

**Tests (`tests/test_missingness_realism.py`):** realized rates land near targets; MNAR has higher missing rate on extreme target values; market-block days are identical across tickers and row-missing aligns.

### Step 8 — Full-grid rerun under the committed expanded config + docs reconciliation

**Why:** resolve C.1 permanently — make committed results match committed config.

- Rerun direction grid (`make experiment`, `--n-jobs 1`, ~2–3 h) and volatility grid (`make experiment-volatility`, same) under the current configs (drop imputer + MCAR-BLOCK + max_iter 50).
- Regenerate figures; update `README.md`, `reports/report.md`, `reports/summary.md`, `reports/pipeline_run.md` numbers + run dates + the staleness caveat removal.
- Remove the now-obsolete "results will be refreshed" caveat; keep limitations (overlap, survivorship, MNAR single-driver, etc.).
- Delete `scripts/diagnose_class_weight.py` only if it is no longer referenced (check first). Keep `results/` and `predictions/` separate (gitignore predictions).

### Step 9 — CI (`/.github/workflows/ci.yml`)

**Why:** the repo currently has zero CI.

- On `push`/`pull_request`: `python -m venv` or setup-python → `pip install -r requirements.txt` → `pytest -q` → `ruff check src tests scripts` → a smoke serve test that fits a tiny artifacts set on synthetic data and forecasts one row (keeps CI offline, no yfinance call).

### Step 10 — Documentation (`README.md`, `docs/COMPONENT_DOCUMENTATION.md`)

- README: new "Live / real-time usage" section — daily run, artifacts dir, prediction log schema, refit schedule, drift reports, latest_signal.csv.
- COMPONENT_DOCUMENTATION.md: add `serve.py`, `refit.py`, `artifacts.py`, `inject_mnar`, market-correlated block, incremental ingest, `verify_results_config`; mark the runs with the current config as single-source-of-truth.

---

## Verification matrix

| Step | Test file | Command |
|---|---|---|
| 1 | `tests/test_artifacts.py` | `pytest tests/test_artifacts.py -v` |
| 2 | `tests/test_serve.py` | `pytest tests/test_serve.py -v` |
| 3 | `tests/test_data_incremental.py` | `pytest tests/test_data_incremental.py -v` |
| 4 | `tests/test_refit.py` | `pytest tests/test_refit.py -v` |
| 5 | `tests/test_features_policy.py` | `pytest tests/test_features_policy.py -v` |
| 6 | `tests/test_run_metadata.py` | `pytest tests/test_run_metadata.py -v` |
| 7 | `tests/test_missingness_realism.py` | `pytest tests/test_missingness_realism.py -v` |
| 8 | n/a (rerun) | `make experiment` / `make experiment-volatility` + figures |
| 9 | n/a (CI) | workflow run on GitHub |
| 10 | n/a (docs) | `make test` + eyeball |

Full suite gate before any final report: `python -m pytest -q` and `ruff check src tests scripts` must be green. Alwayls: read the file first, keep diffs minimal, no commit/push until the user asks.

---

## Out of scope (explicitly not in this pass)

- Live market execution, order routing, brokerage integration, position sizing — research-only, per the README scope notice.
- Cost/tax/reg-trade modeling.
- Replacing yfinance with a licensed feed.
- Any multi-day backtest redesign (purge-gap and fold structure stay as-is, only their documentation and results change).
