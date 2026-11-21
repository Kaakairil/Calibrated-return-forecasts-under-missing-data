# CalibMiss Enhancement Plan — Runnable Live System & CI Guard

> **Status:** Proposed 2026-11-22. This plan covers the small units that complete the
> "real-time ready" story left open by `docs/REAL_TIME_PREPARATION.md` (Steps 1–7 and 9–10
> are done; Step 8, the full grid rerun, is running separately). Each item is a separate
> unit with its own tests, its own commit, and user approval before each commit and push.

---

## Context

The real-time preparation work shipped the building blocks — `artifacts.py`
(save/load with manifest), `serve.py` (forecast + prediction log), `refit.py`
(policy + drift report), `download_incremental` (tail-only ingest), and the
config-integrity guard (`verify_results_config`) — but **nothing wires them into a
runnable flow**. Today a deploy means copy-pasting a Python snippet from the README.
Three concrete gaps:

1. **No CLI entry points for the live path.** There is no script that builds artifacts
   from config, and no script that performs the daily run (ingest → merge → forecast →
   log → refit check). The Makefile has `run-single` for experiments only.
2. **The config-integrity guard is not enforced in CI.** `make verify-results` exists
   locally (and `scripts/verify_results_config.py` exits 1 on mismatch), but
   `.github/workflows/ci.yml` only runs lint, pytest, and the serve smoke test. The
   staleness bug the guard exists to catch would not fail a push.
3. **`predictions/` is unignored.** The prep doc said "gitignore predictions" but it was
   never added to `.gitignore` — any local daily-run test dirties the tree.

Also worth noting: `test_serve.py` already covers `forecast_next_day`/`run_daily`
locally on synthetic data, so no new serve smoke test is needed — only the CI *guard*
step and the wiring scripts are missing.

---

## Task 1 — `scripts/build_artifacts.py` (fit + persist in one command)

**What:** CLI that turns config + data into a deployable artifact set.

```
python scripts/build_artifacts.py \
    --model logistic_regression --imputer median --calibrator platt \
    --artifacts-dir artifacts/latest
```

**Steps in the script:**
1. `get_config("config")` → read tickers/date range/`zero_volume_policy` from
   `config/universe.yaml`, model/imputer/calibrator names from `config/experiment.yaml`
   (CLI flags override).
2. `load_raw()` (existing, `src/calibmiss/data.py`) → cached OHLCV parquet.
3. `compute_features(raw, zero_volume_policy=...)` (existing, `src/calibmiss/features.py`).
4. Drop incomplete warmup rows; `fit_pipeline_artifacts(X, y, model=..., imputer=...,
   calibrator=..., cal_frac=0.2)` (existing, `src/calibmiss/refit.py`).
5. `save_artifacts(artifacts, Path(args.artifacts_dir), manifest)` (existing,
   `src/calibmiss/artifacts.py`) — manifest records `config.experiment_file`, fit window
   (`X.index` min/max as `window_start`/`window_end`), fit timestamp, package versions
   (reuse the version-collection approach from `write_run_metadata`).

**Test (`tests/test_build_artifacts.py`):** run `main(["--artifacts-dir", tmp, ...])`
against synthetic data (monkeypatch `load_raw`), assert joblib files + `manifest.json`
exist, `load_artifacts(..., expected_experiment="experiment.yaml")` round-trips.

**Make target:** `build-artifacts: python scripts/build_artifacts.py`.

---

## Task 2 — `scripts/run_daily.py` (the daily live run)

**What:** one command that performs a full daily cycle using the saved artifacts.

```
python scripts/run_daily.py --artifacts-dir artifacts/latest --log-dir predictions
```

**Steps in the script:**
1. `load_artifacts(args.artifacts_dir, expected_experiment=...)` (existing).
2. `load_raw()` → cached frame; `download_incremental(tickers, last_date, end,
   cache_dir="data/raw")` (existing, `src/calibmiss/data.py`) → only-new rows.
   *Offline/dev mode:* `--offline` skips the fetch and re-forecasts the cached tail.
3. `append_latest_rows(raw, new_rows)` (existing, `src/calibmiss/serve.py`).
4. `run_daily(artifacts, raw, merged, log_dir=..., zero_volume_policy=...)` (existing)
   → writes `predictions/predictions.csv` + `predictions/latest_signal.csv`.
5. Refit check: read `predictions/refit_events.csv` (or a small `last_fit.json`) →
   `should_refit(...)` per artifact (`RefitPolicy` from config or CLI defaults:
   model 30d, calibrator 7d) → if due, `fit_pipeline_artifacts` + `save_artifacts` +
   `refit_on_schedule` to log the event (existing, `src/calibmiss/refit.py`).
6. Print the forecast table to stdout (ticker, date, p_raw, p_cal).

**Test (`tests/test_run_daily_script.py`):** synthetic data, monkeypatched
`download_incremental` (or `--offline`), assert `predictions.csv`/`latest_signal.csv`
written, forecast columns present, refit event appended when `last_fit` is empty.

**Make target:** `run-daily: python scripts/run_daily.py`.

---

## Task 3 — CI: add the `verify-results` guard step

**What:** one step in `.github/workflows/ci.yml`, after the pytest step:

```yaml
      - name: Verify results match recorded config
        run: python scripts/verify_results_config.py
```

**Caveat:** the guard currently fails ("run_metadata.json has no grid_dimensions")
because the committed metadata predates the snapshot feature. **Order matters:** this
step only goes green after Task 8's grid rerun writes a fresh `run_metadata.json` with
`config_snapshot` + `grid_dimensions`. Either (a) land this step in the same unit as the
Step-8 results update, or (b) land it now and accept a red CI until the rerun lands.
Recommended: (a).

---

## Task 4 — `.gitignore`: ignore `predictions/`

**What:** add one line to `.gitignore`:

```
predictions/
```

Per the prep doc ("Keep `results/` and `predictions/` separate — gitignore
predictions"). The daily-run logs are runtime state, not committed artifacts.

**Test:** none needed; verify with `git check-ignore -v predictions/x.csv`.

---

## Task 5 — `Makefile` + README wiring

- Add `build-artifacts` and `run-daily` to `.PHONY` and as targets (Tasks 1–2).
- README "Live / Real-Time Usage" section: replace the copy-paste Python snippets for
  steps 1–2 with the two commands, keeping the snippets as "what it does internally".
  Add both Make targets to the target list.

---

## Known asymmetry (document, don't fix here)

`serve.py`'s live features do not drop zero-return days (only optional zero-volume
drop), while the batch target drops them by construction. Harmless for direction
forecasting (the model just scores the row), but note it in the README live section
when rewriting the snippets.

## Out of scope (unchanged)

- Full grid rerun + doc-number updates = **Step 8**, running now, tracked separately.
- MNAR / market-correlated MCAR-BLOCK **into the committed grid** — implemented and
  tested in `missingness.py` but not yet in `experiment.yaml`; adding them is another
  Step-8-style rerun (defer).
- LightGBM feature-importance stability across missingness (stretch goal, component doc §11).
- Live execution/trading — permanently out of scope per README scope notice.

---

## Execution order & commit units

| # | Unit | Files | Depends on |
|---|------|-------|-----------|
| 1 | `scripts/build_artifacts.py` + test + Make target | `scripts/build_artifacts.py`, `tests/test_build_artifacts.py`, `Makefile` | — |
| 2 | `scripts/run_daily.py` + test + Make target | `scripts/run_daily.py`, `tests/test_run_daily_script.py`, `Makefile` | Unit 1 |
| 3 | `.gitignore` predictions/ | `.gitignore` | — |
| 4 | README live-section rewrite | `README.md` | Units 1–2 |
| 5 | CI verify-results step | `.github/workflows/ci.yml` | Step 8 rerun landed |

One commit per unit, author `Wildan akmal <239170433+Khairil-12@users.noreply.github.com>`,
no trailers, user approval before each commit and each push.

---

## Verification

- Per unit: `pytest tests/<new test> -q`, then `ruff check src tests scripts`.
- After Unit 2: run `python scripts/run_daily.py --offline --artifacts-dir artifacts/latest`
  end-to-end locally; confirm `predictions/latest_signal.csv` has p_cal in [0, 1] and
  `predictions/predictions.csv` gained rows; `git status` must stay clean
  (predictions/ ignored after Unit 3).
- Full gate before any push: `python -m pytest -q` and `ruff check src tests scripts` green.
