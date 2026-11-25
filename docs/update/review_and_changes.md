11/25/26, 11:33 AM 

review_and_changes 

# **CalibMiss — Review Report and Changes** 

Date: 2026-11-24 Scope: missing-value handling, results integrity, and the fix applied this session. Head commit reviewed: `8aa5b4d6ba9c7fd8d8004ef49faa65f639b30747` (2026-1122). 

## **1. What the repo does well** 

- Real package structure: `src/calibmiss` with 15 focused modules (data, missingness, features, imputers, calibration, splits, serve, refit) — not one giant script. 

- 25 test files (~3,300 lines) covering the things that matter here: `test_features_lookahead.py` (leakage), `test_data_incremental.py` , `test_calibration.py` . 

- CI ( `.github/workflows/ci.yml` ): lint, `pytest -q` , offline serve/refit smoke test on Python 3.11. 

- Reproducibility: `results/run_metadata.json` records Python/package versions; `summary.csv` reports seed counts and stds; `statistical_tests.csv` uses paired 

- Wilcoxon with Holm correction; `make verify-results` guards config/results consistency. 

## **2. The missing-value issue (main finding)** 

### **The results grid is complete — no missing combos.** 

`results/volatility/summary.csv` : 288 rows = 2 models × 4 imputers × 3 calibrators × 3 mechanisms × 4 rates; every cell present at 110 seeds. At rate 0.0 all three mechanisms give identical Brier (verified by script). 

**The real problem:** **`drop` is evaluated on a much smaller, selected test set — and nothing recorded it.** 

Imputers are fit on train only (no fit leakage, `src/calibmiss/pipeline.py` ), but `DropImputer` also deletes NaN rows from the test split. Eval-set sizes, reconstructed from reliability `bin_counts` in `metrics.csv` (MCAR, logistic regression, fold 0): 

1/4 

about:blank 

11/25/26, 11:33 AM 

review_and_changes 

|**test rows**|**0%**|**10%**|**20%**|**30%**|
|---|---|---|---|---|
|drop|1,869|818|315|**117**|
|median / forward_fill / iterative|1,869|1,869|1,869|1,869|



Consequences: 

- At 30% MCAR, `drop` 's Brier (~0.22) is computed on only 117 rows (−94%), and those rows are the _complete_ ones — a selected, non-random subset. 

- The drop-vs-imputer contrasts in `statistical_tests.csv` pair seeds but not test rows, so the headline RQ1 comparison mixes information loss with sample selection and high variance. 

- `metrics.csv` had no `n_test` column, so the artifacts never exposed this. 

## **3. Change applied** 

Record post-imputation split sizes on every metrics row so the shrinkage is visible in the artifacts: 

- `src/calibmiss/pipeline.py` — `run_single_config` now returns `n_train` , `n_cal` , `n_test` (captured right after row alignment). 

- `src/calibmiss/run_experiment.py` — persistence and naive-training-rate baseline 

- rows carry the same columns (raw split sizes; they don't impute). Complete-data oracle rows inherit them via `run_single_config` , keeping `metrics.csv` rectangular. 

- `tests/test_pipeline.py` — new test 

- `test_split_sizes_recorded_and_drop_shrinks_eval_set` : columns present and 

- correct for `median` ; `drop` yields a strictly smaller, non-empty `n_test` . 

## **4. Verification** 

- Baseline (before the change): **154 passed, 1 failed** — 

- `tests/test_imputers.py::TestIterativeImputerWrapper::test_iterative_imputer` , 

- a numpy "assignment destination is read-only" error that exists on `main` in this workspace (newer numpy than the pinned 1.26.4). Untouched. 

- After the change: **155 passed, 1 failed** — same pre-existing failure; the new test passes; all 23 tests in `test_pipeline.py` + `test_run_experiment.py` (the modules covering the edit) pass. 

2/4 

about:blank 

11/25/26, 11:33 AM 

review_and_changes 

- Results CSV not regenerated here: the repo's pinned deps ( `numpy==1.26.4` , 

- `lightgbm==4.3.0` ) don't install on this workspace's Python 3.14. CI on 3.11 is the 

- source of truth. 

## **5. Other findings from the review** 

1. **Python pinning is implicit.** `numpy==1.26.4` / `lightgbm==4.3.0` fail to build on Python ≥3.13. CI uses 3.11, so it's green, but cloners on a new Python hit a confusing build error. Suggested: `requires-python = ">=3.11,<3.13"` in `pyproject.toml` and a README note. 

2. **Deprecated ruff keys.** `pyproject.toml` uses top-level `select` / `ignore` / `fixable` ; move under `[tool.ruff.lint]` . `ruff check .` on everything reports 8 errors (unsorted imports in `notebooks/exploration.ipynb` ) while CI checks only `src tests scripts` and passes — exclude notebooks or lint them deliberately. 

3. **Config drift.** `experiment_volatility.yaml` sets `class_weight: "balanced"` on logistic regression; `experiment.yaml` doesn't. Document it or align — results across the two experiments aren't comparable otherwise. 

4. **Hand-updated reports.** README RQ1 numbers and `reports/` are edited after reruns; a small `make report` regenerating them from `summary.csv` would remove the stale-number risk. 

## **6. Recommended next steps (in order)** 

1. **Regenerate results on the pinned env** so `n_test` lands in `results/metrics.csv` ( `make run-experiment` or your Makefile's target). 

2. **Fair drop-vs-imputer comparison** : score all imputers on a common complete-row eval subset as a secondary metric; pair rows (not just seeds) in the Wilcoxon tests. 

3. **Row-level uncertainty** : bootstrap CIs over test rows, or flag cells with `n_test < 500` in figures — at 117 rows, seed-level std understates uncertainty. 

4. **Re-state RQ1** : the `drop` degradation at high MCAR is partly a subset-selection effect; the imputed curves (constant 1,869 rows) are the trustworthy ones. Update README and `reports/report.md` after rerunning. 

## **7. Limitations** 

3/4 

about:blank 

11/25/26, 11:33 AM 

review_and_changes 

- Test suite verified in this workspace (Python 3.14 + current wheels), not the pinned 

- 3.11 env; one pre-existing failure is environment-related. 

- GitHub PR/issue state not checked (no `GH_TOKEN` available here). 

- Eval-set sizes in §2 were reconstructed from reliability `bin_counts` because 

- `metrics.csv` didn't record them — exact for fold 0, MCAR, logreg; other cells 

- follow the same mechanism. 

4/4 

about:blank 


---

## 8. Session 2026-11-25 — review recommendations implemented

Applies the recommendations in §5–§6 and closes the remaining grid gap (Gap A in
`GAP_ANALYSIS.md`).

### Changes

1. **`n_train`/`n_cal`/`n_test` recorded on every metrics row** (§6 step 1).
   - `src/calibmiss/pipeline.py` — `run_single_config` captures the post-imputation
     split sizes right after row alignment and returns them with the metrics.
   - `src/calibmiss/run_experiment.py` — naive and persistence baseline rows carry the
     raw split sizes (they do not impute); oracle rows inherit via `run_single_config`.
   - Test: `tests/test_pipeline.py::test_split_sizes_recorded_and_drop_shrinks_eval_set`
     (RED→GREEN): median keeps the full sizes; `drop` yields a strictly smaller,
     non-empty `n_test`. The next grid rerun exposes the drop-imputer's eval-set
     shrinkage (e.g. 117 rows at 30% MCAR) directly in `metrics.csv`.
2. **Ruff config fixed** (§5 finding 2): `select`/`ignore`/`fixable`/`unfixable` moved
   under `[tool.ruff.lint]` in `pyproject.toml`; 19 un-sorted import blocks auto-fixed
   across `src`/`tests`/`scripts`. `ruff check src tests scripts` now passes locally
   (it was exit 1 before — CI's lint step would have been red on a fresh checkout).
3. **Python pinning explicit** (§5 finding 1): `requires-python = ">=3.11,<3.13"` —
   numpy 1.26.4 / lightgbm 4.3.0 do not build on ≥3.13; CI (the source of truth) runs 3.11.
4. **class_weight drift documented** (§5 finding 3): comments above each
   `class_weight: "balanced"` in `config/experiment_volatility.yaml` state the drift is
   deliberate (volatility target ~50/50 by construction) and that results across the
   two grids are not model-comparable.
5. **Gap A closed — MNAR + MCAR-BLOCK-MARKET in the committed grid.**
   - `src/calibmiss/pipeline.py` — mechanism dispatch extended from 3 to 5
     (`MCAR`, `MCAR-BLOCK`, `MAR`, `MNAR`, `MCAR-BLOCK-MARKET`); MNAR uses the
     `config/missingness.yaml` rule's `target_feature`/params, MARKET uses
     `inject_mcar_block_market` with seed/seed+1/seed+2 per split like the others.
   - `config/experiment.yaml` + `config/experiment_volatility.yaml` — mechanisms
     registered as `["MCAR", "MCAR-BLOCK", "MAR", "MNAR", "MCAR-BLOCK-MARKET"]`.
   - Tests: `test_mnar_dispatches_through_pipeline`,
     `test_market_block_dispatches_through_pipeline`,
     `test_unknown_mechanism_is_rejected_even_at_zero_rate` (now uses a bogus name),
     and the config loader expectation updated.

### Verification (this workspace, Python 3.12.6 + pinned numpy 1.26.4 / lightgbm 4.3.0)

- Full suite: **158 passed**; `ruff check src tests scripts`: clean, no deprecation warning.
- The §4 note about the iterative-imputer numpy failure no longer applies to this
  workspace (it passed here); it may still fail on other newer-numpy environments.

### Rerun (this session, per instruction)

- `results_backup_pre_task8/` deleted; data/features regenerated; both grids rerun
  under the 5-mechanism config with `n_train`/`n_cal`/`n_test` now recorded.
- Figures + doc numbers updated after the rerun; `verify_results_config` passes on the
  fresh `run_metadata.json`.

### Also in this session

- **CI config-integrity guard** (Enhancement Plan Task 3): `.github/workflows/ci.yml`
  now runs `python scripts/verify_results_config.py` after pytest. It can go green
  immediately because the rerun's `run_metadata.json` records `config_snapshot` +
  `grid_dimensions` (the pre-snapshot metadata was the reason it was deferred).

### Not done

- §6 steps 2–4 (fair drop-vs-imputer common-subset metric, row-level bootstrap CIs,
  RQ1 re-statement in README/report) — analysis layered on the rerun's `metrics.csv`;
  a follow-up pass can read `n_test` directly now that it is recorded.
