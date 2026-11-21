# CalibMiss Gap Analysis — Cross-Plan Status

> **Status:** Compiled 2026-11-22. Cross-references the three planning documents and
> verifies, against the actual code, what is done and what remains. This is the single
> place to check "what is still missing" before starting new work.

---

## Plans cross-referenced

| Plan | File | Scope |
|---|---|---|
| 7-issue review fixes | (plan-mode plan, issues 1–5) | Reviewer-flagged bugs: class_weight artifact, config cache, MAR threshold, dead config fields, survivorship note |
| Real-time preparation | `docs/REAL_TIME_PREPARATION.md` | 10-step conversion from batch research repo to real-time-ready system |
| Enhancement plan | `docs/ENHANCEMENT_PLAN.md` | 5 units completing the runnable live system + CI guard |

---

## 1. Seven-issue review — ALL RESOLVED (verified in code)

| Issue | Status | Evidence |
|---|---|---|
| 1. `class_weight: "balanced"` artifact in Platt ECE | Done | `models.py:22` default `None`; direction `experiment.yaml` unweighted; balanced kept only in `experiment_volatility.yaml` (allowed — volatility prevalence ~50% by construction) |
| 2. Config cache ignores arguments | Done | `config.py:100-125` — cache removed, fresh load per call; `reset_config` kept as no-op |
| 3. MAR quantile threshold per-split | Done | `pipeline.py:79-84` — threshold computed on train, passed to cal/test via `depends_on_threshold` |
| 4. Dead config fields (`method`, `evaluation.metrics`) | Done | `method` params deleted; `evaluation.metrics` wired at `run_experiment.py:256` |
| 5. Survivorship bias note | Done | `reports/report.md:137` — META/May-2012 wording present |

## 2. Real-time preparation (10 steps) — 9 done, Step 8 running

| Step | Status | Notes |
|---|---|---|
| 1. Artifacts module | Done | `src/calibmiss/artifacts.py`; tests in `tests/test_artifacts.py` |
| 2. Serve module | Done | `src/calibmiss/serve.py`; tests in `tests/test_serve.py` |
| 3. Incremental ingest | Done | `download_incremental` in `data.py`; `tests/test_data_incremental.py` |
| 4. Refit policy + drift | Done | `src/calibmiss/refit.py`; `tests/test_refit.py` |
| 5. Zero-volume policy | Done | `features.py`, `serve.py`, `config.py`, `universe.yaml` |
| 6. Config snapshot + verify guard | Done | `write_run_metadata(config=...)`, `verify_results_config`, `make verify-results` |
| 7. Missingness realism (MNAR, market-block) | Done | `inject_mnar`, `inject_mcar_block_market` + tests (in `tests/test_missingness.py`, not a separate `test_missingness_realism.py` — tests exist, file name differs from the plan's matrix) |
| 8. Full-grid rerun + docs reconciliation | **RUNNING** | Direction grid in background (~40%+ at last check, ~2.5h remaining); volatility grid after; then figures + README/report/summary/pipeline_run numbers |
| 9. CI workflow | Done | `.github/workflows/ci.yml` (lint, pytest, offline serve smoke test; YAML fixed in `92d4a03`) |
| 10. Docs | Done | README live section + `docs/COMPONENT_DOCUMENTATION.md` §4.14–4.17 (commits `8f7b17c`, `b82eb2b`) |

## 3. Enhancement plan — NOT YET IMPLEMENTED

All five units of `docs/ENHANCEMENT_PLAN.md` are pending (this is the plan, not a gap):

1. `scripts/build_artifacts.py` + test + Make target
2. `scripts/run_daily.py` + test + Make target (with `--offline`)
3. `.gitignore`: add `predictions/`
4. README live-section rewrite to use the new commands
5. CI `verify-results` step (land with Step 8 results — red until then)

## 4. REMAINING GAPS — not covered by any plan

### Gap A — MNAR + MCAR-BLOCK-MARKET not in the committed grid (MEDIUM)

`inject_mnar` and `inject_mcar_block_market` are implemented and tested
(`missingness.py:226`, `missingness.py:294`) but `config/experiment.yaml` mechanisms
are still `["MCAR", "MCAR-BLOCK", "MAR"]`. **Impact:** the committed results say
nothing about MNAR or market-correlated outages; the mechanisms exist only as library
capabilities. **Cost:** another full-grid rerun (each added mechanism multiplies the
grid by ~1.33 for MNAR, +1 mechanism for MARKET). **Recommendation:** defer until
after Step 8 completes; if run, fold into the same rerun pass rather than a third run.

### Gap B — Step 8 completion itself (IN PROGRESS)

The grid rerun is necessary but not sufficient: after it finishes we must regenerate
figures, update the numbers in `README.md`, `reports/report.md`,
`reports/summary.md`, `reports/pipeline_run.md`, remove the staleness caveat, and
confirm `python scripts/verify_results_config.py` passes on the new
`run_metadata.json`. **This is the acceptance test for the whole config-integrity
story.**

### Gap C — minor, low-cost polish (LOW)

- `scripts/diagnose_class_weight.py` is a one-off diagnostic still in `scripts/`;
  delete it if no longer referenced (prep doc said "check first").
- `results_backup_pre_task8/` should be deleted once Step 8 lands and is verified.
- The serve path does not drop zero-return days (batch target does) — documented as a
  known asymmetry in `docs/ENHANCEMENT_PLAN.md`; no code change planned.

---

## Recommended sequence

1. **Wait for Step 8's direction grid** → regenerate figures → update doc numbers →
   run `verify_results_config` (Gap B).
2. **Volatility grid rerun** (second 2–3h run, also Step 8).
3. **Enhancement units 1–4** (safe to do while grids run — they don't touch `results/`).
4. **Enhancement unit 5** (CI guard) immediately after Step 8's results land.
5. **Gap A** (MNAR/market in grid) — optional, fold into a future rerun if desired.
6. **Gap C cleanup** at the very end.
