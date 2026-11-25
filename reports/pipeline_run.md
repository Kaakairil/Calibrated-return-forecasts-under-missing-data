# CalibMiss pipeline run report

Run date: 2026-11-12; direction grid rerun without class weighting 2026-11-13. Machine: Windows 11, Python 3.12.6, 16 logical cores.

`make all` was not invoked as one command. Its stages were run individually in order, because the first full attempt crashed and two code fixes were required before the grid could finish. Every stage below maps to a `make` target.

## Stage results

| Stage | Command | Result |
|---|---|---|
| data | `python scripts/download_data.py` | OK. 113,190 raw rows, 30 tickers, cached to `data/raw/` with SHA256 in filename |
| features | `python scripts/compute_features.py` | OK. 106,028 feature rows written to `data/processed/` |
| experiment | `python scripts/run_grid.py` | OK. Exit code 0, 24.5 min for the parallel grid, 15,994 rows in `results/metrics.csv` |
| figures | `python scripts/generate_figures.py` | OK. 5 PNGs in `results/figures/` |
| report (PDF) | `pandoc reports/report.md -o reports/report.pdf` | NOT RUN. `pandoc` and `weasyprint` are not installed. The Makefile swallows this with `\|\| echo ... skipping PDF`, so `make all` would report success here anyway |
| tests | `python -m pytest` | OK. 87 passed in 13.4 s |

## Errors hit and fixes

1. **Grid crashed immediately on first attempt.**
   `ValueError: depends_on values must be finite` inside `inject_mar` (`src/calibmiss/missingness.py:98`), raised from a joblib worker.
   Cause: `compute_features` keeps the rolling-window warmup rows, so `volume_ratio` (the MAR driver) contained 553 NaN values. MCAR never touches that column's values, so the earlier single-config MCAR run passed and hid this.
   Fix: drop rows with any NaN feature before the matrix is used, in both `scripts/compute_features.py` and `_load_data` in `src/calibmiss/run_experiment.py`. Row count went from 107,444 to 106,028. All 87 tests still pass.

2. **Grid would have taken hours.**
   First benchmark: 2.1 tasks/s, projected ~2 h for 15,840 tasks. LightGBM defaults to `n_jobs=-1`, so each of the 16 worker processes spawned threads for all cores and they fought each other.
   Fix: `LightGBMModel` now defaults `n_jobs=1` (`src/calibmiss/models.py`). A controlled comparison on the same data showed 14.25 s vs 1.59 s for 32 fits. Throughput rose to 5.6 tasks/s and the full grid finished in 24.5 min. Also set `LOKY_MAX_CPU_COUNT` so joblib stops warning about failing to detect physical cores on Windows.

## Warnings (not failures)

- `ConvergenceWarning: [IterativeImputer] Early stopping criterion not reached` fired repeatedly for the `iterative` imputer. It is configured with `max_iter: 10`, which is not enough for these matrices to converge. Imputed values are still produced and all iterative rows completed; results for that imputer are usable but the imputation is under-converged. Raising `max_iter` would slow the slowest cells of the grid further.

## Headline numbers

15,840 experiment rows (22 walk-forward folds x 5 seeds x 2 models x 3 imputers x 3 calibrators x 2 mechanisms x 4 rates) plus 154 baseline rows.

- **Nothing beats the naive base rate.** Predicting the training-fold up-day frequency gives Brier 0.2489, ECE 0.0170, ROC AUC 0.500. The best model configuration (logistic regression, median imputer, Platt, MAR, 30% missing) reaches Brier 0.2488 and ROC AUC 0.516 — a Brier improvement of 0.0001, inside the noise. Discrimination is effectively absent across the whole grid (ROC AUC 0.459 to 0.571).

- **Calibration, not imputation, moves the metrics.** For logistic regression with median imputation under MCAR, going from 0% to 30% missingness changes ECE by -0.002 (0.0234 to 0.0219) and Brier not at all. Adding Platt scaling at 0% missingness improves ECE from 0.0234 to 0.0199 and lowers Brier from 0.2491 to 0.2488. Uncalibrated LightGBM is the worst-calibrated cell in the grid (ECE 0.0407 at 0% missing) and Platt brings it to 0.0194. (Numbers from the 2026-11-13 unweighted rerun; the 2026-11-12 balanced-weighted run showed larger uncalibrated ECE — 0.0397 LR / 0.0512 LightGBM — because balanced weighting pulls probabilities toward 0.5.)

- **507 of 1,728 paired comparisons are significant after Holm correction**, but the effects are small and concentrated on the calibrator contrast: calibrated vs uncalibrated LightGBM differs by about 0.004 to 0.005 Brier. Missingness rate and mechanism (MCAR vs MAR) produce no meaningful degradation anywhere in the grid.

## 2026-11-22 rerun (expanded grid, config-snapshot metadata)

Full direction grid under the expanded 4-imputer / 3-mechanism config (`MCAR`, `MCAR-BLOCK`, `MAR` x 4 rates), `n_jobs=1`, ~290 min on the same machine. Outputs refreshed: 31,680 experiment rows + 154 baseline rows, 4,752 paired comparisons (1,307 significant after Holm; median significant Brier effect 0.0029), 5 figures regenerated.

- Headline claims unchanged: naive Brier 0.2489 / ECE 0.0170; best cells (logistic regression + Platt) Brier 0.2488 / AUC 0.516; LGBM ECE 0.0407 → 0.0194 with Platt; LR ECE 0.0234 → 0.0199.
- AUC span widened to 0.387–0.609; the extremes are drop-imputer cells whose small complete-case test sets are noisy. Grid median 0.510; 973 of 31,680 rows exceed AUC 0.55.
- `run_metadata.json` now records `config_snapshot` + `grid_dimensions`; `scripts/verify_results_config.py` passes against the committed metrics for the first time.
- Earlier-run notes kept above for the historical record: the 0.571/fold-13 permutation check and the class-weighting comparison refer to the 2026-11-13 run.

## 2026-11-22 volatility grid rerun (expanded config)

Full volatility grid under the 3-mechanism config (`MCAR`, `MCAR-BLOCK`, `MAR`), `n_jobs=1`, ~372 min on the same machine. Outputs refreshed: 31,856 rows in `results/volatility/metrics.csv` (first run: 16,016, MCAR+MAR only), 4,752 paired comparisons (1,038 significant after Holm), 5 figures regenerated.

- Signal claims unchanged: best cell AUC 0.729 (LR, forward_fill, none, MAR, 30%); persistence 0.673; naive 0.500; both models average AUC 0.719.
- **Degradation revised upward.** The first run saw ECE 0.0666 → 0.0700 from 0% to 30% MCAR. The rerun measures 0.0666 → 0.0883 (+0.022) under MCAR and +0.016 under MCAR-BLOCK, with AUC falling 0.019 and 0.028 respectively; MAR stays nearly flat (+0.002). The first run's "small effect" summary was an artifact of its missing MCAR-BLOCK slice and lighter imputer set — treat the 2026-11-22 numbers as authoritative.
- Platt still adds distortion rather than removing it on this target (best-cell ECE 0.0785 with vs 0.0711 without).

## 2026-11-25 rerun (five-mechanism grids)

Both grids rerun under the full five-mechanism config (`MCAR`, `MCAR-BLOCK`, `MAR`, `MNAR`, `MCAR-BLOCK-MARKET` x 4 rates), adding MNAR and market-correlated block missingness to the grid, with post-imputation split sizes (`n_train`/`n_cal`/`n_test`) now recorded in `metrics.csv`.

- **Direction grid:** `n_jobs=8`, ~95 min. 52,954 rows in `results/metrics.csv` (52,800 experiment + 154 baseline), 7,920 paired comparisons (2,042 significant after Holm; median significant Brier effect 0.0055, middle 50% 0.0033–0.018), 5 figures regenerated. Headline claims unchanged: naive 0.2489/0.0170; LR Platt 0.0234 → 0.0199; LGBM none 0.0407 → Platt 0.0194; grid AUC median 0.510, 1,499 of 52,800 rows exceed 0.55. The two new mechanisms behave like the original three on this signal-free target (LR-median ECE at 30%: MAR 0.0231, MNAR 0.0235, MARKET 0.0230 vs MCAR 0.0219).
- **Volatility grid:** `n_jobs=4`, ~181 min (two earlier attempts at `n_jobs` 8 and 6 crashed with worker `MemoryError` under Windows memory pressure; 4 workers completed cleanly). 52,976 rows in `results/volatility/metrics.csv` (52,800 experiment + 176 baseline), 7,920 paired comparisons (1,720 significant after Holm; median significant effect 0.0095), 5 figures regenerated. Signal unchanged: persistence 0.673, naive 0.500; models average AUC 0.717. Degradation at 30% (LightGBM, no calibrator): MCAR 0.0666 → 0.0883, MCAR-BLOCK → 0.0822, MCAR-BLOCK-MARKET → 0.0848, MNAR → 0.0746, MAR → 0.0687. Platt and isotonic both raise mean ECE on this target (LightGBM: 0.0721 none vs 0.0845 Platt vs 0.0804 isotonic). At 30% cell-wise MCAR the drop imputer keeps only ~116 of 1,884 test rows per fold (0.7⁸ survival across 8 features) versus ~1,320 under the row-wise mechanisms — the `n_test` column quantifies this selection.
- `scripts/verify_results_config.py` passes for both results directories against their `run_metadata.json` grid dimensions.

## Outputs produced

- `results/metrics.csv` — 52,954 rows (2026-11-25 five-mechanism rerun; earlier: 31,834 from 2026-11-22, 15,994 from 2026-11-13)
- `results/summary.csv` — 480 configuration summaries with mean/std/count
- `results/statistical_tests.csv` — 7,920 paired Wilcoxon tests with Holm-adjusted p-values
- `results/run_metadata.json` — download date, package versions, config snapshot, grid dimensions
- `results/volatility/metrics.csv` — 52,976 rows (2026-11-25 five-mechanism rerun; earlier: 31,856 from 2026-11-22, 16,016 from 2026-11-13)
- `results/volatility/summary.csv` — 480 configuration summaries
- `results/volatility/statistical_tests.csv` — 7,920 paired Wilcoxon tests
- `results/volatility/run_metadata.json` — config snapshot + grid dimensions
- `results/figures/` and `results/volatility/figures/` — 5 figures each: `brier_vs_missingness.png`, `ece_vs_missingness.png`, `heatmap_calibration.png`, `reliability_0pct.png`, `reliability_30pct.png`

## Sanity check: shuffled labels

The highest AUC in the grid, 0.571, is a single fold (fold 13, seed 999, logistic regression, median imputation, MCAR, 10%) and is identical for all three calibrators, so it comes from the base model rather than from calibration. Rerunning five configurations with the target permuted (seed 20261112):

| Config | Real AUC | Shuffled AUC |
|---|---:|---:|
| fold 13, logistic regression, median, MCAR 10% (the 0.571 cell) | 0.5707 | 0.4986 |
| same, Platt | 0.5707 | 0.4986 |
| fold 0, logistic regression, median, Platt, MAR 30% | 0.5092 | 0.5017 |
| fold 5, LightGBM, iterative, MCAR 0% | 0.5345 | 0.5185 |
| fold 10, LightGBM, forward fill, Platt, MAR 20% | 0.4853 | 0.5156 |

Shuffled AUC sits at 0.5. The 0.571 is a fold-level fluctuation, not label leakage. 495 of 15,840 rows exceed 0.55 in the rerun, which is the expected scatter around a true AUC of 0.5.

## Not committed

The two fixes (`scripts/compute_features.py`, `src/calibmiss/run_experiment.py`, `src/calibmiss/models.py`), the regenerated `data/processed/` parquet files, and everything under `results/` are uncommitted.
