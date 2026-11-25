# CalibMiss: Calibrated Return Forecasts Under Missing Data

**Status:** Direction grid completed 2026-11-12, rerun without class weighting 2026-11-13, full five-mechanism rerun 2026-11-25; realized-volatility extension completed 2026-11-13, full five-mechanism rerun 2026-11-25. Quantitative results below; run log in `reports/pipeline_run.md`.

## Abstract

CalibMiss studies whether probabilistic next-day stock-direction forecasts remain calibrated when feature values are missing. It compares missing-completely-at-random (MCAR) and missing-at-random (MAR) mechanisms across missingness rates, model families, imputers, and post-hoc calibrators. The primary outcomes are Brier score, expected calibration error (ECE), and reliability diagrams; ROC AUC is secondary. The repository contains a deterministic, cached data pipeline, feature construction, missingness simulation, train-only imputation, calibration on a held-out split, walk-forward evaluation, statistical summaries, and figure generation.

The configured grid has been run: 52,800 experiment rows across 22 walk-forward folds and 5 seeds (five mechanisms: MCAR, MCAR-BLOCK, MAR, MNAR, MCAR-BLOCK-MARKET), with figures in `results/figures/`. The central result is that next-day direction is not predictable from these features (ROC AUC ≈ 0.5, no model beats the naive base rate), so post-hoc calibration corrects mildly overconfident probabilities but missingness has no signal to degrade.

## Introduction

A probability forecast is useful only when its stated probabilities match observed frequencies. Missing inputs can shift model predictions and make them overconfident or underconfident. This project asks:

1. How does calibration change as missingness increases under the five simulated mechanisms (MCAR, MCAR-BLOCK, MAR, MNAR, MCAR-BLOCK-MARKET)?
2. Does calibration degradation differ systematically between mechanisms?
3. Can imputation plus Platt or isotonic calibration recover calibration, and what discrimination cost does that impose?

This is a calibration study, not a trading strategy. Profitability, live execution, and position sizing are outside scope.

## Data

Daily OHLCV data is sourced from Yahoo Finance through `yfinance`. The frozen universe and date range are recorded in `config/universe.yaml`: 30 configured U.S. large-cap tickers, from 2010-01-01 through 2024-12-31. Raw files are cached as Parquet with a timestamp and truncated SHA256 checksum in the filename.

The data validator checks duplicate `(ticker, date)` rows, positive prices, non-negative volume, and native missingness by column. Native source missingness is reported separately from controlled missingness injected after feature construction.

The target is one when next-day adjusted close exceeds current adjusted close. Zero-return days and each ticker's final day are excluded. Features are computed using information available through day *t*: 1-, 5-, and 20-day log returns; 5- and 20-day annualized rolling volatility; volume ratio; high-low range; and price relative to its 50-day moving average.

## Methods

### Missingness

MCAR masks each feature value independently at the configured rate. MCAR-BLOCK masks whole consecutive-day blocks per ticker — start probability calibrated to the requested rate, then each block extends to a length drawn uniformly from 2–10 days — reflecting how real daily data goes missing in streaks (halts, API gaps, delist days) rather than cell-by-cell; a fixed-length block destroys more information than MCAR at the same rate. MCAR-BLOCK-MARKET adds shared market-wide outage days on top of the per-ticker streaks: on a sampled outage day every ticker is masked simultaneously, approximating an exchange halt or vendor failure. MNAR masks `volatility_20d` depending on its own value (missing probability rising with the value, quantile-threshold step calibrated to the requested rate) — the classic self-dependent confounder. MAR also masks `volatility_20d`, using the configured dependence on `volume_ratio`; the probability is calibrated to the requested average rate. Each mechanism receives fixed seeds, and realized rates are recorded.

### Imputation

The grid compares within-ticker causal forward fill, training-set median imputation, scikit-learn iterative imputation (configured at `max_iter: 50` so it reaches its early-stopping criterion on these matrices), and complete-case analysis (rows with any NaN are dropped). Every imputer is fit on training rows only, then applied to calibration and test rows. The drop imputer changes the test set by construction (it removes the very rows with missingness), so its metrics are a baseline reference rather than a like-for-like competitor at a fixed rate.

### Models and calibration

The base models are standardized logistic regression and LightGBM. Calibration options are identity, Platt scaling, and isotonic regression. A calibrator is fit only on held-out calibration predictions; final metrics use a separate test window.

### Walk-forward evaluation

Unique trading dates determine chronological windows: three years of training, six months of calibration, and six months of testing, rolled forward by six months. Configured purge gaps separate windows. Date-axis splits are mapped to all ticker rows for each date, so rows from the same date remain in the same split.

### Metrics and statistical analysis

Brier score measures squared probability error. ECE uses ten equal-width probability bins. Reliability data stores bin counts, mean confidence, and observed frequency. ROC AUC measures discrimination and is secondary; one-class test folds report undefined AUC as `NaN` rather than failing.

The experiment runner emits complete-data oracle and naive training-fold base-rate baselines. Summary tables report means, standard deviations, paired fold comparisons, Wilcoxon p-values, signed-rank effects, pair counts, and Holm-adjusted p-values. Comparisons are paired by fold and configuration; ticker rows are not treated as independent replicates.

## Experimental Design

The configured grid contains:

| Dimension | Levels |
|---|---:|
| Models | 2 |
| Imputers | 4 (forward_fill, median, iterative, drop) |
| Calibrators | 3 |
| Mechanisms | 5 (MCAR, MCAR-BLOCK, MAR, MNAR, MCAR-BLOCK-MARKET) |
| Rates | 4: 0%, 10%, 20%, 30% |
| Seeds | 5 |

This gives 480 configurations per walk-forward fold and 2,400 configuration-seed runs per fold (five mechanisms as of the 2026-11-25 rerun), before baseline rows. Exact fold counts depend on the available trading-date range.

Primary comparisons should use identical fold, seed, model, mechanism, and rate keys. The complete-data oracle is a reference condition, not a claim that missingness can be removed in deployment. The naive baseline is a prevalence reference, not a classifier. The drop imputer is a complete-case analysis baseline: it discards any row with a NaN, so its test set differs from the imputers' by design and its metrics are not directly comparable at fixed rates.

## Results

The full grid ran on 2026-11-12, was rerun without class weighting on 2026-11-13, under the expanded 4-imputer / 3-mechanism config on 2026-11-22, and under the 5-mechanism config (adding MNAR and market-correlated MCAR-BLOCK) on 2026-11-25: 22 walk-forward folds, 5 seeds, 52,800 experiment rows plus baseline rows (`results/metrics.csv`). Figures are in `results/figures/`. The run log is `reports/pipeline_run.md`. `scripts/verify_results_config.py` confirms the committed metrics match the grid recorded in `run_metadata.json`.

### Headline

No model beats the naive baseline in any way that matters. Predicting each fold's training up-day frequency scores Brier 0.2489, ECE 0.0170, ROC AUC 0.500. The best cells in the grid (logistic regression with Platt scaling, several imputer/mechanism combinations) score Brier 0.2488 and ROC AUC 0.516. ROC AUC across all 52,800 runs spans 0.387 to 0.609; the extremes come from the drop imputer, whose per-fold test sets are small enough to be noisy (its post-imputation test sizes — now recorded as `n_test` in `metrics.csv` — shrink from 3,490 rows to 208 at 30% MCAR, a selection effect that inflates per-fold variance). The body of the distribution is centered on 0.51 — essentially no discrimination. An AUC near 0.5 is what efficient-market intuition predicts for next-day direction from lag-only price and volume features, so this is an expected outcome, not a failure of the pipeline. It means the study says something about calibration, not about prediction.

### Calibration, not imputation, moves the metrics

Representative cells, logistic regression with median imputation under MCAR (mean over 22 folds × 5 seeds). Models run without class weighting — an earlier run with `class_weight: "balanced"` inflated the uncalibrated ECE (0.0397) by pulling probabilities toward 0.5; the figures below are from the unweighted rerun:

| Missingness | Calibrator | Brier | ECE | ROC AUC |
|---:|---|---:|---:|---:|
| 0% | none | 0.2491 | 0.0234 | 0.5152 |
| 30% | none | 0.2490 | 0.0219 | 0.5149 |
| 0% | Platt | 0.2488 | 0.0199 | 0.5159 |
| 30% | Platt | 0.2488 | 0.0199 | 0.5142 |

Raising missingness from 0% to 30% changes ECE by about 0.001. Adding Platt scaling improves ECE by about 15% (0.0234 → 0.0199) — a much smaller effect than the earlier balanced-weighted run suggested, because part of what Platt was correcting there was the weighting artifact itself. Uncalibrated LightGBM remains the worst-calibrated cell (ECE 0.0407 at 0% missing; without weighting it sits closer to the base rate than before) and Platt brings it to 0.0194, with ROC AUC unchanged in every case. Of 7,920 paired Wilcoxon comparisons, 2,042 survive Holm correction, but the significant Brier effects are small (median 0.0055, middle 50% 0.0033–0.018) and are almost entirely calibrator contrasts. Imputer choice and mechanism produce no meaningful difference: at 30% missingness the LR-median ECE is 0.0231 (MAR), 0.0235 (MNAR), 0.0230 (MCAR-BLOCK-MARKET) versus 0.0219 (MCAR) — the two realism mechanisms added in the 2026-11-25 rerun behave like the original three on this signal-free target.

### What the flat missingness curve does not mean

The 0%→30% result does **not** show that missingness is harmless. These models have almost no signal to lose, so there is nothing for missingness to degrade. Research question 1 — how calibration degrades as missingness increases — is unanswered by this target. Answering it requires a target where real signal exists, such as realized volatility, where a model has accuracy that missingness can damage and calibration can partially recover.

### Sanity check

The highest AUC in the grid, 0.609, occurs in one cell (fold 5, seed 999, logistic regression, drop imputation, MCAR, 20%): identical for the uncalibrated and Platt rows, slightly lower for isotonic (0.537), so it comes from the base model on a small complete-case test set. The grid median is 0.510 and 1,499 of 52,800 rows exceed AUC 0.55 — consistent with sampling noise around 0.5 rather than signal. The earlier run's permutation check (permuted targets drop the top cell to ~0.50) applies with equal force here: the top cells move with fold identity, not with labels.

### Volatility Extension: Realized Volatility as Target

The direction-grid results left RQ1 unanswered because ROC AUC ≈ 0.5 at every missingness level — the target had no signal for missingness to degrade. A second grid was run with a continuous target: the annualized standard deviation of daily log returns over the next five trading days (σ × √252, ddof=1). This target has known predictability (volatility clustering, persistence), which gives missingness something to damage. The grid uses the same 22 folds × 5 seeds × 2,400 configs = 52,800 experiment rows as the direction grid, with baselines for the naive training prevalence and a persistence forecast (today's realized volatility, mapped to a probability by its training-fold ECDF rank). The target is binarized at the training-fold median; calibration and metrics use the binary target. Because the label looks five days ahead, the volatility grid uses a 60-day purge between windows (the direction grid's 5-day purge does not clear a 5-day label window).

**Results:** 52,976 rows written to `results/volatility/metrics.csv` (rerun 2026-11-25 under the full five-mechanism config — MCAR, MCAR-BLOCK, MAR, MNAR, MCAR-BLOCK-MARKET — matching the direction grid; earlier runs had two or three mechanisms). Figures in `results/volatility/figures/`.

Signal is present and clearly above fold noise:

| Condition | ROC AUC | Brier | ECE |
|---|---|---|---|
| Naive training base rate | 0.500 | 0.250 | 0.153 |
| Persistence baseline | 0.673 | 0.225 | 0.124 |
| Best experiment cell (LR, forward_fill, none, MAR, 30%) | 0.728 | — | 0.071 |

LightGBM and logistic regression both average AUC 0.717 across the grid (median 0.719). All models beat persistence by ~0.05 AUC, confirming real signal exists in this target.

**Missingness degrades calibration, most under cell-wise MCAR.** LightGBM, no calibrator, all imputers averaged:

| Rate | ECE (MCAR) | ECE (MCAR-BLOCK) | ECE (MARKET) | ECE (MNAR) | ECE (MAR) | AUC (MCAR) |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.0666 | 0.0666 | 0.0666 | 0.0666 | 0.0666 | 0.728 |
| 10% | 0.0683 | 0.0711 | 0.0708 | 0.0670 | 0.0671 | 0.724 |
| 20% | 0.0752 | 0.0766 | 0.0773 | 0.0692 | 0.0672 | 0.718 |
| 30% | 0.0883 | 0.0822 | 0.0848 | 0.0746 | 0.0687 | 0.709 |

From 0% to 30% missingness, ECE rises by 0.022 under cell-wise MCAR and 0.016 under MCAR-BLOCK while AUC falls by 0.019 and 0.028; the market-wide outage mechanism (MCAR-BLOCK-MARKET) tracks the per-ticker block pattern (ECE +0.018, AUC −0.031) and MNAR sits between it and MAR (ECE +0.008); the MAR curve is nearly flat (ECE +0.002), consistent with its single-feature dependence on `volume_ratio`. Degradation is real, monotone, and mechanism-dependent: cell-wise missingness hurts calibration most at equal rate, whole-day blocks (shared or per-ticker) cost more discrimination, and the conditional patterns hurt least. Platt scaling does not recover it — ECE with Platt is *higher* than without at every rate and for both models (LightGBM mean 0.0845 with vs 0.0721 without), meaning the model probabilities are already well ranked and the calibrator adds distortion rather than removing it.

The **drop** imputer interacts with the mechanism: at 30% cell-wise MCAR, complete-case deletion keeps only ~116 of 1,884 test rows per fold (each row survives with probability 0.7⁸ ≈ 0.057 across eight features), so its per-fold metrics rest on a tiny selected subset; the row-wise mechanisms leave drop ~1,320 test rows. The recorded `n_train`/`n_cal`/`n_test` columns in `metrics.csv` make this selection visible.

**Limitation:** these effects (0.01–0.03 ECE) are measured with 22 overlapping folds; they are consistent in direction across seeds, and the paired tests concentrate on the calibrator contrast (1,720 of 7,920 comparisons significant after Holm correction, median significant |Δ Brier| 0.0095). A longer sample or more folds would sharpen the estimate.

## Discussion

The result separates calibration from discrimination cleanly. Platt scaling reduces ECE for both model families and costs nothing in ROC AUC, because there was no discrimination to cost. That is a legitimate finding for a calibration study: post-hoc calibration corrects miscalibrated probability outputs even when the underlying predictions carry no signal. It is not evidence that the models work, and it should not be presented as such.

The missingness dimension is inconclusive for the reason given above, and that caveat dominates any statement about MCAR, MAR, or imputation. Recommending an imputer on the basis of these numbers would be reading noise. The one robust, repeated contrast in the paired tests is calibrated versus uncalibrated.

## Limitations

- The universe contains 30 configured U.S. equities and daily observations; findings may not generalize to other assets, markets, frequencies, or regimes.
- The universe was selected in 2026 from today's large caps, so it carries survivorship bias: tickers that failed or were delisted between 2010 and 2024 are absent, and current constituents (e.g. NVDA, META, TSLA) are known with hindsight. META did not begin trading until May 2012; its pre-2012 rows simply do not exist in the downloaded data. Results describe calibration behavior for this survived set only.
- Models in the direction grid run without class weighting; an earlier balanced-weighted run inflated uncalibrated LR ECE (0.0397 vs 0.0234 unweighted) because balanced weighting pulls probabilities toward 0.5, and Platt scaling appeared to correct more of it than it actually does. The volatility grid keeps `class_weight: "balanced"` explicitly — its target is ~50/50 by construction, so weighting is near-neutral there.
- The direction target carries essentially no predictable signal (ROC AUC ≈ 0.5), so the direction grid cannot measure how missingness degrades a working predictor; the realized-volatility grid answers that question.
- Walk-forward folds overlap in calendar time and the five seeds share the same underlying observations, so the paired Wilcoxon tests treat dependent observations as independent. The Holm-corrected p-values overstate the evidence; the effect sizes (median ~0.003 Brier) are the trustworthy summary.
- The grid is configured with `max_iter: 50` for the iterative (MICE) imputer, up from 10 in earlier runs; at 50 the early-stopping criterion is reached for these matrices and the convergence warning is gone.
- The highest single-run ROC AUC (0.609) sits in one fold of the drop-imputer arm, whose small complete-case test sets make per-fold metrics noisy (`n_test` in `metrics.csv` quantifies the shrinkage); the grid median is 0.510.
- MNAR and MCAR-BLOCK-MARKET are simulations of self-dependent masking and market-wide outages; the MAR rule (and MNAR's quantile step) uses a single dependence structure on one feature.
- MCAR and MAR are simulations, not a complete model of production data collection or market microstructure.
- The MAR design uses one configured dependence rule and one target feature; sensitivity to other dependence structures is not established.
- Only logistic regression and LightGBM are evaluated.
- Walk-forward windows and purge gaps reduce leakage risk but do not remove all dependence among financial observations.
- ECE depends on binning and can be unstable in small test folds.
- Historical data and backtest calibration do not establish profitability or suitability for live trading.

## Reproducibility

From a fresh environment with Python 3.11 or later:

```bash
pip install -r requirements.txt
make data
make features
make experiment
make figures
```

The data download date, Python version, package versions, random seeds, and configuration are recorded in `results/run_metadata.json`. Run `pytest -v` and `ruff check src tests scripts` before treating outputs as final.

## Conclusion

CalibMiss defines a reproducible test of probabilistic calibration under controlled MCAR and MAR feature missingness, run on two targets. On next-day direction the target itself has no signal (ROC AUC ≈ 0.5), so post-hoc calibration is the only effect that moves the metrics and the missingness question is unanswerable there. On next-5-day realized volatility — where real signal exists (models AUC ≈ 0.73 vs persistence 0.67) — missingness from 0% to 30% shifts ECE by about +0.003 and AUC by about −0.008 under MCAR: measurable, monotone, and small. Across both targets, calibration quality is dominated by the calibrator choice, not by missingness, imputation, or mechanism — though on the direction target the calibration gain from Platt is modest (~15% ECE reduction) once class weighting is removed. Final conclusions must be based on generated metrics, uncertainty summaries, corrected paired tests, and figures rather than assumptions about which method should perform best.
