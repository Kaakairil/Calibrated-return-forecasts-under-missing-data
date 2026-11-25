# CalibMiss Executive Summary

## Purpose

CalibMiss tests whether next-day stock-direction probabilities stay calibrated when engineered input features contain missing values. It compares five missingness mechanisms (MCAR, MCAR-BLOCK, MAR, MNAR, MCAR-BLOCK-MARKET), four imputers, two model families, and three calibration choices using chronological walk-forward evaluation.

## What the pipeline measures

- **Brier score:** squared probability error; lower is better.
- **ECE:** ten-bin expected calibration error; lower is better.
- **Reliability diagrams:** observed frequency versus predicted probability.
- **ROC AUC:** secondary discrimination measure.
- **Stability:** fold and seed dispersion, paired tests, signed-rank effects, and Holm-adjusted p-values.

The pipeline includes complete-data oracle and naive training-prevalence baselines. Imputers fit on training data only; calibrators fit on a separate calibration split; final metrics use a held-out test split.

## Findings

The full grid (22 folds, 5 seeds, 52,800 runs across five missingness mechanisms: MCAR, MCAR-BLOCK, MAR, MNAR, MCAR-BLOCK-MARKET) shows:

- **No model beats the naive base rate.** Training-fold up-day frequency scores Brier 0.2489 and ECE 0.0170. The best model cells reach Brier 0.2488 and ROC AUC 0.516. Next-day direction is effectively unpredictable from these features, which is what efficient-market intuition predicts. This is a calibration result, not evidence that the models predict returns.
- **Calibration fixes overconfidence, not predictions.** Platt scaling improves logistic-regression ECE from 0.0234 to 0.0199 (~15%) and cuts uncalibrated LightGBM ECE from 0.0407 to 0.0194, with ROC AUC unchanged. (Models run without class weighting; a balanced-weighted earlier run inflated the uncalibrated ECE.) The significant paired tests are calibrator contrasts with small Brier effects (median 0.0055; 2,042 of 7,920 comparisons significant after Holm).
- **Missingness is measured, not assumed, on five mechanisms.** The grid simulates MCAR (cell-by-cell), **MCAR-BLOCK** (whole consecutive-day streaks per ticker), MAR (conditional on volume ratio), **MNAR** (masking driven by the feature's own extreme values), and **MCAR-BLOCK-MARKET** (market-wide outage days shared by every ticker), each at 0/10/20/30%, with a complete-case **drop** baseline alongside forward-fill, median, and MICE. The flat direction curve is a property of the target's lack of signal; the new mechanisms behave like the original three there (LR-median ECE at 30%: MAR 0.0231, MNAR 0.0235, MARKET 0.0230).
- **On the direction target, missingness looks harmless only because there is no signal to lose.** ECE moves ~0.001 from 0% to 30% missingness under both MCAR and MAR, but the model has no accuracy for missingness to damage.
- **On a target with signal, missingness damage is measurable and mechanism-dependent.** A second grid on realized volatility (binarized at the training-fold median, 60-day purge; 52,800 runs, same five mechanisms as the direction grid, rerun 2026-11-25) reaches ROC AUC ~0.72 versus 0.67 for persistence and 0.50 naive, then loses ECE 0.0666 → 0.0883 (+0.022) and AUC 0.728 → 0.709 (−0.019) from 0% to 30% MCAR missingness; the market-wide outage mechanism costs 0.018 ECE, MCAR-BLOCK 0.016, MNAR 0.008, and MAR is nearly flat (+0.002). Calibrators do not recover it — Platt and isotonic both *raise* mean ECE on this target — because the model's probabilities are already well ranked.

## Scope limits on the statistics

Walk-forward folds overlap in time and seeds share the same data, so the paired tests treat dependent observations as independent and the Holm-corrected p-values overstate the evidence. The universe was selected from 2026 large caps with hindsight (survivorship bias). The direction grid runs without class weighting; the volatility grid explicitly keeps `class_weight: "balanced"`. The iterative (MICE) imputer is configured at `max_iter: 50` and converges on these matrices; the complete-case drop baseline removes missing rows by construction, so its test set differs from the other imputers'.

## Reproduce

```bash
pip install -r requirements.txt
make data
make features
make experiment
make figures
pytest -v
```

## Scope limits

The study uses daily data for the configured U.S. equity universe, simulated missingness mechanisms, logistic regression and LightGBM, and a finite historical period. Calibration findings do not demonstrate trading profitability or live deployment suitability.
