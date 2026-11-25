"""Extract headline numbers from a fresh results dir for doc updates."""

import sys
from pathlib import Path

import pandas as pd


def headline(results_dir: str) -> dict:
    d = Path(results_dir)
    m = pd.read_csv(d / "metrics.csv")
    exp = m[m["row_type"] == "experiment"]
    base = m[m["row_type"] == "baseline"]
    out = {"n_experiment_rows": len(exp), "n_total_rows": len(m)}
    naive = base[base["baseline"] == "naive_training_base_rate"]
    out["naive_brier"] = round(naive["brier"].mean(), 4)
    out["naive_ece"] = round(naive["ece"].mean(), 4)
    lr = exp[(exp.model == "logistic_regression") & (exp.imputer == "median")]
    for cal, key in (("none", "lr_none"), ("platt", "lr_platt")):
        sel = lr[(lr.calibrator == cal) & (lr.mechanism == "MCAR") & (lr.rate == 0.0)]
        out[key] = {"brier": round(sel["brier"].mean(), 4),
                    "ece": round(sel["ece"].mean(), 4),
                    "auc": round(sel["roc_auc"].mean(), 4)}
    lgbm = exp[(exp.model == "lightgbm") & (exp.mechanism == "MCAR") & (exp.rate == 0.0)]
    for cal in ("none", "platt"):
        out[f"lgbm_{cal}_ece"] = round(lgbm[lgbm.calibrator == cal]["ece"].mean(), 4)
    out["auc_min"] = round(exp["roc_auc"].min(), 3)
    out["auc_max"] = round(exp["roc_auc"].max(), 3)
    out["auc_median"] = round(exp["roc_auc"].median(), 3)
    out["n_auc_gt_055"] = int((exp["roc_auc"] > 0.55).sum())
    st = pd.read_csv(d / "statistical_tests.csv")
    out["n_comparisons"] = len(st)
    out["n_significant_holm"] = int((st["p_holm"] < 0.05).sum())
    sig = st[st["p_holm"] < 0.05]["mean_paired_difference"].abs()
    out["median_sig_effect"] = round(float(sig.median()), 4)
    if "n_test" in exp.columns:
        drop = exp[(exp.imputer == "drop") & (exp.mechanism == "MCAR") & (exp.rate == 0.3)]
        other = exp[(exp.imputer == "median") & (exp.mechanism == "MCAR") & (exp.rate == 0.3)]
        out["drop_n_test_at_30"] = int(drop["n_test"].mean())
        out["median_n_test_at_30"] = int(other["n_test"].mean())
    return out

if __name__ == "__main__":
    import json
    print(json.dumps(headline(sys.argv[1] if len(sys.argv) > 1 else "results"), indent=1))
