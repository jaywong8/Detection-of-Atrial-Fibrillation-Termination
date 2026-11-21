"""Shared evaluation for AF-termination models.

Every model (feature-based or CNN) writes a CSV with columns `window_id, probability`.
This script scores it against a frozen windows CSV (needs `window_id, patient, label`)
and reports point metrics with patient-level bootstrap confidence intervals.

Threshold rules
---------------
* --pick-threshold SENS chooses a threshold on the scored data, so it is only allowed
  with --split validation (each model gets its own threshold).
* On --split test you must pass --threshold explicitly (the value frozen on validation).

Examples
--------
# validation: pick the threshold for 90% sensitivity and log the model to the table
python evaluate.py results/preds/logreg_val.csv data/10s-30s/validation_windows.csv \
    --split validation --pick-threshold 0.9 --name logreg --table results/results_table.csv

# test: apply the frozen threshold (B only, once, at the end)
python evaluate.py results/preds/logreg_test.csv data/10s-30s/test_windows.csv \
    --split test --threshold 0.31 --name logreg --table results/results_table.csv

# paired comparison: is A better than B? (both models are logged to the table)
python evaluate.py results/preds/cnn_val.csv data/10s-30s/validation_windows.csv \
    --split validation --pick-threshold 0.9 --name cnn \
    --compare results/preds/logreg_val.csv --name-b logreg --table results/results_table.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def load(pred_csv, windows_csv, col, min_coverage=0.99):
    w = pd.read_csv(windows_csv)
    for c in ("window_id", "patient", "label"):
        assert c in w.columns, f"{windows_csv} is missing column '{c}'"
    assert w["window_id"].is_unique, "duplicate window_ids in windows CSV"
    assert w["patient"].notna().all(), "null values in the patient column"
    assert w["label"].isin([0, 1]).all(), "label must be 0/1 with no nulls"
    p = pd.read_csv(pred_csv)
    for c in ("window_id", "probability"):
        assert c in p.columns, f"{pred_csv} is missing column '{c}'"
    p = p[["window_id", "probability"]].rename(columns={"probability": col})
    assert p["window_id"].is_unique, f"duplicate window_ids in {pred_csv}"
    assert p[col].notna().all(), f"NaN probabilities in {pred_csv}"
    assert p[col].between(0, 1).all(), f"probabilities outside [0, 1] in {pred_csv}"
    extra = set(p["window_id"]) - set(w["window_id"])
    assert not extra, f"{len(extra)} predicted window_ids are not in the windows CSV"
    d = w[["window_id", "patient", "label"]].merge(p, on="window_id", how="inner")
    cov = len(d) / len(w)
    print(f"{pred_csv}: scored {len(d)}/{len(w)} windows ({cov:.1%})")
    assert cov >= min_coverage, f"coverage {cov:.1%} below {min_coverage:.0%}; windows are missing predictions"
    assert d["label"].nunique() == 2, "need both classes in the scored windows"
    return d


def sens_at_spec(y, p, spec):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[np.searchsorted(fpr, 1 - spec, side="right") - 1])


def threshold_at_sens(y, p, target):
    """Highest threshold reaching the target sensitivity. Choose on validation only."""
    _, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr >= target)])


def at_threshold(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "sensitivity": round(tp / (tp + fn), 3),
        "specificity": round(tn / (tn + fp), 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
    }


def bootstrap(d, cols, n_boot=1000, seed=0):
    """Resample patients with replacement. All models in `cols` share each resample (paired)."""
    rng = np.random.default_rng(seed)
    idx = d.groupby("patient").indices  # patient -> row positions
    pats = list(idx)
    y = d["label"].values
    P = {c: d[c].values for c in cols}
    out = {c: [] for c in cols}
    for _ in range(n_boot):
        pick = rng.integers(0, len(pats), len(pats))
        rows = np.concatenate([idx[pats[i]] for i in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        for c in cols:
            out[c].append((roc_auc_score(y[rows], P[c][rows]),
                           average_precision_score(y[rows], P[c][rows])))
    return {c: np.array(v) for c, v in out.items()}


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def summarize(d, col, boot, thr):
    y, p = d["label"].values, d[col].values
    return {
        "threshold": round(float(thr), 4),
        "ROC-AUC": round(float(roc_auc_score(y, p)), 3), "ROC-AUC_CI": ci(boot[:, 0]),
        "PR-AUC": round(float(average_precision_score(y, p)), 3), "PR-AUC_CI": ci(boot[:, 1]),
        "PR-AUC_chance": round(float(y.mean()), 3),
        "sens@spec90": round(sens_at_spec(y, p, 0.90), 3),
        "sens@spec80": round(sens_at_spec(y, p, 0.80), 3),
        **at_threshold(y, p, thr),
    }


def append_table(path, name, res, key):
    r = res[key]
    row = {"name": name, "split": res["split"], "n_windows": res["n_windows"],
           "n_patients": res["n_patients"], "prevalence": res["prevalence"],
           "threshold": r["threshold"],
           "ROC-AUC": r["ROC-AUC"], "ROC-AUC_lo": r["ROC-AUC_CI"][0], "ROC-AUC_hi": r["ROC-AUC_CI"][1],
           "PR-AUC": r["PR-AUC"], "PR-AUC_lo": r["PR-AUC_CI"][0], "PR-AUC_hi": r["PR-AUC_CI"][1],
           "PR-AUC_chance": r["PR-AUC_chance"], "sens@spec90": r["sens@spec90"],
           "sens@spec80": r["sens@spec80"], "sensitivity": r["sensitivity"],
           "specificity": r["specificity"], "precision": r["precision"]}
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", help="predictions CSV: window_id, probability")
    ap.add_argument("windows", help="frozen windows CSV: window_id, patient, label")
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"],
                    help="which split these windows are (guards threshold selection)")
    ap.add_argument("--compare", help="second predictions CSV for a paired comparison")
    ap.add_argument("--threshold", type=float,
                    help="fixed decision threshold (required on test; default 0.5 elsewhere)")
    ap.add_argument("--pick-threshold", type=float, metavar="SENS",
                    help="choose the threshold that reaches this sensitivity (validation only)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument("--name", help="name of the main model for the results table")
    ap.add_argument("--name-b", help="name of the --compare model for the results table")
    ap.add_argument("--table", help="append one row per model to this CSV")
    ap.add_argument("--out", help="write full results JSON here")
    a = ap.parse_args()

    if a.pick_threshold is not None and a.split != "validation":
        ap.error("--pick-threshold is only allowed with --split validation; "
                 "on test, pass the threshold frozen on validation via --threshold")
    if a.pick_threshold is not None and a.threshold is not None:
        ap.error("use either --pick-threshold or --threshold, not both")
    if a.split == "test" and a.threshold is None:
        ap.error("--split test requires an explicit --threshold (the value frozen on validation)")

    d = load(a.preds, a.windows, "p_a", a.min_coverage)
    cols = ["p_a"]
    if a.compare:
        b = load(a.compare, a.windows, "p_b", a.min_coverage)[["window_id", "p_b"]]
        d = d.merge(b, on="window_id", how="inner")
        cols.append("p_b")
        print(f"windows scored by both models: {len(d)}")

    thr = {c: (a.threshold if a.threshold is not None else 0.5) for c in cols}
    if a.pick_threshold is not None:
        for c in cols:
            thr[c] = threshold_at_sens(d["label"].values, d[c].values, a.pick_threshold)
            print(f"{c}: threshold for sensitivity {a.pick_threshold}: {thr[c]:.4f}  (freeze this for test)")

    boots = bootstrap(d, cols, a.n_boot, a.seed)
    res = {"split": a.split, "n_windows": len(d), "n_patients": int(d["patient"].nunique()),
           "prevalence": round(float(d["label"].mean()), 3),
           "n_boot_used": int(len(boots["p_a"]))}
    for c in cols:
        res[c] = summarize(d, c, boots[c], thr[c])
    if len(cols) == 2:
        delta = boots["p_a"] - boots["p_b"]
        res["a_minus_b"] = {
            "ROC-AUC_diff": round(res["p_a"]["ROC-AUC"] - res["p_b"]["ROC-AUC"], 3),
            "ROC-AUC_diff_CI": ci(delta[:, 0]),
            "PR-AUC_diff": round(res["p_a"]["PR-AUC"] - res["p_b"]["PR-AUC"], 3),
            "PR-AUC_diff_CI": ci(delta[:, 1]),
            "frac_resamples_a_better_ROC": round(float((delta[:, 0] > 0).mean()), 3),
        }
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    if a.table:
        append_table(a.table, a.name or os.path.basename(a.preds), res, "p_a")
        if a.compare:
            append_table(a.table, a.name_b or os.path.basename(a.compare), res, "p_b")


if __name__ == "__main__":
    main()
