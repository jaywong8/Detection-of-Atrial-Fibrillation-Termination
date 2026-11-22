#!/usr/bin/env python3
"""Patient-level bootstrap precision and approximate detectable-difference check.

Run on validation predictions during development. Do not run on test predictions
until the final locked evaluation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def load(pred_path, windows_path, name):
    w=pd.read_csv(windows_path)[["window_id","patient","label"]]
    p=pd.read_csv(pred_path)[["window_id","probability"]].rename(columns={"probability":name})
    if p.window_id.duplicated().any(): raise ValueError(f"{pred_path}: duplicate window_id")
    d=w.merge(p,on="window_id",how="left",validate="one_to_one")
    if d[name].isna().any(): raise ValueError(f"{pred_path}: missing predictions for {d[name].isna().sum()} windows")
    if not d[name].between(0,1).all(): raise ValueError(f"{pred_path}: probabilities must be in [0,1]")
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows",type=Path,required=True)
    ap.add_argument("--pred",action="append",required=True,
                    help="MODEL_NAME=prediction.csv; repeat for each baseline")
    ap.add_argument("--n-boot",type=int,default=5000)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--out",type=Path,default=Path("results/power_precision_validation.json"))
    args=ap.parse_args()
    d=pd.read_csv(args.windows)[["window_id","patient","label"]]
    names=[]
    for item in args.pred:
        if "=" not in item: raise ValueError("--pred must be MODEL_NAME=path/to/pred.csv")
        name,path=item.split("=",1); p=pd.read_csv(path)[["window_id","probability"]]
        p=p.rename(columns={"probability":name})
        if p.window_id.duplicated().any(): raise ValueError(f"{path}: duplicate window IDs")
        d=d.merge(p,on="window_id",how="left",validate="one_to_one")
        if d[name].isna().any(): raise ValueError(f"{name}: missing predictions for {d[name].isna().sum()} windows")
        names.append(name)
    if len(names)<2: raise ValueError("Provide at least two models to estimate detectable paired differences.")
    patients=list(d.groupby("patient").indices)
    idx=d.groupby("patient").indices
    y=d.label.to_numpy()
    rng=np.random.default_rng(args.seed)
    metrics={n:[] for n in names}
    diffs={}
    for i,a in enumerate(names):
        for b in names[i+1:]: diffs[f"{a}_minus_{b}"]=[]
    for _ in range(args.n_boot):
        chosen=rng.integers(0,len(patients),len(patients))
        ix=np.concatenate([idx[patients[j]] for j in chosen])
        if np.unique(y[ix]).size<2: continue
        current={}
        for n in names:
            p=d[n].to_numpy()[ix]
            current[n]=(roc_auc_score(y[ix],p),average_precision_score(y[ix],p))
            metrics[n].append(current[n])
        for i,a in enumerate(names):
            for b in names[i+1:]:
                diffs[f"{a}_minus_{b}"].append(np.subtract(current[a],current[b]))
    report={"split":args.windows.stem,"n_windows":len(d),"n_patients":d.patient.nunique(),
            "positive_prevalence":float(y.mean()),"bootstrap_replicates_used":min(map(len,metrics.values())),
            "models":{},"paired_differences":{}}
    for n,vals in metrics.items():
        arr=np.asarray(vals)
        report["models"][n]={}
        for j,m in enumerate(["ROC-AUC","PR-AUC"]):
            report["models"][n][m]={"bootstrap_CI_95":[float(x) for x in np.percentile(arr[:,j],[2.5,97.5])],
                                    "CI_width":float(np.diff(np.percentile(arr[:,j],[2.5,97.5]))[0]),
                                    "bootstrap_SD":float(arr[:,j].std(ddof=1))}
    for n,vals in diffs.items():
        arr=np.asarray(vals)
        report["paired_differences"][n]={}
        for j,m in enumerate(["ROC-AUC","PR-AUC"]):
            ci=np.percentile(arr[:,j],[2.5,97.5]); sd=arr[:,j].std(ddof=1)
            # Approximate 80%-power MDE for a two-sided 5% test, using paired-bootstrap SD.
            report["paired_differences"][n][m]={"observed_difference":float(np.mean(arr[:,j])),
                "bootstrap_CI_95":[float(ci[0]),float(ci[1])],"CI_width":float(ci[1]-ci[0]),
                "approx_80pct_power_MDE":float((1.96+0.84)*sd)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(__import__("json").dumps(report,indent=2))
    print(__import__("json").dumps(report,indent=2))
    print("\nInterpretation: MDE is an approximate paired detectable difference, not a formal prospective power calculation.")

if __name__=="__main__": main()
