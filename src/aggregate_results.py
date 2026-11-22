#!/usr/bin/env python3
"""Combine evaluate.py JSON/CSV outputs and CNN seed predictions into one table."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def patient_bootstrap(frame, cols, n_boot=2000, seed=2026):
    frame = frame.reset_index(drop=True)
    by_patient = frame.groupby("patient").indices
    patients = list(by_patient)
    y = frame.label.to_numpy()
    probs = {c: frame[c].to_numpy() for c in cols}
    rng = np.random.default_rng(seed)
    out = {c: [] for c in cols}
    diffs = {(a,b): [] for a in cols for b in cols if a < b}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(patients), len(patients))
        ix = np.concatenate([by_patient[patients[j]] for j in chosen])
        if np.unique(y[ix]).size < 2:
            continue
        vals = {}
        for c in cols:
            vals[c] = (roc_auc_score(y[ix], probs[c][ix]),
                       average_precision_score(y[ix], probs[c][ix]))
            out[c].append(vals[c])
        for a,b in diffs:
            diffs[(a,b)].append(np.subtract(vals[a], vals[b]))
    return {k: np.asarray(v) for k,v in out.items()}, {k: np.asarray(v) for k,v in diffs.items()}

def json_rows(paths):
    rows=[]
    for path in paths:
        try: obj=json.loads(path.read_text())
        except Exception as e:
            print(f"Skipping unreadable JSON {path}: {e}"); continue
        split=obj.get("split","unknown")
        for key in ("p_a","p_b"):
            if key not in obj: continue
            m=obj[key]
            name = obj.get("name") if key=="p_a" else obj.get("name_b")
            if not name: name=path.stem
            rows.append({
                "model":name,"split":split,"n_windows":obj.get("n_windows"),
                "n_patients":obj.get("n_patients"),"prevalence":obj.get("prevalence"),
                "ROC-AUC":m.get("ROC-AUC"),"ROC-AUC_lo":(m.get("ROC-AUC_CI") or [None,None])[0],
                "ROC-AUC_hi":(m.get("ROC-AUC_CI") or [None,None])[1],
                "PR-AUC":m.get("PR-AUC"),"PR-AUC_lo":(m.get("PR-AUC_CI") or [None,None])[0],
                "PR-AUC_hi":(m.get("PR-AUC_CI") or [None,None])[1],
                "sens@spec90":m.get("sens@spec90"),"sens@spec80":m.get("sens@spec80"),
                "sensitivity":m.get("sensitivity"),"specificity":m.get("specificity"),
                "precision":m.get("precision"),"threshold":m.get("threshold"),
                "source":str(path)
            })
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir",type=Path,default=Path("results"))
    ap.add_argument("--table",type=Path,help="optional evaluate.py --table CSV")
    ap.add_argument("--windows",type=Path,help="validation windows CSV; needed for CNN ensemble")
    ap.add_argument("--pred-dir",type=Path,default=Path("results/preds"))
    ap.add_argument("--out",type=Path,default=Path("results/master_comparison.csv"))
    ap.add_argument("--n-boot",type=int,default=2000)
    args=ap.parse_args()
    rows=json_rows(list(args.results_dir.rglob("*.json")))
    if args.table and args.table.exists():
        t=pd.read_csv(args.table)
        t=t.rename(columns={"name":"model"})
        if "source" not in t: t["source"]=str(args.table)
        rows.extend(t.to_dict("records"))
    df=pd.DataFrame(rows)
    if df.empty: df=pd.DataFrame(columns=["model","split"])
    # Deduplicate repeated append-table runs; keep last occurrence.
    if {"model","split"}.issubset(df.columns):
        df=df.drop_duplicates(["model","split"],keep="last")

    # CNN seed summary and seed-averaged ensemble are computed from predictions,
    # not by averaging already-rounded metrics.
    if args.windows and args.windows.exists() and args.pred_dir.exists():
        w=pd.read_csv(args.windows)[["window_id","patient","label"]]
        seed_files=[]
        for p in args.pred_dir.glob("*.csv"):
            m=re.search(r"cnn(?:_seed|_s)(\d+).*validation",p.stem,re.I)
            if m: seed_files.append((int(m.group(1)),p))
        seed_files.sort()
        if seed_files:
            merged=w.copy(); predcols=[]
            for seed,p in seed_files:
                x=pd.read_csv(p)[["window_id","probability"]].rename(columns={"probability":f"seed_{seed}"})
                if x.window_id.duplicated().any(): raise ValueError(f"Duplicate window IDs in {p}")
                merged=merged.merge(x,on="window_id",how="left",validate="one_to_one")
                predcols.append(f"seed_{seed}")
            if merged[predcols].isna().any().any():
                raise ValueError("CNN seed predictions do not cover every validation window")
            boot,_=patient_bootstrap(merged,predcols,args.n_boot)
            seed_metric=[]
            for c in predcols:
                y=merged.label.to_numpy(); p=merged[c].to_numpy()
                seed_metric.append((roc_auc_score(y,p),average_precision_score(y,p)))
            arr=np.asarray(seed_metric)
            # Mean/SD across seed point estimates; CI from seed-averaged probability ensemble.
            ens=merged[predcols].mean(axis=1)
            merged["cnn_seed_ensemble"]=ens
            ensboot,_=patient_bootstrap(merged,["cnn_seed_ensemble"],args.n_boot)
            y=merged.label.to_numpy()
            b=ensboot["cnn_seed_ensemble"]
            erow={"model":"cnn_seed_averaged_ensemble","split":"validation",
                  "n_windows":len(merged),"n_patients":merged.patient.nunique(),
                  "prevalence":float(y.mean()),"ROC-AUC":roc_auc_score(y,ens),
                  "ROC-AUC_lo":np.percentile(b[:,0],2.5),"ROC-AUC_hi":np.percentile(b[:,0],97.5),
                  "PR-AUC":average_precision_score(y,ens),
                  "PR-AUC_lo":np.percentile(b[:,1],2.5),"PR-AUC_hi":np.percentile(b[:,1],97.5),
                  "source":"mean probability across CNN seed prediction CSVs"}
            rows2=[erow]
            if len(arr)>1:
                # Patient-bootstrap CI for the mean metric across seeds. Each
                # bootstrap replicate is shared across seeds (paired resampling).
                seed_boot_roc=np.mean(np.stack([boot[c][:,0] for c in predcols]),axis=0)
                seed_boot_pr=np.mean(np.stack([boot[c][:,1] for c in predcols]),axis=0)
                rows2.append({"model":"cnn_seed_summary_mean","split":"validation",
                    "n_windows":len(merged),"n_patients":merged.patient.nunique(),
                    "ROC-AUC":arr[:,0].mean(),"ROC-AUC_seed_sd":arr[:,0].std(ddof=1),
                    "ROC-AUC_lo":np.percentile(seed_boot_roc,2.5),
                    "ROC-AUC_hi":np.percentile(seed_boot_roc,97.5),
                    "PR-AUC":arr[:,1].mean(),"PR-AUC_seed_sd":arr[:,1].std(ddof=1),
                    "PR-AUC_lo":np.percentile(seed_boot_pr,2.5),
                    "PR-AUC_hi":np.percentile(seed_boot_pr,97.5),
                    "source":"mean and sample SD across seed point estimates; CI from paired patient bootstrap"})
            df=pd.concat([df,pd.DataFrame(rows2)],ignore_index=True,sort=False)

    # Sort by split then ROC-AUC descending; retain missing CIs visibly as blank.
    if "ROC-AUC" in df:
        df=df.sort_values(["split","ROC-AUC"],ascending=[True,False],na_position="last")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out,index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {args.out}")
    missing_ci=df[df.get("ROC-AUC_lo",pd.Series(index=df.index,dtype=float)).isna() |
                   df.get("PR-AUC_lo",pd.Series(index=df.index,dtype=float)).isna()]
    if len(missing_ci):
        print("\nWARNING: these rows do not yet have both metric CIs:")
        print(missing_ci[["model","split"]].to_string(index=False))

if __name__=="__main__": main()
