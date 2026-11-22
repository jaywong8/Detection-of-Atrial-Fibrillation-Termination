#!/usr/bin/env python3
"""Train feature baselines on train features and write validation predictions.

This script intentionally does NOT score the test set. Test predictions are only
written after the team freezes the split and explicitly authorizes final evaluation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

FEATURES = [
    "sig_mean", "sig_std", "sig_rms", "sig_min", "sig_max", "sig_ptp",
    "line_length", "n_beats", "mean_rr", "sdnn", "rr_cv", "rmssd",
    "med_abs_drr", "mean_hr", "pnn50",
]
RR_FEATURES = ["n_beats", "mean_rr", "sdnn", "rr_cv", "rmssd",
               "med_abs_drr", "mean_hr", "pnn50"]
CLEAN_SIGNAL = ["sig_std", "line_length"]

def model_factories(seed: int):
    """Return the required primary models plus Agam/Riya notebook variants."""
    def logreg(weight=None):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight=weight, random_state=seed))
    def rf(weight=None):
        return make_pipeline(SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                   class_weight=weight, random_state=seed, n_jobs=-1))
    def grad():
        return make_pipeline(SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=300, class_weight="balanced",
                                           random_state=seed))
    return {
        # Main standardized full-feature baselines.
        "logreg": lambda: logreg("balanced"),
        "random_forest": lambda: rf("balanced_subsample"),
        "gradient_boosting": grad,
        # Agam's earlier unweighted full-feature notebook baselines.
        "agam_logreg_unweighted": lambda: logreg(None),
        "agam_random_forest_unweighted": lambda: rf(None),
        # Agam's RR-only model family (time excluded from candidate predictors).
        "agam_rr_only_logreg": lambda: logreg("balanced"),
        "agam_rr_only_random_forest": lambda: rf("balanced"),
        "agam_rr_only_gradient_boosting": grad,
        "agam_rr_signal_logreg": lambda: logreg("balanced"),
        "agam_rr_signal_random_forest": lambda: rf("balanced"),
        "agam_rr_signal_gradient_boosting": grad,
        # Riya's full-feature class-balanced model settings.
        "riya_logreg": lambda: logreg("balanced"),
        "riya_random_forest": lambda: rf("balanced_subsample"),
    }

def load_feature_file(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} feature file: {path}\n"
            "Generate it from the ECG windows first; *_windows.csv files are metadata, "
            "not feature tables."
        )
    df = pd.read_csv(path)
    required = {"window_id", "patient", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if df.window_id.duplicated().any():
        raise ValueError(f"{path} has duplicate window_id values")
    if not set(df.label.dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{path}: label must be binary 0/1")
    return df

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=Path("data/10s-30s/outputs/train_features.csv"))
    ap.add_argument("--validation", type=Path, default=Path("data/10s-30s/outputs/validation_features.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/preds"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-controls", action="store_true",
                    help="also fit time-only, metadata-only, and shuffled-label controls")
    args = ap.parse_args()

    train = load_feature_file(args.train, "train")
    val = load_feature_file(args.validation, "validation")
    absent = [c for c in FEATURES if c not in train or c not in val]
    if absent:
        raise ValueError(f"Feature files are missing required predictors: {absent}")
    if set(train.patient) & set(val.patient):
        raise ValueError("Patient leakage: train and validation contain overlapping patients")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    X_val = val[FEATURES].apply(pd.to_numeric, errors="coerce")
    y_train = train.label.astype(int)

    rr_signal_features = RR_FEATURES + CLEAN_SIGNAL
    configs = model_factories(args.seed)
    for name, factory in configs.items():
        if name.startswith("agam_rr_only_"):
            use_cols = RR_FEATURES
        elif name.startswith("agam_rr_signal_"):
            use_cols = rr_signal_features
        else:
            use_cols = FEATURES
        model = factory()
        model.fit(train[use_cols].apply(pd.to_numeric, errors="coerce"), y_train)
        pred = pd.DataFrame({"window_id": val.window_id,
                             "probability": model.predict_proba(
                                 val[use_cols].apply(pd.to_numeric, errors="coerce"))[:, 1]})
        path = args.out_dir / f"{name}_validation.csv"
        pred.to_csv(path, index=False)
        print(f"Wrote {path} ({len(pred)} predictions)")

    if args.include_controls:
        # Time-only is deliberately a control, never a primary model feature.
        required_time = {"prediction_time_sec", "episode_start_sec"}
        if not required_time.issubset(train.columns) or not required_time.issubset(val.columns):
            raise ValueError("Time-only control needs prediction_time_sec and episode_start_sec")
        tr = train.assign(time_in_episode=train.prediction_time_sec-train.episode_start_sec)
        va = val.assign(time_in_episode=val.prediction_time_sec-val.episode_start_sec)
        time_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(max_iter=3000, class_weight="balanced",
                                                      random_state=args.seed))
        time_model.fit(tr[["time_in_episode"]], y_train)
        pd.DataFrame({"window_id": val.window_id,
                      "probability": time_model.predict_proba(va[["time_in_episode"]])[:, 1]
                     }).to_csv(args.out_dir/"time_only_validation.csv", index=False)

        # Metadata-only is an explicit leakage/control benchmark using record-time
        # position fields. Never interpret it as a deployable model.
        meta_cols = ["input_start_sec", "prediction_time_sec"]
        if not set(meta_cols).issubset(train.columns) or not set(meta_cols).issubset(val.columns):
            raise ValueError("Metadata-only control needs input_start_sec and prediction_time_sec")
        meta_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(max_iter=3000, class_weight="balanced",
                                                      random_state=args.seed))
        meta_model.fit(train[meta_cols], y_train)
        pd.DataFrame({"window_id": val.window_id,
                      "probability": meta_model.predict_proba(val[meta_cols])[:, 1]
                     }).to_csv(args.out_dir/"metadata_only_validation.csv", index=False)

        # Negative-control experiment: shuffle training labels, refit the
        # full-feature logistic baseline, and repeat with independent seeds.
        for shuffle_seed in range(5):
            rng = np.random.default_rng(shuffle_seed)
            shuffled_y = rng.permutation(y_train.to_numpy())
            shuffled_model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced",
                                   random_state=shuffle_seed)
            )
            shuffled_model.fit(X_train, shuffled_y)
            pd.DataFrame({"window_id": val.window_id,
                          "probability": shuffled_model.predict_proba(X_val)[:, 1]
                         }).to_csv(args.out_dir/f"shuffled_labels_seed{shuffle_seed}_validation.csv",
                                   index=False)

    print("Done. No test predictions were generated.")

if __name__ == "__main__":
    main()
