"""Improved baseline hospital mortality model on MIMIC-IV Demo.

Changes vs. the age + gender version:
  1. More informative features available during the first 24 hours (admission
      details, demographics, early ICU transfer, and common lab summaries).
  2. Patient-grouped, stratified 5-fold cross-validation instead of one 70/30
     split. The demo has only ~15 deaths, so a single test split is extremely noisy.
  3. Metrics that suit imbalanced data (ROC AUC, average precision, balanced
     accuracy) instead of plain accuracy.
  4. A small sweep over regularization strength C.

Education only, not clinical use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import physionet

DATASET = "mimic-iv-demo"
VERSION = "2.2"
TARGET = "hospital_expire_flag"

LAB_ITEMS = {
    50813: "lactate",
    50912: "creatinine",
    51006: "urea_nitrogen",
    50983: "sodium",
    50971: "potassium",
    50882: "bicarbonate",
    50885: "bilirubin",
    51222: "hemoglobin",
    51301: "white_blood_cells",
    51265: "platelets",
}
LAB_FEATURES = [f"lab_{name}" for name in LAB_ITEMS.values()]
NUMERIC = [
    "anchor_age", "admit_hour", "prior_admissions", "had_icu", *LAB_FEATURES
]
CATEGORICAL = [
    "gender", "admission_type", "admission_location",
    "insurance", "language", "marital_status", "race",
]
FEATURES = NUMERIC + CATEGORICAL


def build_cohort() -> pd.DataFrame:
    patients = physionet.load(
        DATASET, "hosp/patients", version=VERSION,
        usecols=["subject_id", "anchor_age", "gender"],
    )
    admissions = physionet.load(
        DATASET, "hosp/admissions", version=VERSION,
        usecols=[
            "subject_id", "hadm_id", "admittime", "admission_type",
            "admission_location", "insurance", "language",
            "marital_status", "race", TARGET,
        ],
        parse_dates=["admittime"],
    )
    icustays = physionet.load(
        DATASET, "icu/icustays", version=VERSION,
        usecols=["hadm_id", "intime"], parse_dates=["intime"],
    )

    df = admissions.merge(patients, on="subject_id", how="inner", validate="many_to_one")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    # Features
    df["admit_hour"] = df["admittime"].dt.hour
    df = df.sort_values(["subject_id", "admittime"])
    df["prior_admissions"] = df.groupby("subject_id").cumcount()
    icu_admissions = icustays.merge(
        admissions[["hadm_id", "admittime"]], on="hadm_id", how="inner"
    )
    early_icu_ids = icu_admissions.loc[
        (icu_admissions["intime"] >= icu_admissions["admittime"])
        & (icu_admissions["intime"] < icu_admissions["admittime"] + pd.Timedelta(hours=24)),
        "hadm_id",
    ]
    df["had_icu"] = df["hadm_id"].isin(early_icu_ids).astype(int)

    # Collapse the long tail of race labels (e.g. "WHITE - RUSSIAN" -> "WHITE")
    df["race"] = df["race"].astype(str).str.split(" - ").str[0].str.split("/").str[0]

    labs = physionet.load(
        DATASET, "hosp/labevents", version=VERSION,
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    labs = labs[labs["itemid"].isin(LAB_ITEMS) & labs["valuenum"].notna()]
    labs = labs.merge(
        df[["hadm_id", "admittime"]].drop_duplicates(),
        on="hadm_id",
        how="inner",
        validate="many_to_one",
    )
    labs = labs[
        (labs["charttime"] >= labs["admittime"])
        & (labs["charttime"] < labs["admittime"] + pd.Timedelta(hours=24))
    ]
    early_labs = labs.groupby(["hadm_id", "itemid"])["valuenum"].median().unstack()
    early_labs = (
        early_labs.reindex(columns=list(LAB_ITEMS))
        .rename(columns=LAB_ITEMS)
        .add_prefix("lab_")
    )
    early_labs.columns.name = None
    df = df.merge(early_labs, left_on="hadm_id", right_index=True, how="left", validate="one_to_one")
    return df.reset_index(drop=True)


def build_model(C: float) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Group rare categories together so the model doesn't overfit them
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
    ])


def main() -> None:
    df = build_cohort()
    X, y, groups = df[FEATURES], df[TARGET], df["subject_id"]
    print(f"{len(df)} admissions, {groups.nunique()} patients, "
          f"{y.sum()} deaths ({y.mean():.1%})")

    # Each patient stays in one fold; deaths are spread across folds
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nOut-of-fold performance by regularization strength C "
          "(smaller C = stronger regularization):")
    results = {}
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        proba = cross_val_predict(
            build_model(C), X, y, groups=groups, cv=cv, method="predict_proba"
        )[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[C] = proba
        print(
            f"  C={C:<5} ROC AUC={roc_auc_score(y, proba):.3f}  "
            f"PR AUC={average_precision_score(y, proba):.3f}  "
            f"balanced acc={balanced_accuracy_score(y, pred):.3f}"
        )

    best_C = max(results, key=lambda c: roc_auc_score(y, results[c]))
    proba = results[best_C]
    pred = (proba >= 0.5).astype(int)
    print(f"\nBest C by ROC AUC: {best_C}")
    print(f"Baseline PR AUC for a no-skill model = prevalence = {y.mean():.3f}")
    print("\nOut-of-fold classification report:")
    print(classification_report(y, pred, zero_division=0))
    print("Confusion matrix [rows=true, cols=predicted]:")
    print(confusion_matrix(y, pred, labels=[0, 1]))

    # Final model on all data, for inspecting coefficients
    final = build_model(best_C).fit(X, y)
    names = final.named_steps["prep"].get_feature_names_out()
    coefs = final.named_steps["clf"].coef_[0]
    table = (
        pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
        .sort_values("coef", key=np.abs, ascending=False)
        .head(12)
    )
    print("\nLargest coefficients (numeric features standardized):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
