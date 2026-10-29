# Detection-of-Atrial-Fibrillation-Termination
# Icentia11k AFib Interval-End Prediction
## Team README: Data Access, Subsampling, and Analysis Plan



## 1. Project Overview

The initial experiment uses:
- **Input:** 10 seconds of ECG.
- **Primary prediction horizon:** 30 seconds.
- **Target:** Whether the annotated AFib interval ends within the next 30 seconds.
- **Baseline model:** Logistic regression.
- **Comparison model:** Random forest.
- **Eventual model:** 1D CNN
	- possible addition of transformers or rolling windows
- **Data strategy:** Patient-level train/validation/test split, followed by patient- and episode-capped sampling of training windows.



### Available Files

Relevant files include:

| File | Purpose |
|---|---|

*****
| `train_windows_patient_episode_capped_natural.csv` | Working training-window table |
***THIS WILL BE OUR MAIN TRAINING TABLE, WHICH DETERMINES WHAT EPISODES TO ANALYZE

| `patient_split_manifest.csv` | Patient-to-partition assignments for training, validation, and testing |
| `train_windows.csv` | Previously generated training-window table; verify its sampling settings before reuse |
| `validation_windows.csv` | Previously generated validation-window table |
| `test_windows.csv` | Previously generated test-window table |

### Accessing Files

*****************
#1 Git Clone
*************************
    !git clone 
    !git clone https://github.com/YOUR_USERNAME/Detection-of-Atrial-Fibrillation-Termination.git
    %cd Detection-of-Atrial-Fibrillation-Termination


- If the repository has already been cloned in the current runtime, navigate to the existing directory instead of cloning it again.

    %cd /content/Detection-of-Atrial-Fibrillation-Termination


- REPO_DIR can then be used to reference files inside the cloned repository.

    from pathlib import Path

    REPO_DIR = Path.cwd()
    print("Repository:", REPO_DIR)

- Accessing TRAINING SAMPLE specifically

    TRAIN_FILE = REPO_DIR / "data" / "10s-30s" / "train_windows_patient_episode_capped_natural.csv"

    import pandas as pd

    train_windows = pd.read_csv(TRAIN_FILE)


*************
#2 Google Drive (This is if you want to download files onto your google drive. This part is included because you might see some of my notebooks being mounted in drive instead of github)
**************

If you want to replicate, my working directory is:

`/content/drive/MyDrive/icentia11k_afib_audit/`

The pilot directory is:

`/content/drive/MyDrive/icentia11k_afib_audit/pilot_10s_30s/`

Configurations to access Drive data:
    from pathlib import Path

    REPO_DIR = Path.cwd()

    DATA_DIR = Path(
        "/content/drive/MyDrive/icentia11k_afib_audit"
    )
    PILOT_DIR = DATA_DIR / "pilot_10s_30s"

    TRAIN_SAMPLE_FILE = (
        PILOT_DIR / "train_windows_patient_episode_capped_natural.csv"
    )



### Saving Files/CSVs
*Step 1: Save the CSV to the repository


    OUTPUT_DIR = REPO_DIR / "data"/ "10s-30s"/"outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

- Example: save a generated DataFrame 

(Replace features_df with the name of your actual DataFrame and change the filename as needed.)

    features_df.to_csv(
        OUTPUT_DIR / "ecg_features.csv",
        index=False
    )

*Step 2: Check that the CSV was saved

    csv_path = OUTPUT_DIR / "ecg_features.csv"

    print("Exists:", csv_path.exists())
    print("Size (MB):", round(csv_path.stat().st_size / 1e6, 2))

Step 3: Stage, commit, and push the new CSV

    %cd /content/YOUR_REPOSITORY

    !git status
    !git add outputs/ecg_features.csv
    !git commit -m "Add ECG feature table"
    !git push


### Expected metadata fields

The exact column names should be confirmed by inspecting the CSV. The current candidate-window index is expected to contain at least the following fields:

| Field | Meaning |
|---|---|
| `patient` | Patient identifier |
| `episode_id` | Identifier for the annotated AFib episode associated with the window |
| `label` | Binary prediction target: 1 = annotated interval ends within the prediction horizon; 0 = it does not |
| `prediction_time_sec` | Time within the record at which the prediction is made |
| `partition` | Patient partition: `train`, `validation`, or `test` |
| `record_stem` | Record identifier used to retrieve the ECG segment, if present |

Additional metadata may include window start/end times, episode boundaries, and record identifiers. Confirm the actual schema before assuming a field exists.

### How to interpret a row

For a 10-second input, the ECG window ends at prediction time \(t\). Its input signal covers the preceding 10 seconds.

For the 30-second horizon:

- **Positive (`label = 1`):** The annotated AFib interval ends after \(t\), but no later than \(t+30\) seconds.
- **Negative (`label = 0`):** The annotated interval is not scheduled to end within that horizon, subject to the project's eligibility and censoring rules.
- **Excluded:** Windows for which the outcome cannot be determined reliably, such as relevant boundary or censoring cases.

The label definition must be checked against the candidate-window generation code before final analysis.



## 6. ECG Signal Access

The PhysioNet dataset base URL is:

`https://physionet.org/files/icentia11k-continuous-ecg/1.0`

A record identifier such as `p00020_s03` is associated with a patient and segment. The source files include a header, signal data, and annotations. The record's directory is organized by patient shard.




## 7. Feature Engineering

The first analysis phase should convert each 10-second ECG input into a row of numerical features.

### A. Signal-level features

Calculate basic signal descriptors:

- Mean and standard deviation
- Root mean square amplitude
- Minimum and maximum amplitude
- Peak-to-peak range
- Signal length and missing/non-finite sample count
- Line length or average absolute successive-sample difference
- Signal-quality indicators where feasible

These features are inexpensive to calculate and provide an initial signal-quality baseline.

### B. RR-interval features

Detect heartbeat fiducial points consistently, then calculate intervals between successive detections.

At 250 Hz:

\[
RR_i = \frac{p_{i+1}-p_i}{250}
\]

where \(p_i\) is the sample index of a detected beat.

Candidate features include:

- Mean RR interval
- Standard deviation of RR intervals (SDNN)
- Coefficient of variation of RR intervals
- RMSSD (root mean square of successive RR differences)
- Median absolute successive RR difference
- Mean heart rate
- Fraction of unusually large successive RR changes

For RMSSD:

\[
\mathrm{RMSSD}
=
\sqrt{
\frac{1}{N-1}
\sum_{i=1}^{N-1}
(RR_{i+1}-RR_i)^2
}
\]

If RR intervals are in seconds, RMSSD is also in seconds; multiply by 1,000 to report milliseconds.

**Quality control is essential.** Ten seconds is a short window and may contain relatively few beats. Missed beats, false detections, ectopic beats, and signal noise can distort the features. Plot representative windows, validate the peak detector, and retain quality flags rather than treating every extracted feature as reliable.

### C. Frequency-domain features

As an exploratory extension, calculate power spectral features using an appropriate method such as Welch's method.

First validate whether the chosen frequency bands and interpretation are meaningful for the ECG representation and short window duration. Do not automatically interpret ECG spectral power as a validated measure of autonomic activity.

### D. Feature table structure

Create one feature row per selected window, retaining identifiers and metadata for auditing:

| Column type | Examples |
|---|---|
| Identifiers | `patient`, `episode_id`, `record_stem` |
| Window metadata | Input start time, prediction time, horizon |
| Target | `label` |
| Signal features | `signal_std`, `signal_rms`, `line_length` |
| RR features | `mean_rr`, `sdnn`, `rmssd`, `rr_cv` |
| Quality indicators | Peak count, detection failures, signal-quality flags |

Identifiers must never be used as predictive features. The label, prediction horizon, future interval end time, and any other information unavailable at prediction time must also be excluded from \(X\).

## 8. Modeling Plan

The initial comparison should use two interpretable, conventional models.

### Model 1: Logistic regression

Logistic regression estimates the probability of a positive label from a weighted combination of features.

\[
P(y=1\mid X)=
\frac{1}{1+\exp[-(\beta_0+\beta^T X)]}
\]

Use it as the baseline because it is relatively simple and provides interpretable coefficients. Standardize numerical features and fit any imputation procedure using training data only.

### Model 2: Random forest

A random forest combines predictions from multiple decision trees. It can capture nonlinear relationships and interactions between RR variability, heart rate, and signal features.

It generally does not require standardization, but hyperparameters such as tree depth, number of trees, and minimum leaf size should be tuned using validation data.

### Evaluation design

- Train on training patients only.
- Select model settings and decision thresholds using validation patients.
- Evaluate the final selected model on test patients.
- Use identical eligible windows and feature definitions when comparing algorithms.
- Report PR-AUC, ROC-AUC, sensitivity, precision, and calibration as appropriate.
- Report the positive prevalence in each partition.
- Investigate performance across patients rather than treating all windows as independent observations.

Because neighboring windows from the same patient can be correlated, uncertainty estimates should account for patient clustering when feasible.
