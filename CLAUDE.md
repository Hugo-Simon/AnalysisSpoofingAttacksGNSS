# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for detecting GPS/GNSS spoofing attacks using machine learning. Raw GPS signals are processed in MATLAB (via the FGI-GSRx receiver), then classified using Python ML/DL pipelines.

## Environment Setup

**Windows:**
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate
```

**Linux:**
```bash
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Data Pipeline (end-to-end workflow)

1. **Label data** (0 = clean, 1 = spoofed):
   ```bash
   python add_attack_type.py -o tracking_gpsl1_19_cleanStatic_ml.csv ..\Matlab\cleanStatic\tracking_gpsl1_19.csv 0
   python add_attack_type.py -o tracking_gpsl1_ds1_19_ml.csv ..\Matlab\ds1\tracking_gpsl1_19.csv 1
   ```

2. **Merge datasets** (use `--skip N` when spoofing starts after ~100s, i.e., skip ~110000 rows):
   ```bash
   python merge_csv.py -o trackData_gpsl1_ds1_19_merge.csv tracking_gpsl1_19_cleanStatic_ml.csv tracking_gpsl1_ds1_19_ml.csv
   ```

3. **Select features** (outputs a `*_ml.csv` file with the relevant columns):
   ```bash
   python columns_selection.py trackData_gpsl1_ds1_19_merge.csv
   ```

4. **Train and evaluate ML/DL models:**
   ```bash
   python spoofing_gnss_ml_dl.py -i trackData_gpsl1_ds1_19_merge_ml.csv --prefix ds1_19_
   ```

## Key Scripts

| Script | Purpose |
|---|---|
| `add_attack_type.py` | Appends `attack_type` column (0/1) to tracking CSV |
| `merge_csv.py` | Concatenates clean + spoofed CSVs; `--skip N` trims rows |
| `columns_selection.py` | Filters to ML-relevant features, outputs `*_ml.csv` |
| `spoofing_gnss_ml_dl.py` | Main ML/DL training (6 classifiers + MLP) with GridSearchCV |
| `spoofing_gnss_pca_ml_dl.py` | PCA variant of above |
| `plot_csv_column.py` | Visualizes a CSV column; `--ma N` applies moving average |
| `spoofing_tracking_analysis.py` | Analyzes raw tracking signal quality metrics |

## Architecture

**MATLAB → CSV → Python pipeline:**

```
FGI-GSRx (MATLAB)        Raw GPS I/Q → tracking correlators per satellite
    ↓
add_attack_type.py        Labels rows 0 (clean) or 1 (spoofed)
    ↓
merge_csv.py              Combines clean + attack scenario data
    ↓
columns_selection.py      Selects 6 discriminative features:
                          CN0fromSNR, I_P, Q_P, pllLockIndicator,
                          doppler, dllDiscr  →  + attack_type label
    ↓
spoofing_gnss_ml_dl.py    Trains 8 models:
                          LogisticRegression, KNN, GaussianNB,
                          DecisionTree, RandomForest, XGBoost,
                          Keras MLP (two variants)
                          Uses: StratifiedShuffleSplit (80/20),
                          SMOTE/RandomOverSampler/RandomUnderSampler,
                          GridSearchCV + StratifiedKFold CV,
                          SHAP for feature importance
    ↓
Output PNGs + CSVs        confusion matrices, feature importance,
                          SHAP values, correlation heatmaps
```

## Data Source

TEXBAT (Texas Spoofing Test Battery) dataset. Scenarios named `ds1`–`ds8` are spoofed; `cleanStatic` is the baseline. Files are named `tracking_gpsl1_<sat>.csv` where `<sat>` is the satellite/channel number.

## MATLAB Setup

- Requires FGI-GSRx receiver project
- Replace its `doTracking.m` with the one in `Matlab/`
- Configure scenario via `complete_Texbat_Dataset-*.txt` and the `.bat` launcher scripts
