# Regulatory Approval Predictor — Health Canada

> Two XGBoost models over Health Canada drug submissions: how long a submission
> will spend in review, and whether it will be approved — with a SHAP-based
> regulatory reading of the drivers.

![status](https://img.shields.io/badge/status-active-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

- **Data:** [*Drug and health product submissions under review*](https://www.canada.ca/en/health-canada/services/drug-health-product-review-approval/submissions-under-review.html)
  — Health Canada, public, monthly. Snapshot `2026-07-31`, real (not synthetic).
- **Models:**
  1. `XGBRegressor` → `review_days` (calendar days accepted → concluded)
  2. `XGBClassifier` → `approved` (marketing authorisation vs cancelled / withdrawn / non-compliant)
- **Scope (V1):** Python modelling only. Flat CSV exports feed a later dashboard
  (Power BI or Tableau — not yet decided).

## Quick start

```bash
pip install -r requirements.txt

python scripts/build_dataset.py          # raw .xlsx -> data/processed/submissions_clean.csv (+ QA summary)
python scripts/train_models.py           # trains both models, writes metrics / SHAP / predictions
python scripts/calibrate_reprojection.py # temporal-holdout calibration for the late/overdue re-projection
python scripts/predict_under_review.py   # approval probability + estimated conclusion date (with intervals) for the 168 in-flight submissions; re-projects late/overdue conclusion dates

pytest                                   # data + pipeline checks
```

Optional — the survival-analysis exploration (documented alternative, not the
primary model; see [docs/survival-analysis.md](docs/survival-analysis.md)):

```bash
python scripts/train_survival.py    # discrete-time competing-risks model
python scripts/compare_models.py    # regressor/classifier vs discrete survival -> reports/comparison.md
python scripts/compare_aft.py       # regressor vs continuous parametric AFT   -> reports/comparison_aft.md
```

Place the source workbook at `data/raw/submissions-under-review-2026-07.xlsx`
(already present).

## Results — 2026-07 snapshot

| Model | Key metric (test) | Baseline |
|---|---|---|
| Review time | MAE **77 days**, median AE **19 days** | median predictor: MAE 97 |
| Approval | ROC-AUC **0.66**, Brier **0.076**, calibrated (OOB 0.911 vs 0.913 observed) | 0.50 / 0.078 |

Modest but real lift — the interesting output is *why*. Expedited pathways
(`priority_review`, `project_orbis_a`, `covid19`) shorten reviews to ~120–180
days vs ~300; biosimilars run longer. Supplemental submissions and repeat sponsors
are approved more often. For the 168 in-flight submissions,
`predict_under_review.py` gives a calibrated approval probability with a 90 %
bootstrap interval **and** an estimated conclusion date with a p10–p90 window
(quantile regression, floored at time already elapsed).

**Re-projection for late / overdue submissions.** When a submission has already
outlived its estimate, the point prediction is stale. `src/bayesian.py` treats
the regressor's predictive lognormal as a prior, conditions on "still under
review at `days_elapsed`" (left-truncated lognormal), and reports an updated
conclusion date. `scripts/calibrate_reprojection.py` tunes it on a temporal
holdout (raw band covers 40 % of realised late/overdue durations; σ×2.3 → 74 %).
The `bayes_*` columns of `under_review_forecast.csv` carry the result.

**Survival analysis** was explored to handle censoring in training — both a
discrete-time competing-risks model and a continuous parametric AFT. Both improve
interval coverage under temporal drift but lose median AE (the gradient-boosted
regressor reproduces the sharp 300-day service-standard spike that neither
alternative matches). Not adopted for the point prediction; kept as a documented,
runnable alternative. Full reading in [docs/methodology.md](docs/methodology.md)
and [docs/survival-analysis.md](docs/survival-analysis.md).

## Repo structure

```
├── data/
│   ├── raw/          submissions-under-review-2026-07.xlsx  (source workbook)
│   └── processed/    submissions_clean.csv, predictions.csv, under_review_forecast.csv  (built)
├── src/              importable library
│   ├── config.py     paths, constants, raw-column mappings
│   ├── data.py       load / merge / clean the workbook
│   ├── features.py   feature engineering + leak-safe FrequencyEncoder pipeline
│   ├── models.py     train + cross-validate + evaluate
│   ├── explain.py    SHAP
│   ├── uncertainty.py   bootstrap intervals for the approval probability
│   ├── bayesian.py     survival-conditioned re-projection for late/overdue submissions
│   ├── survival.py       discrete-time competing-risks model  (explored, not adopted)
│   ├── survival_aft.py   continuous parametric AFT (lifelines) (explored, not adopted)
│   └── survival_metrics.py  Harrell/Uno C, IPCW Brier, time-dependent AUC
├── scripts/          build_dataset.py, train_models.py, calibrate_reprojection.py,
│                     predict_under_review.py  (primary) · train_survival.py,
│                     compare_models.py, compare_aft.py, predict_under_review_survival.py  (exploration)
├── notebooks/        01_eda.ipynb
├── models/           trained model artifacts  (gitignored, rebuilt by the scripts)
├── reports/          metrics.json, shap_*.csv, feature_effects.csv, reprojection_calibration.json,
│                     comparison*.md, survival/, figures/
├── tests/            test_data.py, test_models.py, test_uncertainty.py, test_bayesian.py,
│                     test_survival.py, test_survival_aft.py
└── docs/             data-dictionary.md, methodology.md, survival-analysis.md
```

The `sql/` and `dashboard/` folders are placeholders from the project template,
kept for the eventual dashboard layer.

## Outputs

| File | Contents |
|---|---|
| `data/processed/submissions_clean.csv` | merged, cleaned training table (1,515 rows) |
| `data/processed/predictions.csv` | per-submission predictions incl. 168 in-flight submissions |
| `data/processed/under_review_forecast.csv` | approval probability (90% interval) + estimated conclusion date (p10/p50/p90) per in-flight submission, incl. `bayes_*` re-projection columns for late/overdue rows |
| `reports/metrics.json` | CV + hold-out metrics vs baselines |
| `reports/reprojection_calibration.json` | temporal-holdout calibration constants for the re-projection |
| `reports/shap_regressor.csv` / `reports/shap_classifier.csv` | mean \|SHAP\| per feature |
| `reports/feature_effects.csv` | median review time & approval rate with/without each class flag |
| `reports/figures/shap_*.png` | SHAP bar + beeswarm summaries |
| `reports/comparison.md` / `reports/comparison_aft.md` | regressor/classifier vs survival-analysis alternatives |

## Limitations

- ~1,500 rows is small for gradient boosting — predictions are indicative.
- Coverage bounded by regulatory transparency (NDS 2015+, SNDS 2016+); pre-2018
  rows lack sponsor names.
- Single snapshot, no automated refresh in V1.
- `review_days` is wall-clock time (includes sponsor "clock stops").
- Right-censoring is not modelled in training (submissions still under review are
  excluded), so the slowest reviews are under-represented and the p10–p90 band
  misses ~90 % of reviews that run past 700 days. The late/overdue re-projection
  and the survival-analysis exploration both address this; neither fully solves it.

See [docs/methodology.md](docs/methodology.md) for the full list.

## License

MIT — see [LICENSE](LICENSE).
