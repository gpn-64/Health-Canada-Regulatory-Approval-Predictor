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

python scripts/build_dataset.py         # raw .xlsx -> data/processed/submissions_clean.csv (+ QA summary)
python scripts/train_models.py          # trains both models, writes metrics / SHAP / predictions
python scripts/calibrate_reprojection.py # temporal-holdout calibration for the late/overdue re-projection (writes reports/reprojection_calibration.json)
python scripts/predict_under_review.py  # approval probability + estimated conclusion date, both with intervals, for the 168 in-flight submissions; re-projects late/overdue conclusion dates

pytest                                  # data + pipeline checks
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
(quantile regression, floored at time already elapsed). Full reading in
[docs/methodology.md](docs/methodology.md).

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
│   └── uncertainty.py  bootstrap intervals for the approval probability
├── scripts/          build_dataset.py, train_models.py, predict_under_review.py  (run directly)
├── notebooks/        01_eda.ipynb
├── models/           trained model artifacts  (gitignored, rebuilt by the script)
├── reports/          metrics.json, shap_*.csv, feature_effects.csv, figures/
├── tests/            test_data.py, test_models.py, test_uncertainty.py
└── docs/             data-dictionary.md, methodology.md
```

The `sql/` and `dashboard/` folders are placeholders from the project template,
kept for the eventual dashboard layer.

## Outputs

| File | Contents |
|---|---|
| `data/processed/submissions_clean.csv` | merged, cleaned training table (1,515 rows) |
| `data/processed/predictions.csv` | per-submission predictions incl. 168 in-flight submissions |
| `data/processed/under_review_forecast.csv` | approval probability (90% interval) + estimated conclusion date (p10/p50/p90) for each in-flight submission |
| `reports/metrics.json` | CV + hold-out metrics vs baselines |
| `reports/shap_regressor.csv` / `reports/shap_classifier.csv` | mean \|SHAP\| per feature |
| `reports/feature_effects.csv` | median review time & approval rate with/without each class flag |
| `reports/figures/shap_*.png` | SHAP bar + beeswarm summaries |

## Limitations

- ~1,500 rows is small for gradient boosting — predictions are indicative.
- Coverage bounded by regulatory transparency (NDS 2015+, SNDS 2016+); pre-2018
  rows lack sponsor names.
- Single snapshot, no automated refresh in V1.
- `review_days` is wall-clock time (includes sponsor "clock stops").

See [docs/methodology.md](docs/methodology.md) for the full list.

## License

MIT — see [LICENSE](LICENSE).
