# Methodology

## Goal

Two models over completed Health Canada drug submissions:

1. **Regressor** (`XGBRegressor`) — predict `review_days`, the calendar time a
   submission spends in review.
2. **Classifier** (`XGBClassifier`) — predict `approved`: did the submission reach
   a marketing authorisation (NOC / NOC-c / Interim Order) versus being cancelled,
   withdrawn, or found non-compliant.

The point of the project is the **regulatory reading** of the drivers, not a
leaderboard score — the signal in ~1,500 rows of public metadata is inherently
modest and the results below say so plainly.

## Pipeline

| Step | Code | Output |
|---|---|---|
| Load + merge NDS/SNDS, clean, compute targets | `src/data.py` → `scripts/build_dataset.py` | `data/processed/submissions_clean.csv` |
| Feature engineering + leak-safe encoding | `src/features.py` | in-memory pipeline |
| Train + cross-validate + evaluate | `src/models.py` → `scripts/train_models.py` | `models/`, `reports/metrics.json` |
| SHAP interpretation | `src/explain.py` | `reports/shap_*.csv`, `reports/figures/shap_*.png` |
| Regulatory effect table | `scripts/train_models.py` | `reports/feature_effects.csv` |
| Prediction export (incl. 168 in-flight submissions) | `scripts/train_models.py` | `data/processed/predictions.csv` |
| In-flight forecast: approval probability (bootstrap interval) + estimated conclusion date (quantile interval) | `src/uncertainty.py`, `src/models.py` → `scripts/predict_under_review.py` | `data/processed/under_review_forecast.csv` |

## Data preparation

- Merge the two `... completed` sheets; tag `submission_type` and `is_supplemental`.
- Class flags: `"ü"` → 1, blank → 0. `new_active_substance` (NDS-only) → 0 for SNDS.
- `review_days = (date_concluded − date_accepted).days`.
- **Drop the 1 row with `review_days < 0`** (control number 300215 — conclusion
  dated before acceptance, a data-entry error). Logged, not silently removed.
- `outcome` → binary `approved` via the mapping in the data dictionary; unmapped
  outcomes are dropped and counted (none in the 2026-07 snapshot).

## Features

| Group | Columns | Notes |
|---|---|---|
| Submission class | 12 binary flags | `priority_review`, `biosimilar`, `noc_c`, `covid19`, Project Orbis A/B/C, … |
| Submission type | `is_supplemental` | NDS vs SNDS in one dataset (per project decision) |
| Calendar | `accept_year`, `accept_month`, `accept_quarter` | captures process drift / backlog over time |
| Sponsor / area | `company_name_freq`, `therapeutic_area_freq` | **frequency encoding** — high cardinality (~280 / ~79) makes one-hot impractical |
| Composition | `n_ingredients` | count of active ingredients |

`medicinal_ingredients` is **not** used as a category (near-unique) — only its
element count.

### Leakage control

Frequency encoding is a `FrequencyEncoder` transformer inside a
`sklearn.Pipeline`, so in every cross-validation fold (and every bootstrap
resample) the frequencies are learned from that fold's training rows only. Unseen
categories map to `0`.

## Models

Deliberately small trees — `max_depth=3`, low learning rate, `reg_lambda=2`,
`min_child_weight=3`, `subsample=colsample_bytree=0.8`. `n_estimators` is **fixed**
(300 for the regressor and each quantile model, 150 for the classifier): the
held-out slice needed for early stopping is too small and noisy at this sample
size to be trustworthy, so we lean on regularisation + cross-validation instead.

- **Regressor** is trained on `log1p(review_days)` with an **absolute-error**
  objective. The target spans 8 to 3,000+ days with a heavy right tail; this
  combination roughly halves the median absolute error versus squared error on
  the raw scale. Predictions are `expm1`-inverted (`predict_review_days`).
- **Classifier** is left at its natural class balance (**no `scale_pos_weight`**).
  That keeps the probability outputs calibrated — out-of-bag mean predicted
  probability 0.911 vs an observed approval rate of 0.913, and the reliability
  table below stays within ±0.035 of the diagonal across the whole range — which
  is what makes the probability + interval output in `src/uncertainty.py`
  meaningful. For the "flag this submission for a closer look" use, a submission
  is flagged when its predicted probability falls below **0.90** (`FLAG_THRESHOLD`,
  ≈ the bottom third of the distribution) rather than the useless 0.5.

### Evaluation

Single stratified 80/20 split (stratified on `approved`, shared by both models so
the test rows match) + 5-fold cross-validation on the training portion. Metrics
are always reported against a naive baseline: predict the training median for the
regressor; predict the base rate (a "no-skill" classifier, ROC-AUC 0.5) for the
classifier.

## Results — 2026-07 snapshot

*(will drift as Health Canada publishes new snapshots; re-run to refresh)*

**Regressor** — n_train 1,212 / n_test 303

| Metric | Model (test) | Baseline (median) |
|---|---|---|
| MAE (days) | **77** | 97 |
| Median AE (days) | **19** | — |
| RMSE (days) | 246 | 263 |
| R² | 0.13 | 0 |

The model tracks the *typical* submission well (median error ~19 days) but the
long tail — multi-year reviews with repeated clarifax cycles — is not predictable
from this metadata, which is why RMSE and R² stay weak.

**Prediction interval.** Three more boosters are trained with the pinball
(quantile) objective at α = 0.10 / 0.50 / 0.90 on `log1p(review_days)`; per row
the three predictions are sorted to stay monotone. Out-of-fold the p10–p90 band
covers **81 %** of actual durations (nominal 80 %) with a median width of ~250
days — honest, but wide. Coverage collapses to ~10 % for reviews that actually
ran past 700 days: the band cannot anticipate the extreme tail (see limitations).

**Classifier** — same split

| Metric | Model (test) | Baseline |
|---|---|---|
| ROC-AUC | **0.66** | 0.50 |
| PR-AUC | 0.95 | 0.91 (positive rate) |
| Brier score | **0.076** | 0.078 (predict base rate) |
| Balanced accuracy @0.90 | **0.65** | 0.50 |
| Rejection recall / precision @0.90 | 0.62 / 0.16 | — |

Real but modest lift on *ranking* (ROC-AUC), and the Brier score is barely below
"just predict the 91 % base rate" — the probabilities are trustworthy but the
model rarely departs far from the base rate. Non-approval is rare and often
driven by information not in the file (clinical data quality, sponsor commercial
decisions), so at the 0.90 flag threshold the model catches ~6 of every 10 real
non-approvals but 5 of every 6 flags are false alarms. It is a triage signal, not
a verdict.

## Forecast for in-flight submissions

`scripts/predict_under_review.py` scores the 168 submissions still under review
(the `... under review` sheets) and produces, for each:

**Approval probability + 90 % bootstrap interval.** The classifier pipeline is
refit on 200 bootstrap resamples of the completed data; every submission is scored
by all 200 models; the interval is the 5th–95th percentile of those scores.

- It measures **estimation uncertainty** — how much P(approval) moves with a
  different training sample. Median band width ≈ 0.05 (e.g. 0.93 → [0.88, 0.96]);
  a few unusual submissions are much wider (Moderna's combined influenza /
  SARS-CoV-2 mRNA vaccine: 0.84 → [0.66, 0.94]).
- It does **not** remove **outcome uncertainty**: a submission at 0.90 with a
  tight band is still refused about 1 time in 10.
- Reliability is checked out-of-bag on the completed data (each row scored only by
  the resamples that excluded it) — the gap between predicted and observed
  approval rate stays within ±0.035 across all five probability quintiles.

**Estimated conclusion date + interval.** `date_accepted` plus the p10 / p50 / p90
review-time predictions. Because these submissions have already been in review for
a known number of days, every quantile is **floored at that elapsed duration** —
the review cannot end in the past. A `status` column records where each stands:

| status | meaning | date reported |
|---|---|---|
| `on_track` | elapsed < p50 | genuine future p50 date + [p10, p90] window |
| `late` | p50 ≤ elapsed ≤ p90 | "not before the snapshot"; only `conclusion_date_latest` informative |
| `overdue` | elapsed > p90 | model expected it concluded; metadata doesn't explain the delay |

In the 2026-07 snapshot: 130 `on_track`, 20 `late`, 18 `overdue`.

Output: `data/processed/under_review_forecast.csv` (`approval_proba`,
`approval_ci_low/high`, `est_conclusion_date`, `conclusion_date_earliest/latest`,
`remaining_days_p50`, `status`, `days_in_review_so_far`, and the
`bayes_conclusion_date` / `bayes_conclusion_earliest/latest` /
`bayes_remaining_days_p50` / `bayes_update_applied` re-projection columns below).

**Bayesian re-projection (`late` / `overdue`).** For a submission that has already
outlived its initial estimate, the point prediction is stale — by observation it
is one of the slower reviews. `src/bayesian.py` treats the regressor's predictive
distribution as a **prior**, conditions on the evidence *"still under review at
`days_in_review_so_far`"*, and reads an updated median / p10 / p90 off the
**posterior**. Same idea as the Aily Lab GRA deck (slide 44), made explicit:

- prior: `review_days ~ LogNormal(μ, σ)`, with `μ = ln(p50)` and `σ` from the
  p10–p90 spread in log space (matched to each row's predicted quantiles;
  consistent with training on `log1p`);
- evidence: `review_days > c`, `c` = days elapsed;
- posterior: the left-truncated lognormal, which has a closed form, so the update
  is a few `norm.cdf` / `norm.ppf` calls and extrapolates past the model's p90
  (where the raw quantile grid runs out). Capped at 7 years (`MAX_REVIEW_DAYS`).

**Calibration (`scripts/calibrate_reprojection.py`).** The prior is fit on
completed submissions only, so it is optimistic on slow reviews. On a *temporal*
holdout (train on submissions concluded by 2022-12-31; evaluate on the 128 that
were in-flight then and have since concluded) the raw truncated-lognormal p10–p90
band covers only **40 %** of the realised durations for the `late` / `overdue`
subset. A flat shift on `μ` over-corrects the barely-late rows (they do conclude
soon), so instead the location correction is left to the truncation — which is
position-aware — and only `σ` is widened, by the factor that maximises holdout
coverage (**×2.3**, → **74 %** coverage). Constants live in
`reports/reprojection_calibration.json` and are read back by `src/bayesian.py`;
re-run after a snapshot refresh. 74 % < 80 % nominal: the residual gap is the
lognormal shape being too light-tailed for multi-year reviews — treat
`bayes_conclusion_latest` as an informed bound, not a calibrated p90.

`on_track` rows keep their initial (elapsed-floored) estimate. Output columns:
`bayes_conclusion_date`, `bayes_conclusion_earliest/latest`,
`bayes_remaining_days_p50`, `bayes_update_applied`, `bayes_note` (`initial` /
`reprojected` / `beyond_model` — the last when the posterior is itself exhausted,
i.e. elapsed past its own p90, where only "not before the snapshot" is asserted).
In the 2026-07 snapshot the 38 `late` / `overdue` submissions are re-projected a
median ~3 months past the snapshot.

Still not modelled: right-censoring in *training* (the slowest reviews remain
under-represented in the target). A proper treatment (parametric AFT / random
survival forest / XGBoost `survival:aft`, with the in-flight submissions as
censored observations) is the `feature/survival-analysis` line of work — its
discrete-time competing-risks variant improved tail coverage and RMSE but lost
~11 days of median AE to 30-day binning, so it was not adopted; a *continuous*
AFT is the open follow-up. The re-projection here corrects at inference time but
does not fix the fit.

## Differentiator — regulatory reading of the drivers

Top SHAP features and the direction they push (`reports/shap_*.csv`,
`reports/feature_effects.csv`):

**Review time.** `priority_review` is the dominant driver and shortens reviews
sharply — median 179 days with the flag vs 300 without; `project_orbis_a`
(175 days) and `covid19` (120 days) are similar, consistent with the targeted
timelines of expedited and pandemic pathways. `biosimilar` pushes the other way
(median 320 vs 299) — comparability assessments add review cycles even though the
active ingredient is already known. `is_supplemental` shaves a few days
(SNDS 298 vs NDS 300) — a narrower change set to assess.

**Approval.** The strongest signals are `accept_year` (process drift over time),
`is_supplemental` (SNDS approval rate 95 % vs NDS 88 % — an incremental change to
an approved product clears more often), sponsor history (`company_name_freq` —
repeat filers fare better), and `priority_review` (99 % approval — the policy
selects submissions with strong evidence packages). `covid19` lowers the
predicted probability (71 % vs 92 %), reflecting Interim Order products that
later expired or were withdrawn.

## Assumptions and limitations

- **Small sample** (~1,500 rows) for gradient boosting — hence shallow trees and
  heavy regularisation; treat point predictions as indicative.
- **Temporal coverage** limited by regulatory transparency: NDS from 2015,
  SNDS from 2016. Pre-2018 rows have no sponsor name.
- **Single snapshot** (2026-07-31) — no automated refresh in V1.
- **Censoring:** submissions still under review are excluded from training, so the
  slowest reviews are under-represented in the target. The p10–p90 review-time
  band consequently misses ~90 % of reviews that end up running past 700 days, and
  the in-flight conclusion dates lean optimistic for the `late` / `overdue` cases.
- **`review_days`** is wall-clock time, which includes sponsor response time
  ("clock stops"), not just Health Canada review effort.
- SHAP values for the regressor are in `log1p` units (monotone with days).

## Change log

| Date | Change | Author |
|---|---|---|
| 2026-09-07 | Initial V1 modelling pipeline | GPien |
| 2026-09-07 | Classifier switched to natural class balance (calibrated); added bootstrap approval intervals for in-flight submissions | GPien |
| 2026-09-07 | Added quantile review-time models (p10/p50/p90) and estimated conclusion dates with intervals for in-flight submissions | GPien |
| 2026-09-08 | Bayesian re-projection of the conclusion date for `late` / `overdue` submissions (left-truncated lognormal posterior); `src/bayesian.py` | GPien |
