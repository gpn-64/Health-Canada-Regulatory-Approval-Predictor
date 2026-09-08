# Survival analysis — discrete-time competing risks

This document records why the survival formulation was tried, the two models
that were built (discrete-time competing risks, and a continuous parametric
AFT), and the comparison result against the regressor + classifier that remains
the primary model. Merged into `main` as a documented alternative — not the
default. See `reports/comparison.md` / `reports/comparison_aft.md` for the full
numeric comparisons and `reports/survival/metrics_survival.json` for the raw
training/CV/test metrics.

## Why survival at all

The regressor + classifier pipeline on `main` has three structural problems a
time-to-event formulation fixes directly:

1. **168 in-flight submissions are dropped from training.** A submission at
   395 days with no conclusion is not "unknown" — it tells us `T > 395`. That
   is a right-censored observation, and it is concentrated exactly where the
   point regressor is weakest: coverage of the p10–p90 interval for reviews
   that actually take > 700 days is 10% (`reports/metrics.json`,
   `quantile_interval_cv.coverage_actual_over_700d`).
2. **The "completed" sheet is truncated by the snapshot date.** A submission
   accepted in 2025 only appears as "completed" if it concluded before
   2026-07-31. The 2025 cohort (n=108) has p90 = 339 days versus 543 in 2024
   and 624 in 2023 — administrative truncation, not a real speedup. A
   regressor trained on this frame learns the truncation.
3. **The conclusion-date conditioning in `predict_under_review.py` is a
   clip.** `np.maximum(q, elapsed)` floors each quantile at the elapsed time;
   the right quantity is `S(t | T > elapsed)`, which falls out of a survival
   model with no special-casing.

## Model choice: why discrete time, not Cox or an AFT model

The decisive fact is the *shape* of the hazard, not just its average. Binning
`review_days` into 30-day windows on the completed data:

| Bin | n | share |
|---|---|---|
| (270, 300] days | 648 | 43% |
| (150, 180] days | 203 | 13% |

These are not smooth peaks — they are the regulatory service-standard targets
(300 days standard review, 180 days priority review), and `priority_review`
**moves** the spike from 300 to 180 rather than rescaling it.

- An **AFT model** (`survival:aft`, lognormal/Weibull) fits a smooth density
  and would spread both spikes into slopes — losing the single most
  informative structure in the data. Rejected as the primary model.
- A **single Cox model** keeps a non-parametric baseline hazard (so it *can*
  represent one spike), but one baseline hazard cannot move a spike from 300
  to 180 days for a subset of rows — that is a direct violation of the
  proportional-hazards assumption on `priority_review`. Only usable
  stratified (see the reference layer below).
- A **discrete-time model with the bin index as a feature** lets tree splits
  on `time_bin x priority_review` reproduce both spikes exactly, with no PH
  assumption anywhere. It also reuses the existing XGBoost + SHAP stack
  as-is. This is what was built.

## Bin width: empirical check

Before writing `src/survival.py`, the completed data was binned at 10, 15,
and 30 days and checked against the two known regulatory targets (180 and 300
days) — the width should isolate each target's spike in one bin, not split it
across a bin edge or dilute it with adjacent days.

| Width | Bin covering ~180d target | n | share | Bin covering ~300d target | n | share |
|---|---|---|---|---|---|---|
| 10 days | (170, 180] | 123 | 8.1% | (300, 310] / (290, 300] split | 339 / 314 | 22.4% / 20.7% |
| 15 days | (165, 180] | 126 | 8.3% | (300, 315] / (285, 300] split | 359 / 330 | 23.7% / 21.8% |
| **30 days** | **(150, 180]** | **203** | **13.4%** | **(270, 300]** | **648** | **42.8%** |

At 10- and 15-day widths the 300-day peak straddles a bin boundary (the exact
day-300 mass is split roughly in half between two adjacent bins because those
widths don't divide evenly into 300 from an origin of 0). At 30 days, with
bins anchored at multiples of 30 (`(0,30], (30,60], ..., (270,300]`), day 300
falls at the *right edge* of a bin and the entire spike lands in one bin —
matching the 43%/13% figures used to justify the model choice above. **30
days was adopted.**

Bins run `t = 1..36` (up to 1,080 days) plus an open-ended 37th bin for the 11
completed submissions (0.7%) that took longer. `src.survival.BIN_WIDTH_DAYS`,
`MAX_BIN`, `OPEN_BIN`.

## Model

- **Person-period expansion** (`src.survival.expand_person_periods`): for
  each submission, one row per bin `t = 1..exit_bin`; label `0` on every row
  except the last, which carries the cause (`0` censored, `1` approved, `2`
  not approved). In-flight submissions enter with `exit_bin` = the bin of
  their elapsed time at the snapshot, every row labelled `0`.
- **Model**: `XGBClassifier(objective="multi:softprob", num_class=3)` inside
  the existing `src.features.make_pipeline`, features = `C.FEATURE_COLUMNS +
  ["time_bin"]`.
- **Reconstruction** (`reconstruct_survival`): `S(t) = S(t-1)·h₀(t)`,
  `CIF_c(t) = CIF_c(t-1) + S(t-1)·h_c(t)`. Verified to sum to 1 at every
  horizon (`tests/test_survival.py::test_cif_and_survival_sum_to_one`).
- **Conditioning** (`conditional_outputs`): given a submission has survived
  through bin `e` (still in review, uncensored, unresolved),
  `S_cond(t) = S(t)/S(e)` and `CIF_c,cond(t) = (CIF_c(t) - CIF_c(e))/S(e)` for
  `t ≥ e`. This replaces the `np.maximum(q, elapsed)` clip in
  `predict_under_review.py` with the statistically correct quantity, and it
  falls out of the reconstruction with no special-casing — a case's
  conditional median is provably `≥` its elapsed days
  (`test_conditional_median_never_before_elapsed`), because the earliest bin
  where cumulative exit probability can be positive, once conditioned on
  bin `e`, is `e+1`.

### Two leakage guards

- **Expansion happens after the train/test split** (`train_test_case_split`
  then `expand_person_periods`), never before — otherwise the frequency
  encoder and any CV fold would see the same submission's rows on both sides.
- **Cross-validation is grouped by submission** (`GroupKFold` on `case_id` in
  `cross_validate_survival`), never a plain `KFold` over person-period rows.
  `tests/test_survival.py::test_no_case_shared_across_groupkfold_folds`
  checks this directly.

### Matching the test set to the original pipeline

`train_test_case_split` stratifies the completed cases on the exact same
column (`approved`, i.e. `C.TARGET_CLASSIFICATION`) and seed as
`src.models.train_test_split_frame` uses on the same underlying rows, so the
two pipelines' test sets are provably identical (checked in
`test_train_test_case_split_matches_original_pipeline_split` and asserted at
runtime in `scripts/compare_models.py`). The 168 censored in-flight
submissions are appended to the training split only.

## Reference layer (lifelines)

`scripts/train_survival.py` also fits, purely for interpretation and a
consistency check, **not** for prediction:

- **Kaplan-Meier** on "any exit" (approval or withdrawal), and
- **Aalen-Johansen** cause-specific cumulative incidence for approval and
  non-approval, and
- **Cox, stratified by `priority_review`** (each stratum gets its own
  baseline hazard, so its own spike — the workaround for the PH violation
  noted above).

The discrete-time model's own mean `S(t)` across all cases tracks the KM
curve within a few points at every checked horizon (150 / 300 / 600 days;
`reports/survival/metrics_survival.json` →
`discrete_model_mean_survival_by_horizon` vs `reference_lifelines.km_survival_at_horizon`),
which is the sanity check this layer exists for.

## Comparison protocol and result

`scripts/compare_models.py` runs both families on the identical seed-42
split, plus the decisive test: train on submissions accepted ≤ 2022 with a
pseudo-snapshot (2022-12-31) that right-censors — for the original pipeline,
*drops* — submissions still open at that date, then evaluate on submissions
accepted in 2023+ whose true outcome is known today. This is the only
configuration that reproduces the actual deployment condition (a live
snapshot mid-cohort) that the survival model is meant to correct for.

**Result (see `reports/comparison.md` for the full tables):**

| Holdout temporal (train ≤2022, eval 2023+) | Régresseur/classifieur | Survie |
|---|---|---|
| MAE médiane | 21.1 j | 24.5 j |
| Couverture p10–p90 pour les revues > 700 j réels | 0.0% | 7.1% |
| Brier (approbation) | 0.0495 | 0.0480 |

The survival model improves the >700-day coverage from 0% to 7.1% on the
holdout (n=14 long reviews in the eval set) and matches the classifier's
Brier score, but its median absolute error degrades by about 16% relative to
the regressor (24.5 vs 21.1 days) — partly a real effect of training on a
smaller, differently-shaped pool, partly the 30-day bin quantisation, which
mechanically floors the achievable MAE at half a bin width. On the same-split
comparison (`reports/comparison.md`, section 1), the survival model's overall
MAE is actually *lower* (71.5 vs 77.0) but its median AE is worse (30.0 vs
19.1) for the same quantisation reason.

**Verdict: stay on `main`.** The decision criterion (adopt only if >700-day
coverage improves *substantially* — more than 5 points — with **no**
degradation of median AE and no degradation of the approval Brier score) is
not met: coverage does improve, and the approval Brier score is preserved,
but the median AE — the metric the current pipeline is specifically good at —
degrades by more than the 15% slack allowed. The branch is kept as a
documented, working alternative: `scripts/train_survival.py`,
`scripts/predict_under_review_survival.py`, and
`scripts/compare_models.py` all run end-to-end and can be revisited if the
in-flight cohort grows (more censored cases sharpens the survival model's
advantage) or if the 30-day bin quantisation is refined (e.g. narrower bins
away from the two spikes, at the cost of a larger person-period table).

## Follow-up: continuous parametric AFT

The suspicion was that the median-AE loss came from the 30-day bins, so a
*continuous* AFT (`src/survival_aft.py`, lifelines `LogNormalAFTFitter` /
Weibull / log-logistic) was tested the same way — completed rows as events,
in-flight rows as right-censored — via `scripts/compare_aft.py`
(`reports/comparison_aft.md`, `reports/survival/comparison_aft.json`).

| Temporal holdout | MAE | Median AE | Coverage p10–p90 | Coverage >700 j |
|---|---|---|---|---|
| Regressor | 76.0 | **21.1** | 74 % | 0 % |
| Discrete survival (30-day bins) | 79.4 | 24.5 | 60 % | 7 % |
| AFT log-normal (continuous) | 75.4 | 39.2 | **89 %** | 0 % |
| AFT log-logistic (continuous) | — | ~31 | — | — |

The continuous AFT does **not** recover the median AE — it is worse than the
binned model. Reason: 43 % of submissions conclude in the (270, 300]-day bin,
exactly at the standard-review service target; the gradient-boosted regressor
reproduces that spike, a linear-in-covariates AFT smears it into a smooth
density (predicted median ~315–323 days vs the true 299). So the median-AE
penalty is **the cost of leaving XGBoost's flexibility**, not a bin artifact.

What the survival framing does buy, consistently across both variants: better
p10–p90 interval coverage under temporal drift (the in-flight cohort stays in
the fit as censored data instead of being dropped). The AFT log-normal's 89 %
temporal-holdout coverage beats both the regressor (74 %) and the `baysian`
branch's post-hoc re-projection (74 %) — so if a calibrated conclusion-date
*interval* for in-flight submissions is the goal, an AFT interval is the better
tool; for the *point* estimate, keep the regressor.

**Verdict unchanged: stay on `main` for the point prediction.** Neither
survival variant is adopted as the primary model. `src/survival_aft.py` and
`scripts/compare_aft.py` are kept alongside the discrete model as documented,
runnable alternatives.

## Files

| File | Purpose |
|---|---|
| `src/survival.py` | Case-frame construction, person-period expansion, model training, `S`/`CIF` reconstruction, conditional prediction, GroupKFold CV. |
| `src/survival_metrics.py` | Harrell's C, Uno's C, IPCW Brier score, time-dependent AUC — implemented directly rather than via `scikit-survival` (version-pinning risk not worth ~150 lines of code in a Python 3.13 / sklearn 1.9 environment). |
| `scripts/train_survival.py` | Trains the model, runs per-class SHAP, fits the lifelines reference layer, writes `models/survival/` and `reports/survival/metrics_survival.json`. |
| `scripts/predict_under_review_survival.py` | Forecast for the 168 in-flight submissions, conditional on elapsed time; writes `data/processed/under_review_forecast_survival.csv`. |
| `scripts/compare_models.py` | Same-split and temporal-holdout comparison; writes `reports/comparison.md` and `reports/survival/comparison.json`. |
| `tests/test_survival.py` | Expansion correctness, no-leakage guarantees, `S+CIF=1`, monotonicity, conditional-median-≥-elapsed, and a KM cross-check. |
| `src/survival_aft.py` | Continuous parametric AFT (lifelines) with the leak-safe preprocessor + z-scoring; `predict_quantiles` with `conditional_after`. |
| `scripts/compare_aft.py` | Same-split and temporal-holdout comparison of the AFT vs the regressor; writes `reports/comparison_aft.md`. |
| `tests/test_survival_aft.py` | Quantile monotonicity/bounds, `conditional_after` behaviour, constant-column drop. |

The original pipeline (`src/models.py`, `src/features.py`, `scripts/train_models.py`,
`scripts/predict_under_review.py`) is unchanged by this work — the survival code
is additive. `requirements.txt` gained `lifelines>=0.29` (discrete-model
reference layer + the AFT). `reports/metrics.json`, `models/*.joblib`, and the
primary `data/processed/*.csv` remain comparable.
