# Data dictionary

## Source

- **File:** `data/raw/submissions-under-review-2026-07.xlsx` — Health Canada,
  *Drug and health product submissions under review* (public, refreshed monthly).
  Field descriptions: <https://www.canada.ca/en/health-canada/services/drug-health-product-review-approval/submissions-under-review.html>
- **Snapshot:** 2026-07-31. A single snapshot — no automated refresh in V1.
- **Real header row:** row 4 of every sheet (`header=3` in pandas).

### Sheets used

| Sheet | Rows | Role |
|---|---|---|
| `New drug sub's completed` (NDS) | 808 | training data |
| `Supplemental sub's completed` (SNDS) | 708 | training data |
| `New drug sub's under review` (NDS) | 89 | live inference only |
| `Supplemental sub's under review` (SNDS) | 79 | live inference only |

NDS = new active substance never approved in Canada. SNDS = new use / change to an
already-authorised drug.

## Cleaned table — `data/processed/submissions_clean.csv`

- **Source:** the two `... completed` sheets, merged.
- **Grain:** one row per completed submission (`control_number` unique).
- **Rows:** 1,515 (808 NDS + 708 SNDS − 1 dropped, see rules below).

| Column | Type | Description | Rule / calculation |
|---|---|---|---|
| `control_number` | int | Submission Control Number | primary key |
| `medicinal_ingredients` | str | Active ingredient(s), comma-separated | as-is |
| `company_name` | str | Sponsor | as-is |
| `therapeutic_area` | str | WHO ATC-based area | as-is (78–79 distinct values) |
| `submission_type` | str | `NDS` or `SNDS` | from the source sheet |
| `is_supplemental` | int 0/1 | 1 for SNDS | feature |
| `date_accepted` | date | Date accepted into review | parsed |
| `date_concluded` | date | Date the review concluded | parsed |
| `review_days` | int | **Regression target.** Calendar days in review | `(date_concluded − date_accepted).days` |
| `outcome` | str | Outcome of submission (7 categories) | as-is |
| `approved` | int 0/1 | **Classification target.** Reached a marketing authorisation | see outcome mapping below |
| `<class flag>` × 12 | int 0/1 | Submission-class markers (see below) | `"ü"` → 1, blank → 0 |
| `accept_year` / `accept_month` / `accept_quarter` | int | Calendar parts of `date_accepted` | feature |
| `n_ingredients` | int | Count of comma-separated items in `medicinal_ingredients` | feature |

### Submission-class flags (12)

Raw columns are named `Submission 'Class': <question>?` and hold the Wingdings
check mark `"ü"` when true, blank otherwise.

| Slug | Meaning |
|---|---|
| `extraordinary_use` | Extraordinary Use New Drug |
| `new_active_substance` | New active substance (NDS sheet only; SNDS → 0) |
| `biosimilar` | Biosimilar biologic drug |
| `priority_review` | Priority Review Policy |
| `noc_c` | Notice of Compliance with Conditions guidance |
| `third_party_data` | Submissions Relying on Third-Party Data guidance |
| `hta_aligned` | Aligned review with an HTA organisation (CDA / INESSS) |
| `covid19` | For use in relation to COVID-19 |
| `project_orbis_a` / `_b` / `_c` | Project Orbis review type |
| `access_consortium` | Access Consortium NAS Work-Sharing Initiative |

### Outcome → `approved` mapping

| Outcome | `approved` |
|---|---|
| Issued Notice of Compliance | 1 |
| Issued Notice of Compliance under the NOC/c Guidance | 1 |
| Authorized under Interim Order | 1 |
| Cancelled by sponsor | 0 |
| Issued Notice of Non-compliance – Withdrawal | 0 |
| Issued Notice of Deficiency – Withdrawal | 0 |
| Interim Order expired | 0 |

Rows whose outcome is outside this list are dropped from training and counted in
the cleaning report (`rows_unmapped_outcome`). None in the 2026-07 snapshot.

## Business rules

- **Negative `review_days`:** dropped. 1 row in the 2026-07 snapshot — Submission
  Control Number `300215`, SNDS, `date_concluded` 7 days before `date_accepted`
  (data-entry error; the recorded outcome is an NOC). Logged in the cleaning
  report as `rows_negative_review_days` / `negative_control_numbers`.
- **`company_name` / `therapeutic_area`:** high cardinality (≈280 / ≈79) — never
  one-hot encoded; frequency-encoded inside the model pipeline (fit on training
  folds only).
- **`under review` sheets:** no control number, no outcome, and a month-level
  `Year, Month Accepted into Review` date; fewer class columns (no Project Orbis
  / Access Consortium). The loader fills the missing class flags with 0 and takes
  the month date as `date_accepted`. Used for prediction only, never for training.

## Prediction export — `data/processed/predictions.csv`

One row per submission (1,515 completed + 168 under review).

| Column | Description |
|---|---|
| `control_number`, `dataset` (`completed` / `under_review`), `submission_type`, `company_name`, `therapeutic_area`, `date_accepted` | identifiers |
| `actual_review_days`, `actual_approved` | blank for `under_review` rows |
| `predicted_review_days` | point regressor output (natural scale) |
| `review_days_p10` / `review_days_p50` / `review_days_p90` | quantile-regression predictions (natural scale, sorted monotone) |
| `predicted_conclusion_date` | `date_accepted` + `review_days_p50` |
| `conclusion_date_p10` / `conclusion_date_p90` | `date_accepted` + the p10 / p90 day counts |
| `predicted_approval_proba` | classifier P(`approved` = 1) |
| `review_days_residual` | `actual − predicted_review_days` (completed rows only) |

Note: `predictions.csv` does **not** apply the elapsed-time flooring — that logic
is in `under_review_forecast.csv` below.

## In-flight forecast — `data/processed/under_review_forecast.csv`

One row per in-flight submission (168), from `scripts/predict_under_review.py`.
Identified by `medicinal_ingredients` + `company_name` + `submission_type` (the
`... under review` sheets carry no control number).

| Column | Description |
|---|---|
| `submission_type`, `medicinal_ingredients`, `company_name`, `therapeutic_area` | identifiers |
| `date_accepted` | month accepted into review (`YYYY-MM`) |
| `days_in_review_so_far` | days between `date_accepted` and the snapshot date (`SNAPSHOT_DATE` = 2026-07-31) |
| `approval_proba` | mean predicted P(`approved`) across 200 bootstrap models |
| `approval_ci_low` / `approval_ci_high` | 5th / 95th percentile of the bootstrap predictions (90 % interval) |
| `est_conclusion_date` | `date_accepted` + predicted p50 review days, floored at today |
| `conclusion_date_earliest` / `conclusion_date_latest` | same for p10 / p90 |
| `remaining_days_p50` | `max(p50 − days_in_review_so_far, 0)` |
| `status` | `on_track` (elapsed < p50) / `late` (p50 ≤ elapsed ≤ p90) / `overdue` (elapsed > p90); in the 2026-07 snapshot: 130 / 20 / 18 |

Here "elapsed" is `days_in_review_so_far` and p50 / p90 are the *unfloored*
quantile predictions. For `late` and `overdue` rows `est_conclusion_date` and
`conclusion_date_earliest` collapse to the snapshot date — only
`conclusion_date_latest` carries information.
