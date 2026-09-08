"""Shared configuration: paths, constants, and column mappings.

Everything the pipeline needs to locate files and interpret the raw Health Canada
spreadsheet lives here so the rest of the code carries no magic strings.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE = RAW_DIR / "submissions-under-review-2026-07.xlsx"
CLEAN_FILE = PROCESSED_DIR / "submissions_clean.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "predictions.csv"
METRICS_FILE = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42

# Effective date of the snapshot (from the file name / publication). Used to age
# in-flight submissions ("months already in review").
SNAPSHOT_DATE = "2026-07-31"

# --- Source spreadsheet layout ------------------------------------------------
# The real header sits on row 4 of every sheet (0-indexed row 3).
HEADER_ROW = 3

COMPLETED_SHEETS = {
    "New drug sub's completed": "NDS",
    "Supplemental sub's completed": "SNDS",
}
UNDER_REVIEW_SHEETS = {
    "New drug sub's under review": "NDS",
    "Supplemental sub's under review": "SNDS",
}

# --- Column names in the raw sheets -----------------------------------------
COL_CONTROL_NUMBER = "Submission Control Number"
COL_INGREDIENTS = "Medicinal Ingredient(s)"
COL_COMPANY = "Company Name"
COL_THERAPEUTIC_AREA = "Therapeutic Area"
COL_DATE_ACCEPTED = "Date Accepted into Review"
COL_DATE_CONCLUDED = "Date Submission Concluded"
COL_OUTCOME = "Outcome of Submission (Hyperlinked if applicable)"

# The "Submission 'Class': ..." boolean columns, mapped to short feature slugs.
# A cell holds "ü" (Wingdings check mark) when true and is empty otherwise.
# NDS carries all 12; SNDS lacks "new_active_substance".
CLASS_COLUMN_SLUGS = {
    "Submission 'Class': Extraordinary use submission?": "extraordinary_use",
    "Submission 'Class': New active substance?": "new_active_substance",
    "Submission 'Class': Biosimilar?": "biosimilar",
    "Submission 'Class': Being reviewed under the Priority Review Policy?": "priority_review",
    "Submission 'Class': Being reviewed under the Notice of Compliance with Conditions Guidance?": "noc_c",
    "Submission 'Class': Being reviewed under theSubmissions Relying on Third-Party Data Guidance?": "third_party_data",
    "Submission 'Class': Part of an 'aligned review' with a health technology assessment (HTA) organization?": "hta_aligned",
    "Submission 'Class': For use in relation to COVID-19?": "covid19",
    "Submission 'Class': Reviewed under Project Orbis Type A?": "project_orbis_a",
    "Submission 'Class': Reviewed under Project Orbis Type B?": "project_orbis_b",
    "Submission 'Class': Reviewed under Project Orbis Type C?": "project_orbis_c",
    "Submission 'Class': Reviewed under the Access Consortium: New Active Substance Work Sharing Initiative?": "access_consortium",
}
CLASS_FLAG_FEATURES = list(CLASS_COLUMN_SLUGS.values())
CLASS_TRUE_MARKER = "ü"

# --- Outcome mapping (classifier target) -----------------------------------
# `approved` = 1 when the submission reached an authorisation to market, else 0.
APPROVED_OUTCOMES = {
    "Issued Notice of Compliance",
    "Issued Notice of Compliance under the NOC/c Guidance",
    "Authorized under Interim Order",
}
NOT_APPROVED_OUTCOMES = {
    "Cancelled by sponsor",
    "Issued Notice of Non-compliance - Withdrawal",
    "Issued Notice of Deficiency - Withdrawal",
    "Interim Order expired",
}

# --- Feature groups --------------------------------------------------------
FREQUENCY_ENCODED_FEATURES = ["company_name", "therapeutic_area"]
NUMERIC_FEATURES = [
    "is_supplemental",
    "accept_year",
    "accept_month",
    "accept_quarter",
    "n_ingredients",
]
FEATURE_COLUMNS = CLASS_FLAG_FEATURES + NUMERIC_FEATURES + FREQUENCY_ENCODED_FEATURES

TARGET_REGRESSION = "review_days"
TARGET_CLASSIFICATION = "approved"
