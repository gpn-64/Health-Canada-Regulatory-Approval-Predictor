"""Load, merge, and clean the Health Canada "submissions under review" file.

The raw workbook has four relevant sheets: completed and in-flight submissions,
each split into new drug submissions (NDS) and supplemental ones (SNDS). This
module turns the two *completed* sheets into a single tidy training frame and
exposes the two *under review* sheets for live inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config as C


@dataclass
class CleaningReport:
    """Counts collected while cleaning, printed by the build script and stored."""

    rows_nds: int = 0
    rows_snds: int = 0
    rows_negative_review_days: int = 0
    negative_control_numbers: list[int] = field(default_factory=list)
    rows_unmapped_outcome: int = 0
    unmapped_outcomes: dict[str, int] = field(default_factory=dict)
    rows_final: int = 0
    n_companies: int = 0
    n_therapeutic_areas: int = 0
    approved_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "rows_nds": self.rows_nds,
            "rows_snds": self.rows_snds,
            "rows_negative_review_days": self.rows_negative_review_days,
            "negative_control_numbers": self.negative_control_numbers,
            "rows_unmapped_outcome": self.rows_unmapped_outcome,
            "unmapped_outcomes": self.unmapped_outcomes,
            "rows_final": self.rows_final,
            "n_companies": self.n_companies,
            "n_therapeutic_areas": self.n_therapeutic_areas,
            "approved_rate": round(self.approved_rate, 4),
        }


def _first_col(df: pd.DataFrame, *prefixes: str) -> str | None:
    """Return the first column whose name starts with one of ``prefixes``.

    The "under review" sheets rename some columns (e.g. "Company Name (available
    for submissions accepted...)" and "Year, Month Accepted into Review*..."), so
    matching on a prefix keeps one loader working for every sheet.
    """
    for prefix in prefixes:
        for col in df.columns:
            if str(col).startswith(prefix):
                return col
    return None


def _read_sheet(sheet_name: str, submission_type: str) -> pd.DataFrame:
    """Read one sheet and normalise its shape (columns, class flags, type tag).

    Handles both the "completed" sheets (control number, exact dates, outcome)
    and the leaner "under review" sheets (no control number, month-level accept
    date, no outcome, fewer class columns).
    """
    df = pd.read_excel(C.RAW_FILE, sheet_name=sheet_name, header=C.HEADER_ROW)

    rename = {
        _first_col(df, C.COL_CONTROL_NUMBER): "control_number",
        _first_col(df, C.COL_INGREDIENTS): "medicinal_ingredients",
        _first_col(df, "Company Name"): "company_name",
        _first_col(df, C.COL_THERAPEUTIC_AREA): "therapeutic_area",
        _first_col(df, C.COL_DATE_ACCEPTED, "Year, Month Accepted"): "date_accepted",
        _first_col(df, C.COL_DATE_CONCLUDED): "date_concluded",
        _first_col(df, C.COL_OUTCOME): "outcome",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k is not None})
    if "control_number" not in df.columns:
        df["control_number"] = pd.NA

    # Class flags: "ü" -> 1, blank -> 0. Columns absent from this sheet (e.g.
    # new_active_substance on SNDS) are created as all-zero.
    for raw_col, slug in C.CLASS_COLUMN_SLUGS.items():
        if raw_col in df.columns:
            marked = df[raw_col].astype("string").str.strip() == C.CLASS_TRUE_MARKER
            df[slug] = marked.fillna(False).astype("int8")
        else:
            df[slug] = pd.Series(0, index=df.index, dtype="int8")

    df["submission_type"] = submission_type
    df["is_supplemental"] = int(submission_type == "SNDS")

    keep = [
        "control_number",
        "medicinal_ingredients",
        "company_name",
        "therapeutic_area",
        "submission_type",
        "is_supplemental",
        "date_accepted",
        *C.CLASS_FLAG_FEATURES,
    ]
    if "outcome" in df.columns:
        keep.append("outcome")
    if "date_concluded" in df.columns:
        keep.append("date_concluded")
    return df[keep]


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_accepted"] = pd.to_datetime(df["date_accepted"])
    df["accept_year"] = df["date_accepted"].dt.year
    df["accept_month"] = df["date_accepted"].dt.month
    df["accept_quarter"] = df["date_accepted"].dt.quarter
    df["n_ingredients"] = (
        df["medicinal_ingredients"].fillna("").astype(str).apply(
            lambda s: len([p for p in s.split(",") if p.strip()])
        )
    )
    return df


def load_completed(report: CleaningReport | None = None) -> pd.DataFrame:
    """Return the merged, cleaned frame of completed submissions.

    Cleaning steps:
      * union the NDS and SNDS sheets (class flags aligned, type tagged);
      * parse dates and compute ``review_days``;
      * drop rows with a negative ``review_days`` (data-entry errors);
      * map ``outcome`` to a binary ``approved`` target, dropping rows whose
        outcome is outside the known mapping.
    """
    report = report or CleaningReport()

    frames = [
        _read_sheet(sheet, kind) for sheet, kind in C.COMPLETED_SHEETS.items()
    ]
    report.rows_nds = len(frames[0])
    report.rows_snds = len(frames[1])
    df = pd.concat(frames, ignore_index=True)

    df["date_concluded"] = pd.to_datetime(df["date_concluded"])
    df = _add_derived_columns(df)
    df["review_days"] = (df["date_concluded"] - df["date_accepted"]).dt.days

    negatives = df["review_days"] < 0
    report.rows_negative_review_days = int(negatives.sum())
    report.negative_control_numbers = df.loc[negatives, "control_number"].tolist()
    df = df.loc[~negatives].copy()

    outcome_stripped = df["outcome"].astype("string").str.strip()
    df["approved"] = pd.NA
    df.loc[outcome_stripped.isin(C.APPROVED_OUTCOMES), "approved"] = 1
    df.loc[outcome_stripped.isin(C.NOT_APPROVED_OUTCOMES), "approved"] = 0

    unmapped = df["approved"].isna()
    report.rows_unmapped_outcome = int(unmapped.sum())
    report.unmapped_outcomes = (
        outcome_stripped[unmapped].value_counts().to_dict()
    )
    df = df.loc[~unmapped].copy()
    df["approved"] = df["approved"].astype("int8")

    report.rows_final = len(df)
    report.n_companies = df["company_name"].nunique()
    report.n_therapeutic_areas = df["therapeutic_area"].nunique()
    report.approved_rate = float(df["approved"].mean())

    return df.reset_index(drop=True)


def load_under_review() -> pd.DataFrame:
    """Return in-flight submissions (accepted date only, no outcome yet)."""
    frames = [
        _read_sheet(sheet, kind) for sheet, kind in C.UNDER_REVIEW_SHEETS.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    df = _add_derived_columns(df)
    df["review_days"] = pd.NA
    df["approved"] = pd.NA
    return df.reset_index(drop=True)
