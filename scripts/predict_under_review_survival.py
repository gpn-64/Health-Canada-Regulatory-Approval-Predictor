"""Forecast for the submissions still under review, using the discrete-time
competing-risks survival model instead of the point regressor + bootstrap
classifier.

Mirrors scripts/predict_under_review.py, but every quantity is a native
survival-model output conditional on time already elapsed — no clipping.
``S(t | T > elapsed)`` replaces the ``np.maximum(q, elapsed)`` clip in the
original script, and the same conditioning removes the need for a separate
"overdue" heuristic: the model's own hazard already knows a case has run past
the day-300 peak once elapsed exceeds it.

Run from the project root (after build_dataset.py):

    python scripts/predict_under_review_survival.py

Writes data/processed/under_review_forecast_survival.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import survival as S  # noqa: E402


def main() -> None:
    print("Building case frame and training the survival model on all completed + censored cases...")
    case_df = S.build_case_frame()
    long_df = S.expand_person_periods(case_df)
    pipeline = S.train_survival_pipeline(long_df)

    pending = case_df[case_df["dataset"] == "under_review"].reset_index(drop=True)
    elapsed_days = pending["elapsed_days"].to_numpy()

    print(f"Scoring {len(pending)} in-flight submissions...")
    outputs = S.predict_case_outputs(pipeline, pending, elapsed_days=elapsed_days)

    accepted = pd.to_datetime(pending["date_accepted"])

    def _date(days: np.ndarray) -> pd.Series:
        return (accepted.reset_index(drop=True) + pd.to_timedelta(np.round(days), unit="D")).dt.date

    out = pd.DataFrame(
        {
            "submission_type": pending["submission_type"],
            "company_name": pending["company_name"],
            "therapeutic_area": pending["therapeutic_area"],
            "date_accepted": accepted.dt.strftime("%Y-%m"),
            "days_in_review_so_far": elapsed_days.round().astype(int),
            "p_approval_ultimate": outputs.p_approval_ultimate.round(3),
            "p_approval_by_700d": outputs.p_approval_by_horizon.round(3),
            "est_conclusion_date": _date(outputs.median_day),
            "conclusion_date_earliest": _date(outputs.p10_day),
            "conclusion_date_latest": _date(outputs.p90_day),
            "remaining_days_p50": np.maximum(outputs.median_day - elapsed_days, 0).round(),
        }
    ).sort_values("est_conclusion_date")
    out.to_csv(S.UNDER_REVIEW_FORECAST_SURVIVAL_FILE, index=False)

    print(f"\nWrote {S.UNDER_REVIEW_FORECAST_SURVIVAL_FILE.relative_to(C.PROJECT_ROOT)}")
    print("\nSoonest expected conclusions")
    print("-" * 72)
    show = [
        "company_name", "days_in_review_so_far", "p_approval_ultimate",
        "est_conclusion_date", "conclusion_date_earliest", "conclusion_date_latest",
    ]
    print(out[show].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
