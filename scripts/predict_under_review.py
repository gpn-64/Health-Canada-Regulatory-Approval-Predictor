"""Forecast for the submissions still under review: approval probability and an
estimated conclusion date, both with an uncertainty band.

Run from the project root (after build_dataset.py):

    python scripts/predict_under_review.py [--n-boot 200] [--level 0.90]

Writes data/processed/under_review_forecast.csv and prints highlights.

Conclusion date = date_accepted + predicted review_days (p10 / p50 / p90 from
quantile regression). Because these submissions have *already* been in review for
a known time, every quantile is floored at the elapsed duration — the review
cannot end in the past — and a submission already past its unconditioned p90 is
flagged (`overdue_vs_model`): the model expected it done by now.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import load_under_review  # noqa: E402
from src.features import build_features  # noqa: E402
from src import models as M  # noqa: E402
from src.uncertainty import (  # noqa: E402
    IntervalConfig,
    bootstrap_approval_proba,
    coverage_diagnostic,
    summarize,
)

OUT_FILE = C.PROCESSED_DIR / "under_review_forecast.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--level", type=float, default=0.90)
    args = ap.parse_args()

    if not C.CLEAN_FILE.exists():
        raise SystemExit("Run scripts/build_dataset.py first.")
    clean = pd.read_csv(C.CLEAN_FILE, parse_dates=["date_accepted", "date_concluded"])
    pending = load_under_review()

    X_train = build_features(clean)
    y_train = clean[C.TARGET_CLASSIFICATION]
    y_days = clean[C.TARGET_REGRESSION]
    X_pending = build_features(pending)

    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    accepted = pd.to_datetime(pending["date_accepted"])
    days_elapsed = (snap - accepted).dt.days.clip(lower=0)

    # --- Conclusion date: quantile regression, floored at elapsed time ---
    print("Fitting quantile regressors for the review-time interval...")
    qmodels = M.train_quantile_regressors(X_train, y_days)
    q = M.predict_review_days_quantiles(qmodels, X_pending).reset_index(drop=True)
    elapsed = days_elapsed.to_numpy()
    status = np.where(
        elapsed > q["p90"].to_numpy(), "overdue",
        np.where(elapsed > q["p50"].to_numpy(), "late", "on_track"),
    )
    q_floored = pd.DataFrame(
        np.maximum(q.to_numpy(), elapsed[:, None]), columns=q.columns
    )

    # --- Approval probability: bootstrap interval ---
    cfg = IntervalConfig(n_boot=args.n_boot, level=args.level)
    print(f"Bootstrapping {cfg.n_boot} approval classifiers (level={cfg.level:.0%})...")
    preds, oob = bootstrap_approval_proba(
        X_train, y_train, X_pending, cfg, collect_oob=True
    )
    bands = summarize(preds, cfg.level).reset_index(drop=True)

    def _date(days: np.ndarray) -> pd.Series:
        return (accepted.reset_index(drop=True) + pd.to_timedelta(np.round(days), unit="D")).dt.date

    out = pd.DataFrame(
        {
            "submission_type": pending["submission_type"],
            "medicinal_ingredients": pending["medicinal_ingredients"],
            "company_name": pending["company_name"],
            "therapeutic_area": pending["therapeutic_area"],
            "date_accepted": accepted.dt.strftime("%Y-%m"),
            "days_in_review_so_far": days_elapsed,
            "approval_proba": bands["approval_proba"].round(3),
            "approval_ci_low": bands["ci_low"].round(3),
            "approval_ci_high": bands["ci_high"].round(3),
            "est_conclusion_date": _date(q_floored["p50"].to_numpy()),
            "conclusion_date_earliest": _date(q_floored["p10"].to_numpy()),
            "conclusion_date_latest": _date(q_floored["p90"].to_numpy()),
            "remaining_days_p50": np.maximum(q_floored["p50"].to_numpy() - elapsed, 0).round(),
            "status": status,
        }
    ).sort_values(["status", "est_conclusion_date"])
    out.to_csv(OUT_FILE, index=False)

    # --- diagnostics ---
    qcv = M.cross_validate_quantiles(X_train, y_days)
    print("\nReview-time interval (out-of-fold on completed data)")
    print("-" * 72)
    print(f"  nominal coverage {qcv['nominal_coverage']:.0%}  |  "
          f"empirical {qcv['empirical_coverage']:.0%}  |  "
          f"median width {qcv['median_interval_days']:.0f} days")
    print(f"  coverage for reviews that actually ran >700 days: "
          f"{qcv['coverage_actual_over_700d']:.0%}  <-- the long tail is missed")

    print("\nApproval probability reliability (out-of-bag on completed data)")
    print("-" * 72)
    print(coverage_diagnostic(oob, y_train.to_numpy()).to_string(index=False))

    show = ["submission_type", "medicinal_ingredients", "company_name",
            "days_in_review_so_far", "approval_proba", "approval_ci_low", "approval_ci_high",
            "est_conclusion_date", "conclusion_date_earliest", "conclusion_date_latest"]
    counts = out["status"].value_counts().to_dict()
    print(f"\nStatus of the {len(out)} in-flight submissions: "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("\nSoonest expected conclusions (status = on_track)")
    print("-" * 72)
    print(out[out.status == "on_track"][show].head(12).to_string(index=False))
    print("\n'late' = past the p50 estimate but within p90 · 'overdue' = past p90 "
          "(metadata does not explain the long review); for both, the date is reported\n"
          "as 'not before the snapshot' and only conclusion_date_latest is informative.")
    print(f"\nWrote {OUT_FILE.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
