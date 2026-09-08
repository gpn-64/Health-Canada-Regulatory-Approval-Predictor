"""Train both models, run SHAP, and write artifacts + metrics + predictions.

Run from the project root (after build_dataset.py):

    python scripts/train_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import load_under_review  # noqa: E402
from src.explain import explain_model  # noqa: E402
from src.features import build_features  # noqa: E402
from src import models as M  # noqa: E402


def _load_clean() -> pd.DataFrame:
    if not C.CLEAN_FILE.exists():
        raise SystemExit("Run scripts/build_dataset.py first.")
    return pd.read_csv(C.CLEAN_FILE, parse_dates=["date_accepted", "date_concluded"])


def _feature_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Mean review time and approval rate with vs without each class flag —
    the raw material for the regulatory reading in docs/methodology.md."""
    rows = []
    for flag in C.CLASS_FLAG_FEATURES + ["is_supplemental"]:
        on = df[df[flag] == 1]
        off = df[df[flag] == 0]
        rows.append(
            {
                "flag": flag,
                "n_with": len(on),
                "median_review_days_with": on["review_days"].median(),
                "median_review_days_without": off["review_days"].median(),
                "approval_rate_with": on["approved"].mean() if len(on) else np.nan,
                "approval_rate_without": off["approved"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("median_review_days_with", ascending=False)


def main() -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_clean()
    train_df, test_df = M.train_test_split_frame(df, stratify_col=C.TARGET_CLASSIFICATION)

    X_train = build_features(train_df)
    X_test = build_features(test_df)
    y_reg_train, y_reg_test = train_df[C.TARGET_REGRESSION], test_df[C.TARGET_REGRESSION]
    y_clf_train, y_clf_test = train_df[C.TARGET_CLASSIFICATION], test_df[C.TARGET_CLASSIFICATION]

    print("Cross-validating (5-fold, train only)...")
    reg_cv = M.cross_validate_regressor(X_train, y_reg_train)
    reg_quantile_cv = M.cross_validate_quantiles(X_train, y_reg_train)
    clf_cv = M.cross_validate_classifier(X_train, y_clf_train)

    print("Fitting final models...")
    regressor = M.train_regressor(X_train, y_reg_train)
    quantile_regressors = M.train_quantile_regressors(X_train, y_reg_train)
    classifier = M.train_classifier(X_train, y_clf_train)

    reg_test = M.evaluate_regressor(regressor, X_test, y_reg_test, y_reg_train)
    clf_test = M.evaluate_classifier(classifier, X_test, y_clf_test, y_clf_train)

    joblib.dump(regressor, C.MODELS_DIR / "regressor.joblib")
    joblib.dump(classifier, C.MODELS_DIR / "classifier.joblib")
    for a, pipe in quantile_regressors.items():
        joblib.dump(pipe, C.MODELS_DIR / f"regressor_q{int(a * 100)}.joblib")
    regressor.named_steps["model"].get_booster().save_model(str(C.MODELS_DIR / "regressor.ubj"))
    classifier.named_steps["model"].get_booster().save_model(str(C.MODELS_DIR / "classifier.ubj"))

    print("Running SHAP...")
    reg_shap = explain_model(regressor, X_test, "regressor", C.REPORTS_DIR / "shap_regressor.csv")
    clf_shap = explain_model(classifier, X_test, "classifier", C.REPORTS_DIR / "shap_classifier.csv")
    _feature_effects(df).to_csv(C.REPORTS_DIR / "feature_effects.csv", index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "regressor": {
            "note": "trained on log1p(review_days); metrics are natural-scale days",
            "cv": reg_cv,
            "quantile_interval_cv": reg_quantile_cv,
            "test": reg_test,
            "beats_baseline": reg_test["mae"] < reg_test["baseline_mae"],
            "top_features": reg_shap.head(8).to_dict("records"),
        },
        "classifier": {
            "cv": clf_cv,
            "test": clf_test,
            "beats_baseline": clf_test["roc_auc"] > 0.65,
            "top_features": clf_shap.head(8).to_dict("records"),
        },
    }
    C.METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    _write_predictions(df, regressor, quantile_regressors, classifier)

    print("\nMetrics")
    print("-" * 60)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {C.METRICS_FILE.relative_to(C.PROJECT_ROOT)}")
    print(f"Wrote {C.PREDICTIONS_FILE.relative_to(C.PROJECT_ROOT)}")
    print("Wrote models + SHAP figures under models/ and reports/")


def _write_predictions(clean_df: pd.DataFrame, regressor, quantile_regressors, classifier) -> None:
    """One row per submission (completed + under review) with predictions."""
    completed = clean_df.copy()
    completed["dataset"] = "completed"

    pending = load_under_review()
    pending["dataset"] = "under_review"

    frames = []
    for part in (completed, pending):
        X = build_features(part)
        accepted = pd.to_datetime(part["date_accepted"].to_numpy())
        q = M.predict_review_days_quantiles(quantile_regressors, X)
        out = pd.DataFrame(
            {
                "control_number": part["control_number"].to_numpy(),
                "dataset": part["dataset"].to_numpy(),
                "submission_type": part["submission_type"].to_numpy(),
                "company_name": part["company_name"].to_numpy(),
                "therapeutic_area": part["therapeutic_area"].to_numpy(),
                "date_accepted": accepted,
                "actual_review_days": part["review_days"].to_numpy(),
                "actual_approved": part["approved"].to_numpy(),
                "predicted_review_days": M.predict_review_days(regressor, X).round(1),
                "review_days_p10": q["p10"].round(1),
                "review_days_p50": q["p50"].round(1),
                "review_days_p90": q["p90"].round(1),
                "predicted_approval_proba": classifier.predict_proba(X)[:, 1].round(4),
            }
        )
        out["predicted_conclusion_date"] = accepted + pd.to_timedelta(q["p50"].round(), unit="D")
        out["conclusion_date_p10"] = accepted + pd.to_timedelta(q["p10"].round(), unit="D")
        out["conclusion_date_p90"] = accepted + pd.to_timedelta(q["p90"].round(), unit="D")
        out["review_days_residual"] = (
            pd.to_numeric(out["actual_review_days"], errors="coerce")
            - out["predicted_review_days"]
        )
        frames.append(out)

    pd.concat(frames, ignore_index=True).to_csv(C.PREDICTIONS_FILE, index=False)


if __name__ == "__main__":
    main()
