"""Train the discrete-time competing-risks survival model, run SHAP, and write
a reference layer (Kaplan-Meier / Aalen-Johansen / stratified Cox via
lifelines) used only for interpretation and a consistency check against the
discrete-time model's own survival curve.

Mirrors scripts/train_models.py but writes into models/survival/ and
reports/survival/ — the original pipeline's artifacts are untouched.

Run from the project root (after build_dataset.py):

    python scripts/train_survival.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import survival as S  # noqa: E402
from src.features import output_feature_names  # noqa: E402
from src.survival_metrics import (  # noqa: E402
    brier_ipcw,
    concordance_harrell,
    concordance_uno,
    time_dependent_auc,
)

CLASS_NAMES = {0: "still_in_review", 1: "approved", 2: "not_approved"}


def _explain_survival(pipeline, long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """SHAP summary (bar + beeswarm) per class, adapted from src/explain.py's
    explain_model: that helper assumes a single-output (binary) classifier,
    but shap.TreeExplainer returns a 3-D array (n, features, 3 classes) for
    this multi:softprob model, so mean|SHAP| and the plots are computed per
    class here instead of reusing explain_model unmodified.
    """
    names = output_feature_names() + [S.TIME_COL]
    X = long_df[S.SURVIVAL_FEATURE_COLUMNS]
    X_enc = pd.DataFrame(
        pipeline.named_steps["prep"].transform(X), columns=names, index=X.index
    )
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(X_enc)

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for c, cname in CLASS_NAMES.items():
        sv_c = shap_values[:, :, c]
        mean_abs = np.abs(sv_c).mean(axis=0)
        table = (
            pd.DataFrame({"feature": X_enc.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        table.to_csv(S.SURVIVAL_REPORTS_DIR / f"shap_survival_{cname}.csv", index=False)
        tables[cname] = table

        for plot_type, suffix in [("bar", "bar"), ("dot", "beeswarm")]:
            plt.figure()
            shap.summary_plot(sv_c, X_enc, plot_type=plot_type, show=False, max_display=20)
            plt.title(f"SHAP ({suffix}) — survival [{cname}]")
            plt.tight_layout()
            plt.savefig(C.FIGURES_DIR / f"shap_survival_{cname}_{suffix}.png", dpi=120)
            plt.close()
    return tables


def _reference_lifelines(case_df: pd.DataFrame) -> dict:
    """Kaplan-Meier, Aalen-Johansen cause-specific incidence, and a Cox model
    stratified by submission type — non-parametric checks, not used for
    prediction. ``duration`` = exit bin's day estimate (bin resolution, to
    stay comparable with the discrete-time model); ``event`` = 1 unless the
    case is still under review at the snapshot (right-censored).
    """
    df = case_df.copy()
    df["duration"] = S.bin_to_day(df["exit_bin"])
    df["event_any"] = (df["cause"] != S.CAUSE_CENSORED).astype(int)

    km = KaplanMeierFitter()
    km.fit(df["duration"], df["event_any"], label="KM (any exit)")

    aj1 = AalenJohansenFitter()
    aj1.fit(df["duration"], df["cause"], event_of_interest=S.CAUSE_APPROVED)
    aj2 = AalenJohansenFitter()
    aj2.fit(df["duration"], df["cause"], event_of_interest=S.CAUSE_NOT_APPROVED)

    cox = CoxPHFitter()
    cox_df = df[["duration", "event_any", "priority_review", "is_supplemental", "n_ingredients"]].copy()
    cox.fit(cox_df, duration_col="duration", event_col="event_any", strata=["priority_review"])

    horizons = list(range(60, 1081, 60))
    km_at = {h: float(km.survival_function_at_times(h).iloc[0]) for h in horizons}
    aj1_cif = aj1.cumulative_density_.iloc[:, 0]
    aj2_cif = aj2.cumulative_density_.iloc[:, 0]
    aj1_at = {h: float(aj1_cif.asof(h)) for h in horizons}
    aj2_at = {h: float(aj2_cif.asof(h)) for h in horizons}

    return {
        "km_survival_at_horizon": km_at,
        "aalen_johansen_cif_approved_at_horizon": aj1_at,
        "aalen_johansen_cif_not_approved_at_horizon": aj2_at,
        "cox_stratified_by_priority_review": {
            "hazard_ratios": cox.hazard_ratios_.to_dict(),
            "p_values": cox.summary["p"].to_dict(),
            "concordance": float(cox.concordance_index_),
        },
        "km_median_days": float(km.median_survival_time_),
    }


def _discrete_model_km_check(pipeline, case_df: pd.DataFrame) -> dict:
    """Mean S(t) predicted by the discrete-time model across all cases at
    each 60-day horizon, next to lifelines' non-parametric KM — should track
    closely if the model's hazards are sane."""
    hazards = S.predict_hazards(pipeline, case_df, max_bin=S.N_BINS)
    recon = S.reconstruct_survival(hazards)
    horizons_bins = list(range(2, S.MAX_BIN + 1, 2))  # every 60 days
    mean_S = {int(S.bin_to_day(b)): float(recon["S"][:, b].mean()) for b in horizons_bins}
    return mean_S


def main() -> None:
    S.SURVIVAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    S.SURVIVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building case frame (completed + under review)...")
    case_df = S.build_case_frame()
    train_case, test_case = S.train_test_case_split(case_df)
    print(f"  train cases: {len(train_case)} (incl. {(train_case['dataset']=='under_review').sum()} censored)")
    print(f"  test cases:  {len(test_case)} (completed only)")

    print("Cross-validating (GroupKFold by case_id, train only)...")
    cv = S.cross_validate_survival(train_case)

    print("Fitting final model on the full training split...")
    long_train = S.expand_person_periods(train_case)
    pipeline = S.train_survival_pipeline(long_train)

    # --- Test-set evaluation (completed cases only; matches original pipeline's test rows) ---
    outputs = S.predict_case_outputs(pipeline, test_case, elapsed_days=np.zeros(len(test_case)))
    actual_days = S.bin_to_day(test_case["exit_bin"].to_numpy())
    event = (test_case["cause"] != S.CAUSE_CENSORED).to_numpy().astype(int)
    approved = (test_case["cause"] == S.CAUSE_APPROVED).to_numpy().astype(int)

    inside = (actual_days >= outputs.p10_day) & (actual_days <= outputs.p90_day)
    over_700 = actual_days > 700
    risk = -outputs.median_day

    from sklearn.metrics import brier_score_loss, roc_auc_score

    test_metrics = {
        "mae_days": float(np.mean(np.abs(actual_days - outputs.median_day))),
        "median_ae_days": float(np.median(np.abs(actual_days - outputs.median_day))),
        "interval_empirical_coverage": float(inside.mean()),
        "coverage_actual_over_700d": float(inside[over_700].mean()) if over_700.any() else None,
        "harrell_c": concordance_harrell(actual_days, event, risk),
        "uno_c": concordance_uno(actual_days, event, risk),
        "time_dependent_auc_700d": time_dependent_auc(actual_days, event, risk, 700),
        "roc_auc_approval": float(roc_auc_score(approved, outputs.p_approval_ultimate)),
        "brier_approval": float(brier_score_loss(approved, outputs.p_approval_ultimate)),
        "brier_ipcw_700d": brier_ipcw(actual_days, event, 1 - outputs.p_approval_by_horizon, 700),
    }

    print("Running SHAP (per class)...")
    long_test = S.expand_person_periods(test_case)
    _explain_survival(pipeline, long_test)

    print("Computing reference layer (lifelines: KM / Aalen-Johansen / stratified Cox)...")
    reference = _reference_lifelines(case_df)
    model_km_check = _discrete_model_km_check(pipeline, case_df)

    joblib.dump(pipeline, S.SURVIVAL_MODELS_DIR / "survival_model.joblib")
    pipeline.named_steps["model"].get_booster().save_model(
        str(S.SURVIVAL_MODELS_DIR / "survival_model.ubj")
    )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train_cases": len(train_case),
        "n_train_censored": int((train_case["dataset"] == "under_review").sum()),
        "n_test_cases": len(test_case),
        "bin_width_days": S.BIN_WIDTH_DAYS,
        "max_bin": S.MAX_BIN,
        "cv": cv,
        "test": test_metrics,
        "reference_lifelines": reference,
        "discrete_model_mean_survival_by_horizon": model_km_check,
    }
    S.SURVIVAL_METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str))

    print("\nMetrics")
    print("-" * 60)
    print(json.dumps(metrics, indent=2, default=str))
    print(f"\nWrote {S.SURVIVAL_METRICS_FILE.relative_to(C.PROJECT_ROOT)}")
    print(f"Wrote models under {S.SURVIVAL_MODELS_DIR.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
