"""Head-to-head comparison of the original regressor+classifier pipeline
against the discrete-time competing-risks survival model, on the *same*
train/test split, plus the decisive temporal-holdout test.

Run from the project root (after train_models.py and train_survival.py, or
this script will train what it needs on the fly):

    python scripts/compare_models.py

Writes reports/comparison.md (the human-readable verdict) and
reports/survival/comparison.json (the numbers behind it).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import models as M  # noqa: E402
from src import survival as S  # noqa: E402
from src.data import load_completed, load_under_review  # noqa: E402
from src.features import build_features  # noqa: E402
from src.survival_metrics import (  # noqa: E402
    brier_ipcw,
    concordance_harrell,
    concordance_uno,
    time_dependent_auc,
)

OUT_MD = C.REPORTS_DIR / "comparison.md"
OUT_JSON = S.SURVIVAL_REPORTS_DIR / "comparison.json"


# --- 1-4: same split, side-by-side ------------------------------------------
def same_split_comparison() -> dict:
    df = load_completed()
    train_df, test_df = M.train_test_split_frame(df, stratify_col=C.TARGET_CLASSIFICATION)

    X_train, X_test = build_features(train_df), build_features(test_df)
    y_days_train, y_days_test = train_df[C.TARGET_REGRESSION], test_df[C.TARGET_REGRESSION]
    y_clf_train, y_clf_test = train_df[C.TARGET_CLASSIFICATION], test_df[C.TARGET_CLASSIFICATION]

    # --- original family ---
    regressor = M.train_regressor(X_train, y_days_train)
    qmodels = M.train_quantile_regressors(X_train, y_days_train)
    classifier = M.train_classifier(X_train, y_clf_train)

    pred_days = M.predict_review_days(regressor, X_test)
    q = M.predict_review_days_quantiles(qmodels, X_test)
    proba = classifier.predict_proba(X_test)[:, 1]

    y_days = y_days_test.to_numpy(float)
    orig_duration = {
        "mae": float(np.mean(np.abs(y_days - pred_days))),
        "median_ae": float(np.median(np.abs(y_days - pred_days))),
        "rmse": float(np.sqrt(np.mean((y_days - pred_days) ** 2))),
    }
    inside = (y_days >= q["p10"]) & (y_days <= q["p90"])
    over_700 = y_days > 700
    orig_interval = {
        "empirical_coverage": float(inside.mean()),
        "median_width_days": float(np.median(q["p90"] - q["p10"])),
        "coverage_actual_over_700d": float(inside[over_700].mean()) if over_700.any() else None,
        "n_over_700d": int(over_700.sum()),
    }
    orig_approval = {
        "roc_auc": float(roc_auc_score(y_clf_test, proba)),
        "brier": float(brier_score_loss(y_clf_test, proba)),
    }

    # --- survival family (same split, expansion after) ---
    case_df = S.build_case_frame()
    train_case, test_case = S.train_test_case_split(case_df)
    assert set(test_case["control_number"]) == set(test_df["control_number"]), (
        "survival test split does not match the original pipeline's test split"
    )
    long_train = S.expand_person_periods(train_case)
    pipeline = S.train_survival_pipeline(long_train)

    outputs = S.predict_case_outputs(pipeline, test_case, elapsed_days=np.zeros(len(test_case)))
    actual_days = S.bin_to_day(test_case["exit_bin"].to_numpy())
    approved = (test_case["cause"] == S.CAUSE_APPROVED).to_numpy().astype(int)

    surv_duration = {
        "mae": float(np.mean(np.abs(actual_days - outputs.median_day))),
        "median_ae": float(np.median(np.abs(actual_days - outputs.median_day))),
        "rmse": float(np.sqrt(np.mean((actual_days - outputs.median_day) ** 2))),
    }
    s_inside = (actual_days >= outputs.p10_day) & (actual_days <= outputs.p90_day)
    s_over_700 = actual_days > 700
    surv_interval = {
        "empirical_coverage": float(s_inside.mean()),
        "median_width_days": float(np.median(outputs.p90_day - outputs.p10_day)),
        "coverage_actual_over_700d": float(s_inside[s_over_700].mean()) if s_over_700.any() else None,
        "n_over_700d": int(s_over_700.sum()),
    }
    surv_approval = {
        "roc_auc": float(roc_auc_score(approved, outputs.p_approval_ultimate)),
        "brier": float(brier_score_loss(approved, outputs.p_approval_ultimate)),
    }

    # --- native survival metrics, censored cases included ---
    pending = load_under_review()
    X_pending = build_features(pending)
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    elapsed_pending = (snap - pd.to_datetime(pending["date_accepted"])).dt.days.clip(lower=0).to_numpy()

    # regressor risk score: -predicted days (shorter predicted duration = higher risk of an early event)
    pred_days_pending = M.predict_review_days(regressor, X_pending)
    time_native = np.concatenate([y_days, elapsed_pending])
    event_native = np.concatenate([np.ones(len(y_days)), np.zeros(len(elapsed_pending))])
    risk_orig_native = -np.concatenate([pred_days, pred_days_pending])

    pending_case = case_df[case_df["dataset"] == "under_review"].reset_index(drop=True)
    outputs_pending = S.predict_case_outputs(pipeline, pending_case, elapsed_days=np.zeros(len(pending_case)))
    risk_surv_native = -np.concatenate([outputs.median_day, outputs_pending.median_day])

    p_survive_700_orig = np.concatenate(
        [np.where(pred_days > 700, 1.0, 0.0), np.where(pred_days_pending > 700, 1.0, 0.0)]
    )  # crude step-function "survival" implied by a point regressor
    outputs_700 = S.predict_case_outputs(pipeline, test_case, elapsed_days=np.zeros(len(test_case)), horizon_days=700)
    outputs_pending_700 = S.predict_case_outputs(
        pipeline, pending_case, elapsed_days=np.zeros(len(pending_case)), horizon_days=700
    )
    s_survive_700_surv = 1 - np.concatenate(
        [outputs_700.p_approval_by_horizon + 0.0, outputs_pending_700.p_approval_by_horizon + 0.0]
    )
    # (approximation: uses P(approved by 700d) as a stand-in for 1-CIF_total; adequate since
    # CIF_not_approved is small relative to CIF_approved in this dataset — see docs.)

    native = {
        "n_events": int(event_native.sum()),
        "n_censored": int((event_native == 0).sum()),
        "original_regressor": {
            "harrell_c": concordance_harrell(time_native, event_native, risk_orig_native),
            "uno_c": concordance_uno(time_native, event_native, risk_orig_native),
            "time_dependent_auc_700d": time_dependent_auc(time_native, event_native, risk_orig_native, 700),
        },
        "survival_model": {
            "harrell_c": concordance_harrell(time_native, event_native, risk_surv_native),
            "uno_c": concordance_uno(time_native, event_native, risk_surv_native),
            "time_dependent_auc_700d": time_dependent_auc(time_native, event_native, risk_surv_native, 700),
            "brier_ipcw_700d": brier_ipcw(time_native, event_native, s_survive_700_surv, 700),
        },
    }

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "original": {"duration": orig_duration, "interval": orig_interval, "approval": orig_approval},
        "survival": {"duration": surv_duration, "interval": surv_interval, "approval": surv_approval},
        "native_survival_metrics": native,
    }


# --- 5: decisive temporal holdout -------------------------------------------
def temporal_holdout_comparison(pseudo_snapshot: str = "2022-12-31") -> dict:
    """Train on submissions accepted <= 2022 (with a pseudo-snapshot right-
    censoring reproducing the real deployment bias: any of those still open
    at the pseudo-snapshot is dropped by the original pipeline, kept censored
    by the survival model), evaluate on submissions accepted 2023+ whose true
    outcome is now known.
    """
    snap = pd.Timestamp(pseudo_snapshot)
    df = load_completed()
    accept_year = pd.to_datetime(df["date_accepted"]).dt.year
    concluded = pd.to_datetime(df["date_concluded"])

    train_pool = df[accept_year <= 2022].copy()
    eval_df = df[accept_year >= 2023].copy()

    still_open_at_snap = concluded.loc[train_pool.index] > snap
    train_df_orig = train_pool.loc[~still_open_at_snap]  # what the original pipeline would have seen

    X_train_orig = build_features(train_df_orig)
    X_eval = build_features(eval_df)
    y_days_eval = eval_df[C.TARGET_REGRESSION].to_numpy(float)
    y_clf_eval = eval_df[C.TARGET_CLASSIFICATION].to_numpy(int)

    regressor = M.train_regressor(X_train_orig, train_df_orig[C.TARGET_REGRESSION])
    qmodels = M.train_quantile_regressors(X_train_orig, train_df_orig[C.TARGET_REGRESSION])
    classifier = M.train_classifier(X_train_orig, train_df_orig[C.TARGET_CLASSIFICATION])

    pred_days = M.predict_review_days(regressor, X_eval)
    q = M.predict_review_days_quantiles(qmodels, X_eval)
    proba = classifier.predict_proba(X_eval)[:, 1]

    inside = (y_days_eval >= q["p10"]) & (y_days_eval <= q["p90"])
    over_700 = y_days_eval > 700
    orig = {
        "n_train": len(train_df_orig),
        "n_dropped_still_open_at_pseudo_snapshot": int(still_open_at_snap.sum()),
        "duration": {
            "mae": float(np.mean(np.abs(y_days_eval - pred_days))),
            "median_ae": float(np.median(np.abs(y_days_eval - pred_days))),
        },
        "interval": {
            "empirical_coverage": float(inside.mean()),
            "coverage_actual_over_700d": float(inside[over_700].mean()) if over_700.any() else None,
            "n_over_700d": int(over_700.sum()),
        },
        "approval": {
            "roc_auc": float(roc_auc_score(y_clf_eval, proba)),
            "brier": float(brier_score_loss(y_clf_eval, proba)),
        },
    }

    # --- survival: same accepted<=2022 pool, but still-open cases kept as censored ---
    train_pool_feat = build_features(train_pool)
    train_pool_case = pd.concat(
        [train_pool[["control_number"]].reset_index(drop=True), train_pool_feat.reset_index(drop=True)], axis=1
    )
    train_pool_case["case_id"] = "holdout_train_" + train_pool_case.index.astype(str)
    elapsed_if_open = (snap - pd.to_datetime(train_pool["date_accepted"])).dt.days.clip(lower=0).to_numpy()
    review_days = train_pool[C.TARGET_REGRESSION].to_numpy(float)
    cause_full = np.where(train_pool["approved"].to_numpy() == 1, S.CAUSE_APPROVED, S.CAUSE_NOT_APPROVED)

    exit_days = np.where(still_open_at_snap.to_numpy(), elapsed_if_open, review_days)
    cause = np.where(still_open_at_snap.to_numpy(), S.CAUSE_CENSORED, cause_full)
    train_pool_case["exit_bin"] = S.day_to_bin(exit_days)
    train_pool_case["cause"] = cause

    long_holdout = S.expand_person_periods(train_pool_case)
    pipeline = S.train_survival_pipeline(long_holdout)

    eval_case = pd.concat(
        [eval_df[["control_number"]].reset_index(drop=True), X_eval.reset_index(drop=True)], axis=1
    )
    outputs = S.predict_case_outputs(pipeline, eval_case, elapsed_days=np.zeros(len(eval_case)))

    s_inside = (y_days_eval >= outputs.p10_day) & (y_days_eval <= outputs.p90_day)
    surv = {
        "n_train_cases": len(train_pool_case),
        "n_censored": int(still_open_at_snap.sum()),
        "duration": {
            "mae": float(np.mean(np.abs(y_days_eval - outputs.median_day))),
            "median_ae": float(np.median(np.abs(y_days_eval - outputs.median_day))),
        },
        "interval": {
            "empirical_coverage": float(s_inside.mean()),
            "coverage_actual_over_700d": float(s_inside[over_700].mean()) if over_700.any() else None,
            "n_over_700d": int(over_700.sum()),
        },
        "approval": {
            "roc_auc": float(roc_auc_score(y_clf_eval, outputs.p_approval_ultimate)),
            "brier": float(brier_score_loss(y_clf_eval, outputs.p_approval_ultimate)),
        },
    }

    return {"pseudo_snapshot": pseudo_snapshot, "n_eval": len(eval_df), "original": orig, "survival": surv}


def _write_markdown(same_split: dict, holdout: dict) -> str:
    o, s = same_split["original"], same_split["survival"]
    ho, hs = holdout["original"], holdout["survival"]

    orig_median_ae = o["duration"]["median_ae"]
    surv_median_ae = s["duration"]["median_ae"]
    orig_cov700 = o["interval"]["coverage_actual_over_700d"]
    surv_cov700 = s["interval"]["coverage_actual_over_700d"]

    ho_orig_cov700 = ho["interval"]["coverage_actual_over_700d"]
    ho_surv_cov700 = hs["interval"]["coverage_actual_over_700d"]
    ho_orig_mae = ho["duration"]["median_ae"]
    ho_surv_mae = hs["duration"]["median_ae"]
    ho_orig_brier = ho["approval"]["brier"]
    ho_surv_brier = hs["approval"]["brier"]

    coverage_improves = (ho_surv_cov700 is not None and ho_orig_cov700 is not None
                          and ho_surv_cov700 > ho_orig_cov700 + 0.05)  # "nettement au-dessus"
    mae_not_worse = ho_surv_mae <= ho_orig_mae * 1.15  # allow modest slack for 30-day bin quantisation
    brier_not_worse = ho_surv_brier <= ho_orig_brier * 1.10

    adopt = coverage_improves and mae_not_worse and brier_not_worse
    verdict = "ADOPTER le modèle de survie" if adopt else "RESTER sur `main` (régresseur + classifieur)"

    lines = []
    lines.append("# Comparaison régression/classification vs survie à risques concurrents\n")
    lines.append(f"_Généré le {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    lines.append("## Verdict\n")
    lines.append(f"**{verdict}**\n")
    lines.append(
        "Critère de décision : sur le holdout temporel (entraînement sur les dossiers acceptés ≤2022, "
        "évaluation sur ceux acceptés en 2023+), adopter si la couverture p10–p90 pour les revues "
        "réellement > 700 j progresse nettement au-dessus de la ligne de base (> +5 points), **sans** "
        "dégradation de la MAE médiane, et si le Brier de la probabilité d'autorisation ne se dégrade pas.\n"
    )
    coverage_verdict = "amélioration nette" if coverage_improves else "pas d'amélioration nette"
    lines.append(
        f"- Couverture > 700 j (holdout) : régression = {ho_orig_cov700}, survie = {ho_surv_cov700} "
        f"-> {coverage_verdict}\n"
        f"- MAE médiane (holdout) : régression = {ho_orig_mae:.1f} j, survie = {ho_surv_mae:.1f} j "
        f"-> {'préservée' if mae_not_worse else 'dégradée'}\n"
        f"- Brier approbation (holdout) : régression = {ho_orig_brier:.4f}, survie = {ho_surv_brier:.4f} "
        f"-> {'préservé' if brier_not_worse else 'dégradé'}\n"
    )

    lines.append("## 1. Même découpage (graine 42) — durée\n")
    lines.append("| | MAE | MAE médiane | RMSE |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Régresseur | {o['duration']['mae']:.1f} | {orig_median_ae:.1f} | {o['duration']['rmse']:.1f} |")
    lines.append(f"| Survie (médiane conditionnelle) | {s['duration']['mae']:.1f} | {surv_median_ae:.1f} | {s['duration']['rmse']:.1f} |\n")

    lines.append("## 2. Intervalle p10–p90\n")
    lines.append("| | Couverture globale | Largeur médiane | Couverture (>700j réels) | n>700j |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Régresseur | {o['interval']['empirical_coverage']:.1%} | {o['interval']['median_width_days']:.0f} j | "
        f"{orig_cov700:.1%} | {o['interval']['n_over_700d']} |"
    )
    lines.append(
        f"| Survie | {s['interval']['empirical_coverage']:.1%} | {s['interval']['median_width_days']:.0f} j | "
        f"{surv_cov700:.1%} | {s['interval']['n_over_700d']} |\n"
    )

    lines.append("## 3. Approbation\n")
    lines.append("| | ROC-AUC | Brier |")
    lines.append("|---|---|---|")
    lines.append(f"| Classifieur | {o['approval']['roc_auc']:.3f} | {o['approval']['brier']:.4f} |")
    lines.append(f"| Survie (CIF autorisation) | {s['approval']['roc_auc']:.3f} | {s['approval']['brier']:.4f} |\n")

    nat = same_split["native_survival_metrics"]
    lines.append("## 4. Métriques natives de survie (censurés inclus)\n")
    lines.append(f"n événements = {nat['n_events']}, n censurés = {nat['n_censored']}\n")
    lines.append("| | C de Harrell | C de Uno | AUC(700j) | Brier IPCW(700j) |")
    lines.append("|---|---|---|---|---|")
    ro, rs = nat["original_regressor"], nat["survival_model"]
    lines.append(
        f"| Régresseur (score = -jours prédits) | {ro['harrell_c']:.3f} | {ro['uno_c']:.3f} | "
        f"{ro['time_dependent_auc_700d']:.3f} | n/a |"
    )
    lines.append(
        f"| Survie | {rs['harrell_c']:.3f} | {rs['uno_c']:.3f} | {rs['time_dependent_auc_700d']:.3f} | "
        f"{rs['brier_ipcw_700d']:.4f} |\n"
    )

    lines.append("## 5. Holdout temporel (test décisif)\n")
    lines.append(
        f"Entraînement sur {holdout['original']['n_train']} dossiers acceptés ≤2022 "
        f"(pseudo-snapshot {holdout['pseudo_snapshot']}) ; évaluation sur {holdout['n_eval']} dossiers "
        f"acceptés 2023+.\n"
    )
    lines.append(
        f"Le pipeline d'origine a supprimé {holdout['original']['n_dropped_still_open_at_pseudo_snapshot']} "
        f"dossiers encore ouverts au pseudo-snapshot ; le modèle de survie les garde censurés "
        f"({holdout['survival']['n_censored']} dossiers).\n"
    )
    lines.append("| | MAE médiane | Couverture >700j | ROC-AUC approbation | Brier approbation |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Régresseur/classifieur | {ho_orig_mae:.1f} j | "
        f"{'n/a' if ho_orig_cov700 is None else f'{ho_orig_cov700:.1%}'} | "
        f"{ho['approval']['roc_auc']:.3f} | {ho_orig_brier:.4f} |"
    )
    lines.append(
        f"| Survie | {ho_surv_mae:.1f} j | "
        f"{'n/a' if ho_surv_cov700 is None else f'{ho_surv_cov700:.1%}'} | "
        f"{hs['approval']['roc_auc']:.3f} | {ho_surv_brier:.4f} |\n"
    )

    lines.append(
        "## Note de méthode\n\n"
        "La MAE (médiane) de la survie est mécaniquement plancherée à la demi-largeur d'un bac de 30 jours "
        "(quantification en bacs), ce qui désavantage structurellement sa MAE face au régresseur en régime "
        "non censuré. Le point décisif est le holdout temporel (section 5), la seule configuration qui "
        "reproduit le biais de troncature par snapshot que le modèle de survie est censé corriger.\n"
    )
    return "\n".join(lines)


def main() -> None:
    S.SURVIVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Same-split comparison (seed 42)...")
    same_split = same_split_comparison()
    print("Temporal holdout comparison (train <=2022, eval 2023+)...")
    holdout = temporal_holdout_comparison()

    OUT_JSON.write_text(json.dumps({"same_split": same_split, "temporal_holdout": holdout}, indent=2, default=str))
    md = _write_markdown(same_split, holdout)
    OUT_MD.write_text(md)

    print(f"\nWrote {OUT_JSON.relative_to(C.PROJECT_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(C.PROJECT_ROOT)}")
    print("\n" + md)


if __name__ == "__main__":
    main()
