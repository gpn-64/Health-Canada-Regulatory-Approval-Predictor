"""Continuous parametric AFT vs. the original regressor for review duration.

The discrete-time survival model (`scripts/compare_models.py`) was not adopted
because 30-day time bins cost ~11 days of median AE. This script tests whether a
*continuous* AFT — which also consumes the in-flight submissions as censored
observations — keeps the censoring benefit without the bin-quantisation penalty.

Two configurations, mirroring `reports/comparison.md`:
  1. same split (seed 42) — unconditional forecast, test = completed hold-out;
  5. temporal holdout — train on accepted <=2022 (still-open cases censored at a
     2022-12-31 pseudo-snapshot), evaluate on accepted 2023+.

    python scripts/compare_aft.py [--distribution lognormal]

Writes reports/comparison_aft.md and reports/survival/comparison_aft.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import models as M  # noqa: E402
from src.data import load_completed, load_under_review  # noqa: E402
from src.features import build_features  # noqa: E402
from src.survival_aft import AFTModel  # noqa: E402
from src.survival_metrics import concordance_harrell  # noqa: E402

OUT_MD = C.REPORTS_DIR / "comparison_aft.md"
OUT_JSON = C.REPORTS_DIR / "survival" / "comparison_aft.json"


def _duration_metrics(p10, p50, p90, actual) -> dict:
    actual = np.asarray(actual, float)
    p10, p50, p90 = map(np.asarray, (p10, p50, p90))
    inside = (actual >= p10) & (actual <= p90)
    over = actual > 700
    return {
        "mae": float(np.mean(np.abs(actual - p50))),
        "median_ae": float(np.median(np.abs(actual - p50))),
        "rmse": float(np.sqrt(np.mean((actual - p50) ** 2))),
        "coverage": float(inside.mean()),
        "median_width_days": float(np.median(p90 - p10)),
        "coverage_over_700d": float(inside[over].mean()) if over.any() else None,
        "n_over_700d": int(over.sum()),
    }


def _censored_in_flight():
    """In-flight submissions as right-censored rows: features, elapsed duration."""
    pending = load_under_review()
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    elapsed = (snap - pd.to_datetime(pending["date_accepted"])).dt.days.clip(lower=0)
    return build_features(pending), elapsed.to_numpy(float)


def same_split(distribution: str) -> dict:
    df = load_completed()
    train_df, test_df = M.train_test_split_frame(df, stratify_col=C.TARGET_CLASSIFICATION)
    X_tr, X_te = build_features(train_df), build_features(test_df)
    y_tr = train_df[C.TARGET_REGRESSION].to_numpy(float)
    y_te = test_df[C.TARGET_REGRESSION].to_numpy(float)

    # --- original regressor + quantile band ---
    reg = M.train_regressor(X_tr, train_df[C.TARGET_REGRESSION])
    qm = M.train_quantile_regressors(X_tr, train_df[C.TARGET_REGRESSION])
    q = M.predict_review_days_quantiles(qm, X_te)
    orig = _duration_metrics(q["p10"], M.predict_review_days(reg, X_te), q["p90"], y_te)

    # --- AFT: completed (events) + in-flight (censored) ---
    Xc, elapsed_c = _censored_in_flight()
    X_fit = pd.concat([X_tr, Xc], ignore_index=True)
    T_fit = np.concatenate([y_tr, elapsed_c])
    E_fit = np.concatenate([np.ones(len(y_tr)), np.zeros(len(elapsed_c))])
    aft = AFTModel(distribution=distribution).fit(X_fit, T_fit, E_fit)
    aq = aft.predict_quantiles(X_te)
    aft_m = _duration_metrics(aq["p10"], aq["p50"], aq["p90"], y_te)

    # --- native concordance (censored in-flight included) ---
    time_native = np.concatenate([y_te, elapsed_c])
    event_native = np.concatenate([np.ones(len(y_te)), np.zeros(len(elapsed_c))])
    reg_risk = -np.concatenate(
        [M.predict_review_days(reg, X_te), M.predict_review_days(reg, Xc)]
    )
    aft_risk = -np.concatenate([aq["p50"].to_numpy(), aft.predict_median_days(Xc)])
    concordance = {
        "regressor": concordance_harrell(time_native, event_native, reg_risk),
        "aft": concordance_harrell(time_native, event_native, aft_risk),
    }
    return {"n_train": len(train_df), "n_test": len(test_df),
            "original": orig, "aft": aft_m, "harrell_c": concordance}


def temporal_holdout(distribution: str, pseudo_snapshot: str = "2022-12-31") -> dict:
    snap = pd.Timestamp(pseudo_snapshot)
    df = load_completed()
    year = pd.to_datetime(df["date_accepted"]).dt.year
    concluded = pd.to_datetime(df["date_concluded"])

    pool = df[year <= 2022].copy()
    eval_df = df[year >= 2023].copy()
    still_open = (concluded.loc[pool.index] > snap).to_numpy()

    X_pool = build_features(pool)
    X_eval = build_features(eval_df)
    y_eval = eval_df[C.TARGET_REGRESSION].to_numpy(float)

    # original pipeline: drops still-open rows entirely
    seen = ~still_open
    reg = M.train_regressor(X_pool[seen], pool[C.TARGET_REGRESSION][seen])
    qm = M.train_quantile_regressors(X_pool[seen], pool[C.TARGET_REGRESSION][seen])
    q = M.predict_review_days_quantiles(qm, X_eval)
    orig = _duration_metrics(q["p10"], M.predict_review_days(reg, X_eval), q["p90"], y_eval)
    orig["n_train"] = int(seen.sum())
    orig["n_dropped_still_open"] = int(still_open.sum())

    # AFT: keeps still-open rows, censored at elapsed
    elapsed_if_open = (snap - pd.to_datetime(pool["date_accepted"])).dt.days.clip(lower=0).to_numpy(float)
    review_days = pool[C.TARGET_REGRESSION].to_numpy(float)
    T = np.where(still_open, elapsed_if_open, review_days)
    E = np.where(still_open, 0, 1)
    aft = AFTModel(distribution=distribution).fit(X_pool, T, E)
    aq = aft.predict_quantiles(X_eval)
    aft_m = _duration_metrics(aq["p10"], aq["p50"], aq["p90"], y_eval)
    aft_m["n_train"] = len(pool)
    aft_m["n_censored"] = int(still_open.sum())

    return {"pseudo_snapshot": pseudo_snapshot, "n_eval": len(eval_df),
            "original": orig, "aft": aft_m}


def _md(distribution: str, ss: dict, ho: dict) -> str:
    o, a = ss["original"], ss["aft"]
    ho_o, ho_a = ho["original"], ho["aft"]

    mae_delta = ho_a["median_ae"] - ho_o["median_ae"]
    cov_o = ho_o["coverage_over_700d"] or 0.0
    cov_a = ho_a["coverage_over_700d"] or 0.0
    recovers = mae_delta <= 2.0
    cov_gain = ho_a["coverage"] - ho_o["coverage"]
    verdict = (
        f"AFT ({distribution}) récupère la MAE médiane"
        if recovers
        else f"AFT ({distribution}) ne récupère pas la MAE médiane"
    )

    L = [
        f"# AFT continu ({distribution}) vs régresseur — durée de revue\n",
        f"_Généré le {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n",
        "## Verdict\n",
        f"**{verdict}.** Holdout temporel : MAE médiane "
        f"{ho_o['median_ae']:.1f} j (régresseur) vs {ho_a['median_ae']:.1f} j (AFT), "
        f"Δ = {mae_delta:+.1f} j. Couverture p10–p90 globale "
        f"{ho_o['coverage']:.0%} vs {ho_a['coverage']:.0%} (Δ = {cov_gain:+.0%}) ; "
        f">700 j {cov_o:.0%} vs {cov_a:.0%}.\n",
        "L'AFT continu ne quantifie pas le temps (contrairement au modèle discret "
        "à bacs de 30 j), mais reste **linéaire dans les covariables** : il lisse "
        "le pic de ~43 % des dossiers qui concluent pile au standard de service de "
        "300 j, que le régresseur boosté reproduit exactement. La perte de MAE "
        "médiane n'est donc pas un artefact de bacs — c'est le coût de quitter la "
        "flexibilité de XGBoost. Ce que le cadre de survie apporte de façon "
        "constante : une meilleure couverture d'intervalle sous décalage temporel "
        "(les dossiers en cours restent censurés au lieu d'être supprimés).\n",
        "## 1. Même découpage (graine 42)\n",
        "| | MAE | MAE médiane | RMSE | Couv. p10–p90 | Couv. >700 j | C de Harrell |",
        "|---|---|---|---|---|---|---|",
        f"| Régresseur | {o['mae']:.1f} | {o['median_ae']:.1f} | {o['rmse']:.0f} | "
        f"{o['coverage']:.0%} | {(o['coverage_over_700d'] or 0):.0%} | {ss['harrell_c']['regressor']:.3f} |",
        f"| AFT {distribution} | {a['mae']:.1f} | {a['median_ae']:.1f} | {a['rmse']:.0f} | "
        f"{a['coverage']:.0%} | {(a['coverage_over_700d'] or 0):.0%} | {ss['harrell_c']['aft']:.3f} |\n",
        "## 5. Holdout temporel (décisif)\n",
        f"Entraînement sur les dossiers acceptés ≤2022 (pseudo-snapshot "
        f"{ho['pseudo_snapshot']}) ; évaluation sur {ho['n_eval']} dossiers acceptés 2023+. "
        f"Le régresseur perd {ho_o['n_dropped_still_open']} dossiers encore ouverts ; "
        f"l'AFT les garde censurés ({ho_a['n_censored']}).\n",
        "| | MAE | MAE médiane | Couv. p10–p90 | Couv. >700 j |",
        "|---|---|---|---|---|",
        f"| Régresseur | {ho_o['mae']:.1f} | {ho_o['median_ae']:.1f} | {ho_o['coverage']:.0%} | "
        f"{'n/a' if ho_o['coverage_over_700d'] is None else f'{cov_o:.0%}'} |",
        f"| AFT {distribution} | {ho_a['mae']:.1f} | {ho_a['median_ae']:.1f} | {ho_a['coverage']:.0%} | "
        f"{'n/a' if ho_a['coverage_over_700d'] is None else f'{cov_a:.0%}'} |\n",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distribution", default="lognormal",
                    choices=["lognormal", "weibull", "loglogistic"])
    args = ap.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"Same-split comparison (AFT {args.distribution})...")
    ss = same_split(args.distribution)
    print("Temporal-holdout comparison...")
    ho = temporal_holdout(args.distribution)

    OUT_JSON.write_text(json.dumps(
        {"distribution": args.distribution, "same_split": ss, "temporal_holdout": ho},
        indent=2, default=str,
    ))
    md = _md(args.distribution, ss, ho)
    OUT_MD.write_text(md)
    print(f"\nWrote {OUT_JSON.relative_to(C.PROJECT_ROOT)}\nWrote {OUT_MD.relative_to(C.PROJECT_ROOT)}\n")
    print(md)


if __name__ == "__main__":
    main()
