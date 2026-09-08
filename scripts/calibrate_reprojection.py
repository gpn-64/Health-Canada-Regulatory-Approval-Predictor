"""Calibrate the survival-conditioned conclusion-date re-projection.

The re-projection in ``src/bayesian.py`` conditions the regressor's predictive
lognormal on "still under review at ``elapsed``". That prior is fit on completed
submissions only, so it inherits the regressor's optimism on slow reviews (the
temporal holdout shows p10-p90 coverage collapsing to ~0 % past 700 days). This
script measures that bias on a *temporal* holdout and derives two corrections
applied to ``late`` / ``overdue`` rows:

  * ``mu_shift_log`` -- additive shift on ``mu`` (log-day units) so the posterior
    median is unbiased for submissions that were in-flight at the pseudo-snapshot;
  * ``sigma_inflation`` -- multiplicative widening of ``sigma`` so the posterior
    p10-p90 band reaches ~80 % empirical coverage on the same rows.

Method:
  pseudo-snapshot 2022-12-31; train quantile regressors on submissions concluded
  by then; the eval set is the submissions accepted by then but concluded later
  (i.e. censored at the pseudo-snapshot, true durations now known). Keep the
  ``late`` / ``overdue`` subset and fit the two constants there.

    python scripts/calibrate_reprojection.py [--pseudo-snapshot 2022-12-31]

Writes ``reports/reprojection_calibration.json`` (read back by ``src/bayesian.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import models as M  # noqa: E402
from src.bayesian import conditional_quantiles, lognormal_params  # noqa: E402
from src.features import build_features  # noqa: E402

OUT_FILE = C.REPORTS_DIR / "reprojection_calibration.json"


def _coverage(mu, sigma, elapsed, actual, k=1.0, shift=0.0) -> float:
    q = conditional_quantiles(mu + shift, sigma * k, elapsed, alphas=(0.10, 0.90))
    inside = (actual >= q["p10"].to_numpy()) & (actual <= q["p90"].to_numpy())
    return float(inside.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-snapshot", default="2022-12-31")
    args = ap.parse_args()
    ps = pd.Timestamp(args.pseudo_snapshot)

    clean = pd.read_csv(
        C.CLEAN_FILE, parse_dates=["date_accepted", "date_concluded"]
    )
    train = clean[clean["date_concluded"] <= ps]
    evalset = clean[
        (clean["date_accepted"] <= ps) & (clean["date_concluded"] > ps)
    ].reset_index(drop=True)
    if len(evalset) < 20:
        raise SystemExit(f"Only {len(evalset)} in-flight rows at {ps.date()} - too few.")

    qmodels = M.train_quantile_regressors(
        build_features(train), train[C.TARGET_REGRESSION]
    )
    q = M.predict_review_days_quantiles(qmodels, build_features(evalset))

    elapsed = (ps - evalset["date_accepted"]).dt.days.clip(lower=0).to_numpy(float)
    actual = evalset[C.TARGET_REGRESSION].to_numpy(float)
    late = elapsed > q["p50"].to_numpy()  # late or overdue at the pseudo-snapshot
    n_late = int(late.sum())
    if n_late < 10:
        raise SystemExit(f"Only {n_late} late/overdue rows - not enough to calibrate.")

    mu, sigma = lognormal_params(q["p10"], q["p50"], q["p90"])
    mu_l, sigma_l = mu[late], sigma[late]
    elapsed_l, actual_l = elapsed[late], actual[late]

    # The prior median is off by ~3x for these rows, but a flat additive shift on
    # mu over-corrects the barely-late dossiers (which do conclude soon). We let
    # the left-truncation carry the location correction -- it is position-aware --
    # and only widen sigma so the tail is heavy enough. mu_shift stays 0 (kept in
    # the schema for a future position-dependent term).
    mu_shift = 0.0

    # sigma inflation: smallest widening reaching >= 80 % coverage, else the best.
    cov_before = _coverage(mu_l, sigma_l, elapsed_l, actual_l)
    grid = np.round(np.arange(1.0, 4.01, 0.1), 2)
    covs = {
        float(k): _coverage(mu_l, sigma_l, elapsed_l, actual_l, k=k, shift=mu_shift)
        for k in grid
    }
    reaching = [k for k, c in covs.items() if c >= 0.80]
    sigma_k = float(min(reaching)) if reaching else float(max(covs, key=covs.get))
    cov_after = covs[sigma_k]
    reaches_nominal = bool(reaching)

    payload = {
        "pseudo_snapshot": str(ps.date()),
        "n_train": int(len(train)),
        "n_eval_in_flight": int(len(evalset)),
        "n_late_or_overdue": n_late,
        "mu_shift_log": round(mu_shift, 4),
        "sigma_inflation": sigma_k,
        "coverage_p10_p90_before": round(cov_before, 3),
        "coverage_p10_p90_after": round(cov_after, 3),
        "reaches_nominal_coverage": reaches_nominal,
        "median_days_correction": round(
            float(np.median(np.expm1(mu_l + mu_shift) - np.expm1(mu_l))), 1
        ),
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_FILE.relative_to(C.PROJECT_ROOT)}")
    print(
        "\nRead automatically by src/bayesian.py; re-run after refreshing the "
        "snapshot or retraining."
    )


if __name__ == "__main__":
    main()
