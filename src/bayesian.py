"""Bayesian-style update of the conclusion date for submissions that have
already outlived the model's initial estimate.

Mirrors the approach sketched in the Aily Lab GRA deck (slide 44): once a
submission is still under review *past* its predicted conclusion date, the initial
point prediction is stale — the review is, by observation, one of the slower
ones. We treat the regressor's predictive distribution over ``review_days`` as a
prior, condition on the event "still under review at ``days_elapsed``" (a
left-truncation of that prior), and read the updated median / p10 / p90 off the
posterior.

The per-submission predictive distribution is approximated by a **lognormal**
matched to the model's predicted p10 / p50 / p90 — consistent with the regressor
being trained on ``log1p(review_days)``. Left-truncating a lognormal has a closed
form, so the update is cheap and, unlike the raw quantile grid, extrapolates
sensibly when the submission has already run past the model's p90.

    prior:      review_days ~ LogNormal(mu, sigma)          (mu = ln p50)
    evidence:   review_days > c                             (c = days elapsed)
    posterior:  review_days | review_days > c   (left-truncated lognormal)

``update_conclusion`` is the entry point used by
``scripts/predict_under_review.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C

# A review is not allowed to be re-projected beyond this horizon. Matches the
# "excessive lead time" bound used elsewhere in the regulatory literature
# (~7 years); past it the metadata carries no signal and a number would be noise.
MAX_REVIEW_DAYS = 7 * 365

# Numerical guard: clip truncation / quantile probabilities away from {0, 1} so
# ``norm.ppf`` stays finite for submissions that have run far past p90.
_EPS = 1e-6
_Z10, _Z90 = norm.ppf(0.10), norm.ppf(0.90)

_CALIBRATION_FILE = C.REPORTS_DIR / "reprojection_calibration.json"


def load_calibration() -> tuple[float, float]:
    """``(mu_shift_log, sigma_inflation)`` from the temporal-holdout calibration.

    Produced by ``scripts/calibrate_reprojection.py``. The prior lognormal is fit
    on completed submissions only and is optimistic on slow reviews, so for
    ``late`` / ``overdue`` rows we shift ``mu`` and widen ``sigma`` by these
    factors. Falls back to ``(0.0, 1.0)`` (no correction) when the file is absent.
    """
    try:
        d = json.loads(_CALIBRATION_FILE.read_text())
        return float(d["mu_shift_log"]), float(d["sigma_inflation"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.0, 1.0


def lognormal_params(
    p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lognormal ``(mu, sigma)`` matched to predicted review-time quantiles.

    ``mu`` is anchored on the model's median (``ln p50``); ``sigma`` comes from
    the p10–p90 spread in log space. Degenerate or non-increasing inputs fall
    back to a small positive ``sigma``.
    """
    p10 = np.maximum(np.asarray(p10, float), 1.0)
    p50 = np.maximum(np.asarray(p50, float), 1.0)
    p90 = np.maximum(np.asarray(p90, float), 1.0)
    mu = np.log(p50)
    sigma = (np.log(p90) - np.log(p10)) / (_Z90 - _Z10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-3), sigma, 1e-3)
    return mu, sigma


def conditional_quantiles(
    mu: np.ndarray,
    sigma: np.ndarray,
    elapsed: np.ndarray,
    alphas: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    """Quantiles of ``LogNormal(mu, sigma)`` conditioned on ``value > elapsed``.

    Returns one column per ``alpha`` (named ``p10``, ``p50``, ...), each the
    posterior quantile in days, floored at ``elapsed`` and capped at
    :data:`MAX_REVIEW_DAYS`.
    """
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    elapsed = np.maximum(np.asarray(elapsed, float), 0.0)

    # F(elapsed): prior mass already ruled out by "still under review".
    z_c = (np.log(np.maximum(elapsed, 1.0)) - mu) / sigma
    F_c = np.clip(norm.cdf(z_c), 0.0, 1.0 - _EPS)

    out = {}
    for a in alphas:
        # target prior-CDF level for posterior quantile a: F_c + a * (1 - F_c)
        target = np.clip(F_c + a * (1.0 - F_c), _EPS, 1.0 - _EPS)
        days = np.exp(mu + sigma * norm.ppf(target))
        days = np.clip(days, elapsed, MAX_REVIEW_DAYS)
        out[f"p{int(round(a * 100))}"] = days
    return pd.DataFrame(out)


def update_conclusion(
    quantile_days: pd.DataFrame,
    days_elapsed: np.ndarray,
    date_accepted: pd.Series,
    status: np.ndarray,
    alphas: tuple[float, ...] = (0.10, 0.50, 0.90),
    mu_shift: float | None = None,
    sigma_inflation: float | None = None,
) -> pd.DataFrame:
    """Re-project the conclusion date for ``late`` / ``overdue`` submissions.

    Parameters
    ----------
    quantile_days
        Initial (unfloored) review-time quantiles, columns ``p10``/``p50``/``p90``.
    days_elapsed
        Calendar days each submission has already been in review at the snapshot.
    date_accepted
        Acceptance date per submission (datetime-like).
    status
        Per-row ``"on_track"`` / ``"late"`` / ``"overdue"`` from the initial
        estimate. Only the latter two are updated; ``on_track`` rows pass through
        their initial (elapsed-floored) quantiles unchanged.
    mu_shift, sigma_inflation
        Temporal-holdout corrections applied to the prior of ``late`` / ``overdue``
        rows (see :func:`load_calibration`). Default to the calibrated values on
        disk, or ``(0.0, 1.0)`` if none.

    Returns a frame aligned to the input with:
      ``bayes_review_days_p10/p50/p90``, ``bayes_conclusion_date``,
      ``bayes_conclusion_earliest``, ``bayes_conclusion_latest``,
      ``bayes_remaining_days_p50``, ``bayes_update_applied``, ``bayes_note``.

    ``bayes_note`` is ``"initial"`` (on_track, unchanged), ``"reprojected"``, or
    ``"beyond_model"`` -- the posterior is exhausted (already past its own p90),
    so only "not before the snapshot" is asserted.
    """
    cal_shift, cal_infl = load_calibration()
    mu_shift = cal_shift if mu_shift is None else mu_shift
    sigma_inflation = cal_infl if sigma_inflation is None else sigma_inflation

    q = quantile_days.reset_index(drop=True)
    elapsed = np.maximum(np.asarray(days_elapsed, float), 0.0)
    status = np.asarray(status, dtype=object)
    accepted = pd.to_datetime(pd.Series(date_accepted).reset_index(drop=True))
    applied = np.isin(status, ("late", "overdue"))

    mu, sigma = lognormal_params(q["p10"], q["p50"], q["p90"])
    # corrections only bite on the re-projected rows
    mu_adj = np.where(applied, mu + mu_shift, mu)
    sigma_adj = np.where(applied, sigma * sigma_inflation, sigma)
    post = conditional_quantiles(mu_adj, sigma_adj, elapsed, alphas)

    # on_track rows keep their initial estimate, floored at elapsed.
    initial_floored = pd.DataFrame(
        np.maximum(q[["p10", "p50", "p90"]].to_numpy(), elapsed[:, None]),
        columns=["p10", "p50", "p90"],
    )
    days = post.where(pd.Series(applied), initial_floored)

    collapsed = applied & (days["p90"].to_numpy() - elapsed <= 1.0)
    note = np.where(collapsed, "beyond_model", np.where(applied, "reprojected", "initial"))

    def _date(col: np.ndarray) -> pd.Series:
        return (accepted + pd.to_timedelta(np.round(col), unit="D")).dt.date

    return pd.DataFrame(
        {
            "bayes_review_days_p10": days["p10"].round(),
            "bayes_review_days_p50": days["p50"].round(),
            "bayes_review_days_p90": days["p90"].round(),
            "bayes_conclusion_date": _date(days["p50"].to_numpy()),
            "bayes_conclusion_earliest": _date(days["p10"].to_numpy()),
            "bayes_conclusion_latest": _date(days["p90"].to_numpy()),
            "bayes_remaining_days_p50": np.maximum(
                days["p50"].to_numpy() - elapsed, 0.0
            ).round(),
            "bayes_update_applied": applied,
            "bayes_note": note,
        }
    )
