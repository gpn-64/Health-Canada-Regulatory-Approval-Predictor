"""Checks for the Bayesian (survival-conditioned) conclusion-date update."""

import numpy as np
import pandas as pd

from src.bayesian import (
    MAX_REVIEW_DAYS,
    conditional_quantiles,
    lognormal_params,
    update_conclusion,
)


def test_lognormal_params_recover_quantiles():
    mu, sigma = lognormal_params([100.0], [200.0], [400.0])
    # median anchored on p50, spread from p10/p90
    assert np.isclose(np.exp(mu[0]), 200.0)
    assert sigma[0] > 0


def test_lognormal_params_handle_degenerate_input():
    mu, sigma = lognormal_params([200.0], [200.0], [200.0])
    assert np.isfinite(mu).all() and (sigma > 0).all()


def test_conditional_quantiles_are_ordered_and_floored():
    mu, sigma = lognormal_params([100.0], [200.0], [400.0])
    elapsed = np.array([300.0])  # already past p90
    q = conditional_quantiles(mu, sigma, elapsed)
    assert q["p10"][0] >= elapsed[0]
    assert q["p10"][0] <= q["p50"][0] <= q["p90"][0]
    assert q["p90"][0] <= MAX_REVIEW_DAYS


def test_more_elapsed_pushes_the_estimate_later():
    mu, sigma = lognormal_params([100.0], [200.0], [400.0])
    early = conditional_quantiles(mu, sigma, np.array([50.0]))["p50"][0]
    late = conditional_quantiles(mu, sigma, np.array([300.0]))["p50"][0]
    assert late > early


def test_update_conclusion_only_touches_late_and_overdue():
    q = pd.DataFrame({"p10": [80, 80, 80], "p50": [150, 150, 150], "p90": [300, 300, 300]})
    elapsed = np.array([40, 200, 500])
    accepted = pd.to_datetime(pd.Series(["2025-01-01", "2025-01-01", "2025-01-01"]))
    status = np.array(["on_track", "late", "overdue"])
    res = update_conclusion(q, elapsed, accepted, status, mu_shift=0.0, sigma_inflation=1.0)

    assert list(res["bayes_update_applied"]) == [False, True, True]
    assert list(res["bayes_note"]) == ["initial", "reprojected", "reprojected"]
    # on_track row unchanged (initial p50 estimate, elapsed < p50)
    assert res["bayes_review_days_p50"][0] == 150
    # overdue row re-projected strictly beyond both elapsed and the old p90
    assert res["bayes_review_days_p50"][2] >= 500
    assert res["bayes_review_days_p90"][2] > 300
    assert (res["bayes_remaining_days_p50"] >= 0).all()


def test_sigma_inflation_widens_the_reprojected_band():
    q = pd.DataFrame({"p10": [80], "p50": [150], "p90": [300]})
    elapsed = np.array([200])
    accepted = pd.to_datetime(pd.Series(["2025-01-01"]))
    status = np.array(["late"])
    narrow = update_conclusion(q, elapsed, accepted, status, mu_shift=0.0, sigma_inflation=1.0)
    wide = update_conclusion(q, elapsed, accepted, status, mu_shift=0.0, sigma_inflation=2.3)
    w_narrow = narrow["bayes_review_days_p90"][0] - narrow["bayes_review_days_p10"][0]
    w_wide = wide["bayes_review_days_p90"][0] - wide["bayes_review_days_p10"][0]
    assert w_wide > w_narrow


def test_beyond_model_flag_when_posterior_is_exhausted():
    q = pd.DataFrame({"p10": [80], "p50": [150], "p90": [300]})
    elapsed = np.array([6000])  # far past MAX_REVIEW_DAYS
    accepted = pd.to_datetime(pd.Series(["2010-01-01"]))
    res = update_conclusion(q, elapsed, accepted, np.array(["overdue"]),
                            mu_shift=0.0, sigma_inflation=1.0)
    assert res["bayes_note"][0] == "beyond_model"
    assert res["bayes_remaining_days_p50"][0] == 0
