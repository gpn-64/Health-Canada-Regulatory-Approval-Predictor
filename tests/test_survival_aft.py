"""Shape and behaviour checks for the continuous parametric AFT model."""

import numpy as np
import pytest

from src.data import load_completed
from src.features import build_features
from src.survival_aft import MAX_REVIEW_DAYS, AFTModel


@pytest.fixture(scope="module")
def fitted():
    df = load_completed().iloc[:600]
    X = build_features(df)
    T = df["review_days"].to_numpy(float)
    # censor the last 80 rows to exercise the censored path
    E = np.ones(len(df))
    E[-80:] = 0
    T[-80:] = np.minimum(T[-80:], 250)
    return AFTModel(distribution="lognormal").fit(X, T, E), X


def test_quantiles_are_monotone_and_bounded(fitted):
    model, X = fitted
    q = model.predict_quantiles(X.iloc[:50])
    assert list(q.columns) == ["p10", "p50", "p90"]
    assert (q["p10"] <= q["p50"]).all() and (q["p50"] <= q["p90"]).all()
    assert (q.to_numpy() >= 1).all() and (q.to_numpy() <= MAX_REVIEW_DAYS).all()


def test_conditional_after_pushes_the_estimate_later(fitted):
    model, X = fitted
    base = model.predict_median_days(X.iloc[:50])
    later = model.predict_median_days(X.iloc[:50], conditional_after=np.full(50, 400.0))
    assert (later >= 400.0).all()
    assert np.median(later) > np.median(base)


def test_constant_columns_are_dropped(fitted):
    model, _ = fitted
    assert "_keep" in model.__dict__ and len(model._keep) >= 5


def test_distribution_choice_is_validated():
    with pytest.raises(KeyError):
        AFTModel(distribution="normal").fit(
            build_features(load_completed().iloc[:100]),
            load_completed()["review_days"].to_numpy(float)[:100],
            np.ones(100),
        )
