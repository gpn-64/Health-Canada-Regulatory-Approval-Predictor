"""Checks on the quantile review-time models."""

import numpy as np
import pytest

from src import models as M
from src.data import load_completed
from src.features import build_features


@pytest.fixture(scope="module")
def split():
    df = load_completed()
    X = build_features(df)
    y = df["review_days"]
    return X.iloc[:1000], y.iloc[:1000], X.iloc[1000:], y.iloc[1000:]


def test_quantile_predictions_are_monotone(split):
    X_tr, y_tr, X_te, _ = split
    models = M.train_quantile_regressors(X_tr, y_tr)
    q = M.predict_review_days_quantiles(models, X_te)
    assert list(q.columns) == ["p10", "p50", "p90"]
    assert (q["p10"] <= q["p50"]).all()
    assert (q["p50"] <= q["p90"]).all()
    assert (q["p10"] >= 0).all()


def test_quantile_interval_roughly_covers(split):
    X_tr, y_tr, X_te, y_te = split
    models = M.train_quantile_regressors(X_tr, y_tr)
    q = M.predict_review_days_quantiles(models, X_te)
    covered = ((y_te.to_numpy() >= q["p10"]) & (y_te.to_numpy() <= q["p90"])).mean()
    # nominal 80% band; allow slack for a small held-out slice
    assert 0.60 <= covered <= 0.95


def test_point_regressor_beats_median_baseline(split):
    X_tr, y_tr, X_te, y_te = split
    model = M.train_regressor(X_tr, y_tr)
    pred = M.predict_review_days(model, X_te)
    mae = np.mean(np.abs(y_te.to_numpy() - pred))
    baseline = np.mean(np.abs(y_te.to_numpy() - np.median(y_tr)))
    assert mae < baseline
