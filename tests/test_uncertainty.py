"""Shape and sanity checks for the bootstrap approval-interval code."""

import numpy as np
import pytest

from src.data import load_completed
from src.features import build_features
from src.uncertainty import IntervalConfig, bootstrap_approval_proba, summarize


@pytest.fixture(scope="module")
def small_inputs():
    df = load_completed()
    X = build_features(df)
    y = df["approved"]
    return X.iloc[:400], y.iloc[:400], X.iloc[400:430]


def test_bootstrap_shape_and_range(small_inputs):
    X_tr, y_tr, X_new = small_inputs
    cfg = IntervalConfig(n_boot=12)
    preds, oob = bootstrap_approval_proba(X_tr, y_tr, X_new, cfg, collect_oob=True)
    assert preds.shape == (len(X_new), 12)
    assert ((preds >= 0) & (preds <= 1)).all()
    assert oob.shape == (len(X_tr),)
    assert np.nanmin(oob) >= 0 and np.nanmax(oob) <= 1


def test_summarize_interval_ordering(small_inputs):
    X_tr, y_tr, X_new = small_inputs
    preds = bootstrap_approval_proba(X_tr, y_tr, X_new, IntervalConfig(n_boot=12))
    s = summarize(preds, level=0.90)
    assert (s["ci_low"] <= s["approval_proba_median"]).all()
    assert (s["approval_proba_median"] <= s["ci_high"]).all()
    assert (s["ci_width"] >= 0).all()
    assert len(s) == len(X_new)


def test_bootstrap_is_deterministic(small_inputs):
    X_tr, y_tr, X_new = small_inputs
    a = bootstrap_approval_proba(X_tr, y_tr, X_new, IntervalConfig(n_boot=8))
    b = bootstrap_approval_proba(X_tr, y_tr, X_new, IntervalConfig(n_boot=8))
    assert np.allclose(a, b)
