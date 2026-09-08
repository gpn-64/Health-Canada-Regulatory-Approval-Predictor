"""Bootstrap uncertainty bands for the approval probability.

For a single submission the "answer" is a probability P(approved). Two things are
uncertain about it:

  1. **Estimation uncertainty** — how much that probability would move if the
     model had been trained on a different sample of past submissions. This is
     what the bootstrap interval below measures.
  2. **Outcome uncertainty** — even at the true probability the outcome is a coin
     flip weighted by P. An interval cannot remove this; a submission at P=0.85
     with a tight band is still approved ~85 % of the time.

Method: fit the classifier pipeline on ``n_boot`` bootstrap resamples of the
training data, predict each new submission with every fitted model, and take the
median and a percentile interval across those predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils import resample

from . import config as C
from .models import make_classifier_pipeline


@dataclass
class IntervalConfig:
    n_boot: int = 200
    level: float = 0.90  # central probability mass covered by [low, high]
    random_state: int = C.RANDOM_STATE


def bootstrap_approval_proba(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_new: pd.DataFrame,
    cfg: IntervalConfig | None = None,
    collect_oob: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return an ``(len(X_new), n_boot)`` matrix of predicted approval probabilities.

    With ``collect_oob=True`` also return a length-``len(X_train)`` vector of
    out-of-bag mean probabilities (each training row scored only by the models
    whose resample excluded it) — an honest calibration reference.
    """
    cfg = cfg or IntervalConfig()
    rng = np.random.RandomState(cfg.random_state)
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    n = len(X_train)

    preds = np.empty((len(X_new), cfg.n_boot), dtype=float)
    oob_sum = np.zeros(n)
    oob_count = np.zeros(n)
    for b in range(cfg.n_boot):
        idx = resample(
            np.arange(n), replace=True, n_samples=n,
            random_state=rng.randint(0, 2**31 - 1),
        )
        if y_train.iloc[idx].nunique() < 2:
            idx = np.arange(n)
        pipe = make_classifier_pipeline()
        pipe.fit(X_train.iloc[idx], y_train.iloc[idx])
        preds[:, b] = pipe.predict_proba(X_new)[:, 1]
        if collect_oob:
            oob_mask = ~np.isin(np.arange(n), idx)
            if oob_mask.any():
                oob_sum[oob_mask] += pipe.predict_proba(X_train.iloc[oob_mask])[:, 1]
                oob_count[oob_mask] += 1
    if collect_oob:
        oob = np.divide(oob_sum, oob_count, out=np.full(n, np.nan), where=oob_count > 0)
        return preds, oob
    return preds


def summarize(preds: np.ndarray, level: float = 0.90) -> pd.DataFrame:
    """Point estimate and interval per row of a bootstrap prediction matrix."""
    lo_q = 100 * (1 - level) / 2
    hi_q = 100 * (1 + level) / 2
    return pd.DataFrame(
        {
            "approval_proba": preds.mean(axis=1),
            "approval_proba_median": np.median(preds, axis=1),
            "ci_low": np.percentile(preds, lo_q, axis=1),
            "ci_high": np.percentile(preds, hi_q, axis=1),
            "ci_width": np.percentile(preds, hi_q, axis=1) - np.percentile(preds, lo_q, axis=1),
            "bootstrap_std": preds.std(axis=1),
        }
    )


def coverage_diagnostic(
    oob_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 5
) -> pd.DataFrame:
    """Reliability table: bin training rows by their out-of-bag predicted
    probability and compare the bin's mean prediction to the observed approval
    rate. Close agreement means the probabilities (and therefore the interval
    centres) are trustworthy."""
    s = pd.DataFrame({"p": np.asarray(oob_proba, float), "y": np.asarray(y_true, float)})
    s = s.dropna()
    s["bin"] = pd.qcut(s["p"], q=n_bins, duplicates="drop")
    grouped = s.groupby("bin", observed=True).agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
    )
    grouped["gap"] = grouped["observed_rate"] - grouped["mean_predicted"]
    return grouped.reset_index()
