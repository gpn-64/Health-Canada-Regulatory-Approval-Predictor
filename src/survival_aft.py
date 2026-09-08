"""Continuous parametric AFT model for review duration (lifelines).

The discrete-time model in :mod:`src.survival` handles censoring well but pays
~11 days of median absolute error to 30-day time bins. A parametric
*accelerated failure time* model is the continuous alternative: it also consumes
the in-flight submissions as right-censored observations, but predicts a real
number of days, so the bin-quantisation floor on median AE disappears.

    log(review_days) = X·β + σ·ε      (ε ~ distribution-specific)

with a lognormal error by default — consistent with the rest of the project
modelling ``log1p(review_days)`` — and Weibull / log-logistic available for the
robustness check.

Covariates go through the same leak-safe :func:`src.features.build_preprocessor`
(frequency encoding fit on training rows only), then are z-scored (AFT
convergence and a fair ridge ``penalizer`` both want standardised inputs).
Near-constant columns are dropped.

``conditional_after`` on the prediction path is the survival-conditioned
re-projection done natively — same idea as ``src/bayesian.py`` on the ``baysian``
branch, but from a model that has actually seen censored reviews.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

from .features import build_preprocessor, output_feature_names

MAX_REVIEW_DAYS = 7 * 365

_FITTERS = {
    "lognormal": LogNormalAFTFitter,
    "weibull": WeibullAFTFitter,
    "loglogistic": LogLogisticAFTFitter,
}


@dataclass
class AFTModel:
    """Leak-safe parametric AFT over ``review_days``.

    Parameters
    ----------
    distribution
        ``"lognormal"`` (default), ``"weibull"`` or ``"loglogistic"``.
    penalizer
        lifelines ridge penalty on the (standardised) coefficients.
    """

    distribution: str = "lognormal"
    penalizer: float = 0.02

    def fit(self, X: pd.DataFrame, durations, event_observed) -> "AFTModel":
        self._prep = build_preprocessor().fit(X)
        Z = self._design(X, first=True)

        df = Z.copy()
        df["_T"] = np.asarray(durations, dtype=float).clip(min=1.0)
        df["_E"] = np.asarray(event_observed, dtype=int)

        Fitter = _FITTERS[self.distribution]
        self._fitter = Fitter(penalizer=self.penalizer)
        self._fitter.fit(df, duration_col="_T", event_col="_E")
        return self

    # --- prediction ---------------------------------------------------------
    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        conditional_after=None,
    ) -> pd.DataFrame:
        """Day-count quantiles of ``review_days`` per row.

        ``quantiles`` are quantiles *of the duration* (0.5 = median). With
        ``conditional_after`` (elapsed days per row) each quantile is the
        posterior given the review is still open at that point.
        """
        Z = self._design(X)
        ca = None if conditional_after is None else np.asarray(conditional_after, float)
        out = {}
        for q in quantiles:
            # lifelines predict_percentile(p) returns t with S(t) = p
            t = self._fitter.predict_percentile(Z, p=1.0 - q, conditional_after=ca)
            t = np.asarray(t, dtype=float)
            t = np.where(np.isfinite(t), t, MAX_REVIEW_DAYS)
            if ca is not None:
                t = np.maximum(t, ca)
            out[f"p{int(round(q * 100))}"] = np.clip(t, 1.0, MAX_REVIEW_DAYS)
        df = pd.DataFrame(out, index=np.arange(len(X)))
        df[:] = np.sort(df.to_numpy(), axis=1)  # enforce monotone rows
        return df

    def predict_median_days(self, X: pd.DataFrame, conditional_after=None) -> np.ndarray:
        return self.predict_quantiles(X, (0.50,), conditional_after)["p50"].to_numpy()

    # --- internals ---------------------------------------------------------
    def _design(self, X: pd.DataFrame, first: bool = False) -> pd.DataFrame:
        """Preprocess -> DataFrame of z-scored covariates, constant columns dropped."""
        mat = np.asarray(self._prep.transform(X), dtype=float)
        Z = pd.DataFrame(mat, columns=output_feature_names(), index=np.arange(len(X)))
        if first:
            std = Z.std(ddof=0)
            self._keep = std[std > 1e-9].index.tolist()
            self._mean = Z[self._keep].mean()
            self._std = Z[self._keep].std(ddof=0).replace(0.0, 1.0)
        return (Z[self._keep] - self._mean) / self._std

    @property
    def summary(self) -> pd.DataFrame:
        return self._fitter.summary
