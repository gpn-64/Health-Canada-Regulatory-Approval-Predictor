"""Feature engineering and the leak-safe modelling pipeline.

The raw frame from :mod:`src.data` already carries the class flags and the
calendar / ingredient-count features. The only transform that must be fit on
training data alone is frequency encoding of the two high-cardinality columns
(company, therapeutic area) — done here inside a :class:`~sklearn.pipeline.Pipeline`
so cross-validation folds never see the whole dataset's frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from . import config as C


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its relative frequency in the training data.

    Unseen categories at transform time map to ``0.0`` (rarer than anything seen).
    Operates column-wise on a 2-D input (DataFrame or ndarray).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.frequencies_ = {
            col: X[col].astype("string").value_counts(normalize=True).to_dict()
            for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape, dtype=float)
        for j, col in enumerate(X.columns):
            freq = self.frequencies_.get(col, {})
            out[:, j] = X[col].astype("string").map(freq).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{n}_freq" for n in names], dtype=object)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model input columns from a frame produced by :mod:`src.data`.

    Missing calendar columns are derived on the fly so the same function works on
    both the cleaned CSV and freshly loaded frames.
    """
    df = df.copy()
    if "accept_year" not in df.columns:
        accepted = pd.to_datetime(df["date_accepted"])
        df["accept_year"] = accepted.dt.year
        df["accept_month"] = accepted.dt.month
        df["accept_quarter"] = accepted.dt.quarter
    if "n_ingredients" not in df.columns:
        df["n_ingredients"] = (
            df["medicinal_ingredients"].fillna("").astype(str).apply(
                lambda s: len([p for p in s.split(",") if p.strip()])
            )
        )
    return df[C.FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    """Frequency-encode the categorical columns, pass everything else through."""
    return ColumnTransformer(
        transformers=[
            ("freq", FrequencyEncoder(), C.FREQUENCY_ENCODED_FEATURES),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def make_pipeline(model) -> Pipeline:
    """Wrap a fitted-per-fold preprocessor and an estimator into one Pipeline."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def output_feature_names() -> list[str]:
    """Column order produced by :func:`build_preprocessor` (freq cols first)."""
    return [f"{c}_freq" for c in C.FREQUENCY_ENCODED_FEATURES] + (
        C.CLASS_FLAG_FEATURES + C.NUMERIC_FEATURES
    )
