"""Train and evaluate the review-time regressor and the approval classifier.

Both models are gradient-boosted trees kept deliberately small (shallow trees,
low learning rate, strong L2) because the training set is only ~1,500 rows and
the signal is modest. Every metric is reported next to a naive baseline so the
lift is explicit.

Design notes:
  * The regressor is trained on ``log1p(review_days)`` with an absolute-error
    objective — the target is heavily right-skewed (8 to 3,000+ days) and this
    combination roughly halves the median absolute error versus a plain
    squared-error fit on the raw scale. Callers get natural-scale predictions
    through :func:`predict_review_days`.
  * ``n_estimators`` is fixed (not early-stopped): the held-out validation slice
    needed for early stopping is too small and noisy here to be reliable, so we
    rely on regularisation and cross-validation instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from xgboost import XGBClassifier, XGBRegressor

from . import config as C
from .features import make_pipeline

_COMMON_PARAMS = dict(
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    min_child_weight=3,
    random_state=C.RANDOM_STATE,
    n_jobs=-1,
)


def _regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=300,
        learning_rate=0.05,
        **_COMMON_PARAMS,
    )


# Decision threshold for the "flag for review" use of the classifier. The target
# is ~91% positive, so a submission is flagged when its predicted approval
# probability drops below this (roughly the bottom third of the distribution).
FLAG_THRESHOLD = 0.90


def _classifier() -> XGBClassifier:
    # No scale_pos_weight: left natural, the XGBoost logistic outputs are already
    # well calibrated here (mean predicted ≈ base rate; see docs/methodology.md),
    # which matters for the probability + interval use in src/uncertainty.py.
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        learning_rate=0.03,
        **_COMMON_PARAMS,
    )


# --- Regressor -----------------------------------------------------------------
def train_regressor(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the pipeline. The returned pipeline predicts in *log1p* space —
    use :func:`predict_review_days` for natural-scale day counts."""
    pipeline = make_pipeline(_regressor())
    pipeline.fit(X_train, np.log1p(y_train))
    return pipeline


def predict_review_days(pipeline, X) -> np.ndarray:
    return np.expm1(pipeline.predict(X))


# Quantiles for the review-time prediction interval (p10 / p50 / p90 -> 80 % band).
QUANTILE_ALPHAS = (0.1, 0.5, 0.9)


def _quantile_regressor(alpha: float) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=300,
        learning_rate=0.05,
        **_COMMON_PARAMS,
    )


def train_quantile_regressors(
    X_train: pd.DataFrame, y_train: pd.Series, alphas=QUANTILE_ALPHAS
) -> dict[float, object]:
    """One pipeline per quantile, each fit on ``log1p(review_days)``."""
    models = {}
    for a in alphas:
        pipe = make_pipeline(_quantile_regressor(a))
        pipe.fit(X_train, np.log1p(y_train))
        models[a] = pipe
    return models


def predict_review_days_quantiles(
    models: dict[float, object], X, alphas=QUANTILE_ALPHAS
) -> pd.DataFrame:
    """Natural-scale day predictions per quantile.

    The separate quantile fits can cross on individual rows; we sort each row so
    the columns are non-decreasing (``p10 <= p50 <= p90``).
    """
    cols = {f"p{int(a * 100)}": np.expm1(models[a].predict(X)) for a in alphas}
    out = pd.DataFrame(cols)
    out[:] = np.sort(out.to_numpy(), axis=1)
    return out


def cross_validate_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    pipeline = make_pipeline(_regressor())
    cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    pred = np.expm1(
        cross_val_predict(pipeline, X_train, np.log1p(y_train), cv=cv)
    )
    y = y_train.to_numpy(float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "median_ae": float(median_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def cross_validate_quantiles(
    X_train: pd.DataFrame, y_train: pd.Series, alphas=QUANTILE_ALPHAS
) -> dict:
    """Out-of-fold empirical coverage of the p10–p90 band and its median width."""
    cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    yl = np.log1p(y_train)
    oof = {}
    for a in alphas:
        pipe = make_pipeline(_quantile_regressor(a))
        oof[a] = np.expm1(cross_val_predict(pipe, X_train, yl, cv=cv))
    lo, hi = np.minimum(oof[alphas[0]], oof[alphas[-1]]), np.maximum(oof[alphas[0]], oof[alphas[-1]])
    y = y_train.to_numpy(float)
    inside = (y >= lo) & (y <= hi)
    return {
        "nominal_coverage": alphas[-1] - alphas[0],
        "empirical_coverage": float(inside.mean()),
        "median_interval_days": float(np.median(hi - lo)),
        "coverage_actual_over_700d": float(inside[y > 700].mean()) if (y > 700).any() else None,
    }


def evaluate_regressor(pipeline, X_test, y_test, y_train) -> dict:
    pred = predict_review_days(pipeline, X_test)
    baseline = np.full(len(y_test), float(np.median(y_train)))
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(root_mean_squared_error(y_test, pred)),
        "median_ae": float(median_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "baseline_rmse": float(root_mean_squared_error(y_test, baseline)),
    }


# --- Classifier ---------------------------------------------------------------
def make_classifier_pipeline():
    """Unfitted approval-classifier pipeline (used for training and bootstrapping)."""
    return make_pipeline(_classifier())


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series):
    pipeline = make_classifier_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    pipeline = make_classifier_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    y = y_train.to_numpy(int)
    pred = (proba >= FLAG_THRESHOLD).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "mean_predicted_proba": float(proba.mean()),
        "balanced_accuracy_at_flag": float(balanced_accuracy_score(y, pred)),
    }


def evaluate_classifier(pipeline, X_test, y_test, y_train, threshold: float = FLAG_THRESHOLD) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    positive_rate = float((y_train == 1).mean())
    y_rej = (y_test == 0).astype(int)
    pred_rej = (pred == 0).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "flag_threshold": threshold,
        "balanced_accuracy_at_flag": float(balanced_accuracy_score(y_test, pred)),
        "rejection_recall": float(recall_score(y_rej, pred_rej, zero_division=0)),
        "rejection_precision": float(precision_score(y_rej, pred_rej, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "baseline_roc_auc": 0.5,
        "baseline_pr_auc": positive_rate,
        "baseline_brier": float(brier_score_loss(y_test, np.full(len(y_test), positive_rate))),
    }


# --- Shared splitter ---------------------------------------------------------
def train_test_split_frame(df: pd.DataFrame, stratify_col: str, test_size: float = 0.2):
    """Single stratified split reused by both models so the test rows match."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=C.RANDOM_STATE,
        stratify=df[stratify_col],
    )
