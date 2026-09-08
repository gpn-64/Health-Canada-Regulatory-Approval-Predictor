"""Discrete-time competing-risks survival model for review duration.

Why discrete time instead of a smooth hazard (Cox / AFT): the empirical
distribution of ``review_days`` is not a smooth curve, it is a comb of sharp
spikes at the regulatory service-standard targets — 43% of completed
submissions conclude in the (270, 300] day bin and 13% in the (150, 180] bin
(the standard and priority-review targets). ``priority_review`` *moves* the
spike from 300 to 180 days rather than rescaling it, which is a hard violation
of the proportional-hazards assumption a single-stratum Cox model relies on.
A discrete-time model with 30-day bins and the bin index as a plain feature
lets tree splits reproduce the spikes exactly and lets XGBoost learn the
bin x priority_review interaction with no PH assumption at all. See
``docs/survival-analysis.md`` for the bin-width comparison that justifies 30
days over 10 or 15.

Two competing causes are modelled per exit bin, alongside the "still in
review" non-event:

  * ``CAUSE_APPROVED``      (1) — an authorisation outcome (see ``src.config``).
  * ``CAUSE_NOT_APPROVED``  (2) — withdrawal / non-compliance / expiry.
  * ``CAUSE_CENSORED``      (0) — right-censored: still under review at the
    snapshot date, or (during expansion) simply "no event yet in this bin".

The pipeline:

  1. :func:`build_case_frame` — one row per submission (completed + under
     review), with a synthetic ``case_id``, the exit bin, and the cause.
  2. :func:`expand_person_periods` — one row per (case, bin up to exit),
     label 0 except on the exit row, which carries the cause. **Must run
     after** any train/test split — the frequency encoder and any grouped CV
     must never see the same case's rows split across folds.
  3. :func:`train_survival_pipeline` — a 3-class ``XGBClassifier`` inside the
     existing :func:`src.features.make_pipeline`, features = ``C.FEATURE_COLUMNS
     + ["time_bin"]``.
  4. :func:`predict_hazards` / :func:`reconstruct_survival` /
     :func:`conditional_outputs` — turn per-bin class probabilities back into
     survival curves, cause-specific cumulative incidence, and (conditional on
     elapsed time) point predictions comparable to the regressor/classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from . import config as C
from .data import load_completed, load_under_review
from .features import make_pipeline, build_features
from .models import train_test_split_frame
from .survival_metrics import brier_ipcw, concordance_harrell, concordance_uno

# --- Paths (kept separate from the original pipeline's artifacts) -----------
SURVIVAL_MODELS_DIR = C.MODELS_DIR / "survival"
SURVIVAL_REPORTS_DIR = C.REPORTS_DIR / "survival"
SURVIVAL_METRICS_FILE = SURVIVAL_REPORTS_DIR / "metrics_survival.json"
UNDER_REVIEW_FORECAST_SURVIVAL_FILE = (
    C.PROCESSED_DIR / "under_review_forecast_survival.csv"
)

# --- Time binning ------------------------------------------------------------
# 30-day bins: verified empirically (docs/survival-analysis.md) to isolate the
# 180-day (priority review) and 300-day (standard review) targets cleanly,
# whereas 10- and 15-day bins split the 300-day spike across a bin boundary.
BIN_WIDTH_DAYS = 30
MAX_BIN = 36  # bins 1..36 cover up to 1,080 days
OPEN_BIN = MAX_BIN + 1  # 37th bin: everything beyond 1,080 days (open-ended)
N_BINS = OPEN_BIN

CAUSE_CENSORED = 0
CAUSE_APPROVED = 1
CAUSE_NOT_APPROVED = 2

TIME_COL = "time_bin"
SURVIVAL_FEATURE_COLUMNS = C.FEATURE_COLUMNS + [TIME_COL]


def day_to_bin(days) -> np.ndarray:
    """Map a day count (or array of them) to a 1..OPEN_BIN bin index."""
    arr = np.asarray(days, dtype=float)
    b = np.ceil(arr / BIN_WIDTH_DAYS).astype(int)
    return np.clip(b, 1, OPEN_BIN)


def bin_to_day(bin_idx) -> np.ndarray:
    """Canonical day estimate for a bin: its right edge (upper bound of the
    30-day window), so a case still at risk at day e always maps to a bin
    whose day estimate is >= e (no clipping ever needed downstream)."""
    arr = np.asarray(bin_idx, dtype=float)
    return arr * BIN_WIDTH_DAYS


def build_case_frame() -> pd.DataFrame:
    """One row per submission (completed + under review): case id, exit bin,
    competing-risks cause, and the standard feature columns."""
    completed = load_completed().reset_index(drop=True)
    completed["case_id"] = "completed_" + completed.index.astype(str)
    completed["exit_bin"] = day_to_bin(completed["review_days"])
    completed["cause"] = np.where(
        completed["approved"] == 1, CAUSE_APPROVED, CAUSE_NOT_APPROVED
    )

    pending = load_under_review().reset_index(drop=True)
    pending["case_id"] = "pending_" + pending.index.astype(str)
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    elapsed = (snap - pd.to_datetime(pending["date_accepted"])).dt.days.clip(lower=0)
    pending["exit_bin"] = day_to_bin(elapsed)
    pending["cause"] = CAUSE_CENSORED
    pending["elapsed_days"] = elapsed

    completed_feat = build_features(completed)
    pending_feat = build_features(pending)

    meta_cols = ["case_id", "exit_bin", "cause", "control_number", "submission_type", "date_accepted"]
    completed_out = pd.concat(
        [completed[meta_cols].reset_index(drop=True), completed_feat.reset_index(drop=True)],
        axis=1,
    )
    completed_out["dataset"] = "completed"
    completed_out["elapsed_days"] = np.nan
    # Kept only so the train/test split can stratify on the exact same column
    # (same values, same dtype) as src.models.train_test_split_frame uses on
    # the original completed frame — guarantees an identical split.
    completed_out[C.TARGET_CLASSIFICATION] = completed["approved"].to_numpy()

    pending_out = pd.concat(
        [pending[meta_cols + ["elapsed_days"]].reset_index(drop=True), pending_feat.reset_index(drop=True)],
        axis=1,
    )
    pending_out["dataset"] = "under_review"

    return pd.concat([completed_out, pending_out], ignore_index=True)


def expand_person_periods(case_df: pd.DataFrame) -> pd.DataFrame:
    """Person-period expansion: for each case, one row per bin ``t = 1..exit_bin``,
    label 0 (still in review) on every row except the last, which carries the
    case's cause (0 if right-censored, 1 approved, 2 not approved).

    Must be called *after* any train/test split of ``case_df`` — expanding
    first would let a case's rows land in both the train and test split.
    """
    exit_bins = case_df["exit_bin"].to_numpy()
    idx = np.repeat(case_df.index.to_numpy(), exit_bins)
    long = case_df.loc[idx].reset_index(drop=True)
    long[TIME_COL] = np.concatenate([np.arange(1, n + 1) for n in exit_bins])
    is_last = long[TIME_COL] == long["exit_bin"]
    long["label"] = 0
    long.loc[is_last, "label"] = long.loc[is_last, "cause"]
    return long


def train_test_case_split(case_df: pd.DataFrame):
    """Split at the case level, matching :func:`src.models.train_test_split_frame`'s
    split of the completed rows (same seed, same stratification) so the *test
    set is identical between the survival model and the original regressor /
    classifier* — the point of ``scripts/compare_models.py``. Under-review
    (censored) cases carry no observed outcome and are appended to the
    training split only; they never appear in a test set.
    """
    completed_cases = case_df[case_df["dataset"] == "completed"].reset_index(drop=True)
    pending_cases = case_df[case_df["dataset"] == "under_review"].reset_index(drop=True)
    train_completed, test_completed = train_test_split_frame(
        completed_cases, stratify_col=C.TARGET_CLASSIFICATION
    )
    train_case = pd.concat([train_completed, pending_cases], ignore_index=True)
    test_case = test_completed.reset_index(drop=True)
    return train_case, test_case


# --- Model --------------------------------------------------------------------
def _survival_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=3,
        n_estimators=300,
        learning_rate=0.05,
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )


def train_survival_pipeline(long_df: pd.DataFrame):
    """Fit the 3-class discrete-time hazard model on an already-expanded
    person-period frame (see :func:`expand_person_periods`)."""
    X = long_df[SURVIVAL_FEATURE_COLUMNS]
    y = long_df["label"].astype(int)
    pipeline = make_pipeline(_survival_model())
    pipeline.fit(X, y)
    return pipeline


# --- Reconstruction: hazards -> survival / cumulative incidence ---------------
def predict_hazards(pipeline, case_df: pd.DataFrame, max_bin: int = N_BINS) -> np.ndarray:
    """Per-case, per-bin class probabilities.

    Returns an ``(n_cases, max_bin, 3)`` array: ``hazards[i, t-1, c]`` is
    ``P(class = c | still at risk entering bin t)`` for case ``i``, class
    ``c`` in ``{0: still in review, 1: approved, 2: not approved}``.
    """
    n = len(case_df)
    base = case_df[C.FEATURE_COLUMNS].reset_index(drop=True)
    tiled = pd.concat(
        [base.assign(**{TIME_COL: t}) for t in range(1, max_bin + 1)],
        ignore_index=True,
    )
    proba = pipeline.predict_proba(tiled[SURVIVAL_FEATURE_COLUMNS])
    return proba.reshape(max_bin, n, 3).transpose(1, 0, 2)


def reconstruct_survival(hazards: np.ndarray) -> dict[str, np.ndarray]:
    """Turn per-bin hazards into survival ``S`` and cause-specific cumulative
    incidence ``CIF1`` / ``CIF2``, each shaped ``(n_cases, max_bin + 1)`` with
    column 0 = t=0 (``S=1``, ``CIF=0``).

    ``S(t) = S(t-1) * h0(t)``; ``CIF_c(t) = CIF_c(t-1) + S(t-1) * h_c(t)``.
    By construction ``S + CIF1 + CIF2 == 1`` at every t.
    """
    n, T, _ = hazards.shape
    S = np.ones((n, T + 1))
    CIF1 = np.zeros((n, T + 1))
    CIF2 = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        h0, h1, h2 = hazards[:, t - 1, 0], hazards[:, t - 1, 1], hazards[:, t - 1, 2]
        S[:, t] = S[:, t - 1] * h0
        CIF1[:, t] = CIF1[:, t - 1] + S[:, t - 1] * h1
        CIF2[:, t] = CIF2[:, t - 1] + S[:, t - 1] * h2
    return {"S": S, "CIF1": CIF1, "CIF2": CIF2}


def conditional_outputs(recon: dict[str, np.ndarray], elapsed_bin) -> dict[str, np.ndarray]:
    """Renormalise ``S`` / ``CIF1`` / ``CIF2`` conditional on having survived
    (still been in review, uncensored and unresolved) through ``elapsed_bin``.

    ``S_cond(t) = S(t) / S(e)``; ``CIF_c,cond(t) = (CIF_c(t) - CIF_c(e)) / S(e)``
    for ``t >= e``. For a case with ``elapsed_bin = 0`` this is the identity
    (no conditioning).
    """
    elapsed_bin = np.asarray(elapsed_bin, dtype=int)
    n = recon["S"].shape[0]
    rows = np.arange(n)
    Se = recon["S"][rows, elapsed_bin]
    Se_safe = np.where(Se <= 0, 1e-12, Se)
    S_cond = recon["S"] / Se_safe[:, None]
    CIF1_cond = (recon["CIF1"] - recon["CIF1"][rows, elapsed_bin][:, None]) / Se_safe[:, None]
    CIF2_cond = (recon["CIF2"] - recon["CIF2"][rows, elapsed_bin][:, None]) / Se_safe[:, None]
    return {"S": S_cond, "CIF1": CIF1_cond, "CIF2": CIF2_cond}


def _first_crossing_bin(cum_prob: np.ndarray, q: float) -> np.ndarray:
    """Smallest bin index (column) at which each row's cumulative probability
    reaches ``q``; the last column if it never does (heavy right tail)."""
    reached = cum_prob >= q
    any_reached = reached.any(axis=1)
    idx = np.where(any_reached, reached.argmax(axis=1), cum_prob.shape[1] - 1)
    return idx


@dataclass
class SurvivalOutputs:
    p_approval_ultimate: np.ndarray  # CIF1 at the final (open) bin
    p_approval_by_horizon: np.ndarray  # CIF1 at a chosen horizon (conditional)
    median_day: np.ndarray
    p10_day: np.ndarray
    p90_day: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "p_approval_ultimate": self.p_approval_ultimate,
                "p_approval_by_horizon": self.p_approval_by_horizon,
                "median_day": self.median_day,
                "p10_day": self.p10_day,
                "p90_day": self.p90_day,
            }
        )


def predict_case_outputs(
    pipeline,
    case_df: pd.DataFrame,
    elapsed_days=None,
    horizon_days: float | None = None,
) -> SurvivalOutputs:
    """Per-case business outputs, conditional on time already elapsed.

    ``elapsed_days`` defaults to 0 for every case (unconditional forecast from
    acceptance). ``horizon_days`` defaults to the open-bin boundary (1,080 d).
    """
    n = len(case_df)
    if elapsed_days is None:
        elapsed_days = np.zeros(n)
    elapsed_bin = np.clip(day_to_bin(elapsed_days) - 1, 0, MAX_BIN)  # bin already survived through
    # A case with 0 elapsed days has not yet "survived" any bin -> condition on t=0.
    elapsed_bin = np.where(np.asarray(elapsed_days) <= 0, 0, elapsed_bin)

    horizon_bin_scalar = (
        MAX_BIN if horizon_days is None else int(np.clip(np.ceil(horizon_days / BIN_WIDTH_DAYS), 1, MAX_BIN))
    )
    # Never evaluate a horizon earlier than the bin already conditioned on.
    horizon_bin = np.maximum(horizon_bin_scalar, elapsed_bin)

    hazards = predict_hazards(pipeline, case_df, max_bin=N_BINS)
    recon = reconstruct_survival(hazards)
    cond = conditional_outputs(recon, elapsed_bin)

    total_exit_cond = cond["CIF1"] + cond["CIF2"]
    median_bin = _first_crossing_bin(total_exit_cond, 0.5)
    p10_bin = _first_crossing_bin(total_exit_cond, 0.1)
    p90_bin = _first_crossing_bin(total_exit_cond, 0.9)

    rows = np.arange(n)
    return SurvivalOutputs(
        p_approval_ultimate=cond["CIF1"][:, -1],
        p_approval_by_horizon=cond["CIF1"][rows, horizon_bin],
        median_day=bin_to_day(median_bin),
        p10_day=bin_to_day(p10_bin),
        p90_day=bin_to_day(p90_bin),
    )


# --- Cross-validation (grouped by case, expanded per fold) --------------------
def cross_validate_survival(train_case_df: pd.DataFrame, n_splits: int = 5) -> dict:
    """GroupKFold (grouped by ``case_id``) cross-validation.

    Expansion into person-periods happens *inside* each fold, after the
    split, so no case's rows ever appear in both a fold's train and
    validation data. Only completed cases in the validation fold have an
    observed outcome, so metrics are computed on those.
    """
    case_df = train_case_df.reset_index(drop=True)
    n_splits = min(n_splits, case_df["case_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    groups = case_df["case_id"].to_numpy()

    all_time, all_event, all_median_day, all_p10, all_p90 = [], [], [], [], []
    all_p_approval_700 = []

    for tr_idx, va_idx in gkf.split(case_df, groups=groups):
        tr_cases = case_df.iloc[tr_idx].reset_index(drop=True)
        va_cases = case_df.iloc[va_idx].reset_index(drop=True)
        va_completed = va_cases[va_cases["dataset"] == "completed"].reset_index(drop=True)
        if va_completed.empty:
            continue
        assert not (set(tr_cases["case_id"]) & set(va_completed["case_id"]))

        long_tr = expand_person_periods(tr_cases)
        pipe = train_survival_pipeline(long_tr)

        outputs = predict_case_outputs(pipe, va_completed, elapsed_days=np.zeros(len(va_completed)))
        actual_days = bin_to_day(va_completed["exit_bin"].to_numpy())  # bin-quantised actual, matches model resolution
        event = (va_completed["cause"] != CAUSE_CENSORED).astype(int).to_numpy()

        all_time.append(actual_days)
        all_event.append(event)
        all_median_day.append(outputs.median_day)
        all_p10.append(outputs.p10_day)
        all_p90.append(outputs.p90_day)
        all_p_approval_700.append(
            predict_case_outputs(pipe, va_completed, horizon_days=700).p_approval_by_horizon
        )

    time_days = np.concatenate(all_time)
    event = np.concatenate(all_event)
    median_day = np.concatenate(all_median_day)
    p10 = np.concatenate(all_p10)
    p90 = np.concatenate(all_p90)

    risk = -median_day
    inside = (time_days >= p10) & (time_days <= p90)
    over_700 = time_days > 700

    return {
        "n_folds": n_splits,
        "harrell_c": concordance_harrell(time_days, event, risk),
        "uno_c": concordance_uno(time_days, event, risk),
        "brier_ipcw_700d": brier_ipcw(
            time_days, event, 1 - np.concatenate(all_p_approval_700), 700
        ),
        "median_ae_days": float(np.median(np.abs(time_days[event == 1] - median_day[event == 1]))),
        "interval_empirical_coverage": float(inside.mean()),
        "coverage_actual_over_700d": float(inside[over_700].mean()) if over_700.any() else None,
    }
