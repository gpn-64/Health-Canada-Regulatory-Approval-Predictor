"""Survival-native evaluation metrics, implemented directly rather than via
``scikit-survival`` (that package pins scikit-learn versions incompatible with
this environment's sklearn 1.9 / Python 3.13 — not worth the install risk for
the ~100 lines below).

All functions take right-censored ``(time, event)`` pairs where ``event`` is 1
if the observed time is a true event and 0 if censored, plus a per-subject
risk score or survival probability. They work for the discrete-time survival
model (any-cause "risk score" = ``1 - S_cond`` at a horizon, or ``-median_day``)
and equally for the original regressor (risk score = ``-predicted_days``,
since a smaller predicted duration means a case is expected to conclude
sooner — higher risk of an early event).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _km_censoring_survival(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the *censoring* distribution G(t) = P(C > t),
    i.e. the KM fit obtained by swapping the roles of "event" and "censored".
    Returns (unique_times, G_at_unique_times), both sorted ascending, plus an
    implicit G=1 before the first unique time.
    """
    order = np.argsort(time)
    t_sorted = time[order]
    censor_event = 1 - event[order]  # "event" for the censoring distribution
    unique_times = np.unique(t_sorted)
    G = np.empty(len(unique_times))
    S = 1.0
    for i, ut in enumerate(unique_times):
        at_risk = (t_sorted >= ut).sum()
        d = censor_event[t_sorted == ut].sum()
        if at_risk > 0:
            S *= 1 - d / at_risk
        G[i] = S
    return unique_times, G


def _g_at(unique_times: np.ndarray, G: np.ndarray, t) -> np.ndarray:
    """Step-function lookup of G at arbitrary time(s) t (G=1 before the first
    unique time; G takes the last known value beyond the last one)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    idx = np.searchsorted(unique_times, t, side="right") - 1
    out = np.where(idx < 0, 1.0, G[np.clip(idx, 0, len(G) - 1)])
    return out


def concordance_harrell(time, event, risk_score) -> float:
    """Harrell's C-index: among all pairs where the earlier time is an
    observed event, the fraction correctly ordered by risk_score (higher
    score = expected sooner event)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)
    n = len(time)
    num = 0.0
    den = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(num / den) if den > 0 else float("nan")


def concordance_uno(time, event, risk_score, tau: float | None = None) -> float:
    """Uno's C-index: like Harrell's, but pairs are weighted by the inverse
    squared probability of remaining uncensored past the earlier event time
    (IPCW), which corrects the bias Harrell's C has under heavy censoring."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)
    tau = float(time.max()) if tau is None else tau
    ut, G = _km_censoring_survival(time, event)

    n = len(time)
    num = 0.0
    den = 0.0
    for i in range(n):
        if event[i] != 1 or time[i] > tau:
            continue
        gi = max(_g_at(ut, G, time[i])[0], 1e-6)
        w = 1.0 / (gi**2)
        later = time > time[i]
        den += w * later.sum()
        num += w * ((risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum())
    return float(num / den) if den > 0 else float("nan")


def brier_ipcw(time, event, surv_prob, horizon: float) -> float:
    """IPCW Brier score at a single horizon tau: mean squared error between
    the predicted survival probability S(tau) and the (weighted) observed
    "still at risk at tau" indicator, weighted by the inverse probability of
    being uncensored long enough to know the true label."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    surv_prob = np.asarray(surv_prob, dtype=float)
    ut, G = _km_censoring_survival(time, event)

    g_ti = np.maximum(_g_at(ut, G, time), 1e-6)
    g_tau = max(float(_g_at(ut, G, np.array([horizon]))[0]), 1e-6)

    had_event_before = (time <= horizon) & (event == 1)
    still_at_risk = time > horizon

    bs = np.zeros(len(time))
    bs[had_event_before] = (0 - surv_prob[had_event_before]) ** 2 / g_ti[had_event_before]
    bs[still_at_risk] = (1 - surv_prob[still_at_risk]) ** 2 / g_tau
    # censored before horizon without an event: excluded (weight 0), the
    # standard IPCW convention since we cannot know their tau-outcome.
    return float(bs.mean())


def integrated_brier_ipcw(time, event, surv_prob_fn, horizons) -> float:
    """Mean of :func:`brier_ipcw` over a grid of horizons. ``surv_prob_fn``
    maps a horizon to a per-subject array of predicted S(horizon)."""
    horizons = np.asarray(list(horizons), dtype=float)
    scores = [brier_ipcw(time, event, surv_prob_fn(h), h) for h in horizons]
    return float(np.mean(scores))


def time_dependent_auc(time, event, risk_score, horizon: float) -> float:
    """AUC(tau): among subjects with an observed event by tau ("cases") vs.
    subjects still at risk past tau ("controls", censored or not), the
    probability risk_score ranks a random case above a random control.
    Subjects censored before tau without an event are excluded (their
    tau-outcome is unknown)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)

    case = (time <= horizon) & (event == 1)
    control = time > horizon
    n1, n2 = int(case.sum()), int(control.sum())
    if n1 == 0 or n2 == 0:
        return float("nan")

    combined = np.concatenate([risk[case], risk[control]])
    ranks = pd.Series(combined).rank().to_numpy()
    auc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return float(auc)
