"""Consistency checks for the discrete-time competing-risks survival model.

These do not touch the original regressor/classifier tests (tests/test_models.py,
tests/test_data.py) — the two pipelines are independent and both stay green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import survival as S


@pytest.fixture(scope="module")
def case_df():
    return S.build_case_frame()


@pytest.fixture(scope="module")
def split(case_df):
    train_case, test_case = S.train_test_case_split(case_df)
    long_train = S.expand_person_periods(train_case)
    pipeline = S.train_survival_pipeline(long_train)
    return pipeline, train_case, test_case


def test_case_frame_has_both_datasets(case_df):
    counts = case_df["dataset"].value_counts()
    assert counts["completed"] > 1000
    assert counts["under_review"] > 100


def test_day_to_bin_roundtrip_edges():
    assert S.day_to_bin(1) == 1
    assert S.day_to_bin(30) == 1
    assert S.day_to_bin(31) == 2
    assert S.day_to_bin(300) == 10
    assert S.day_to_bin(10_000) == S.OPEN_BIN


def test_expand_person_periods_labels(case_df):
    long = S.expand_person_periods(case_df.head(50))
    # every case contributes exactly `exit_bin` rows
    counts = long.groupby("case_id").size()
    expected = case_df.head(50).set_index("case_id")["exit_bin"]
    pd.testing.assert_series_equal(counts.sort_index(), expected.sort_index(), check_names=False)
    # label is 0 everywhere except the last row of each case
    last_rows = long[long[S.TIME_COL] == long["exit_bin"]]
    assert (last_rows["label"] == last_rows["cause"]).all()
    non_last = long[long[S.TIME_COL] != long["exit_bin"]]
    assert (non_last["label"] == 0).all()


def test_train_test_case_split_matches_original_pipeline_split(case_df):
    """The point of this split: the survival test set must be identical to
    src.models.train_test_split_frame's test set on the completed data, so
    scripts/compare_models.py compares both families on the same rows."""
    from src import config as C
    from src import models as M

    df = pd.read_csv(C.CLEAN_FILE, parse_dates=["date_accepted", "date_concluded"])
    _, test_df = M.train_test_split_frame(df, stratify_col=C.TARGET_CLASSIFICATION)

    train_case, test_case = S.train_test_case_split(case_df)
    assert set(test_case["control_number"]) == set(test_df["control_number"])
    # censored (under review) cases never appear in the test split
    assert (test_case["dataset"] == "completed").all()


def test_no_case_shared_across_groupkfold_folds(case_df):
    from sklearn.model_selection import GroupKFold

    train_case, _ = S.train_test_case_split(case_df)
    groups = train_case["case_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    for tr_idx, va_idx in gkf.split(train_case, groups=groups):
        tr_ids = set(train_case.iloc[tr_idx]["case_id"])
        va_ids = set(train_case.iloc[va_idx]["case_id"])
        assert not (tr_ids & va_ids)


def test_cif_and_survival_sum_to_one(split):
    pipeline, _, test_case = split
    hazards = S.predict_hazards(pipeline, test_case.head(20))
    recon = S.reconstruct_survival(hazards)
    total = recon["S"] + recon["CIF1"] + recon["CIF2"]
    assert np.allclose(total, 1.0, atol=1e-8)
    assert (recon["CIF1"] >= -1e-9).all() and (recon["CIF1"] <= 1 + 1e-9).all()
    assert (recon["CIF2"] >= -1e-9).all() and (recon["CIF2"] <= 1 + 1e-9).all()


def test_survival_is_monotone_decreasing(split):
    pipeline, _, test_case = split
    hazards = S.predict_hazards(pipeline, test_case.head(20))
    recon = S.reconstruct_survival(hazards)
    diffs = np.diff(recon["S"], axis=1)
    assert (diffs <= 1e-9).all()


def test_conditional_median_never_before_elapsed(split):
    """A case already `e` days into review must have a conditional median
    (and p10/p90) >= e — no clipping needed, this falls out of the maths."""
    pipeline, _, test_case = split
    n = 15
    cases = test_case.head(n)
    for elapsed in [0, 45, 150, 305, 900]:
        elapsed_arr = np.full(n, float(elapsed))
        outputs = S.predict_case_outputs(pipeline, cases, elapsed_days=elapsed_arr)
        assert (outputs.median_day >= elapsed_arr - 1e-9).all()
        assert (outputs.p10_day >= elapsed_arr - 1e-9).all()
        assert (outputs.p90_day >= outputs.median_day - 1e-9).all()


def test_km_matches_discrete_model_survival(case_df, split):
    """The discrete-time model's mean S(t) across all cases should track the
    non-parametric Kaplan-Meier estimate reasonably closely."""
    lifelines = pytest.importorskip("lifelines")
    pipeline, _, _ = split

    df = case_df.copy()
    df["duration"] = S.bin_to_day(df["exit_bin"])
    df["event_any"] = (df["cause"] != S.CAUSE_CENSORED).astype(int)
    km = lifelines.KaplanMeierFitter()
    km.fit(df["duration"], df["event_any"])

    hazards = S.predict_hazards(pipeline, case_df, max_bin=S.N_BINS)
    recon = S.reconstruct_survival(hazards)
    for bin_idx, day in [(5, 150), (10, 300), (20, 600)]:
        model_s = float(recon["S"][:, bin_idx].mean())
        km_s = float(km.survival_function_at_times(day).iloc[0])
        assert abs(model_s - km_s) < 0.12


def test_predict_case_outputs_probabilities_in_range(split):
    pipeline, _, test_case = split
    outputs = S.predict_case_outputs(pipeline, test_case, elapsed_days=np.zeros(len(test_case)))
    assert (outputs.p_approval_ultimate >= 0).all() and (outputs.p_approval_ultimate <= 1).all()
    assert (outputs.p_approval_by_horizon >= 0).all() and (outputs.p_approval_by_horizon <= 1).all()
