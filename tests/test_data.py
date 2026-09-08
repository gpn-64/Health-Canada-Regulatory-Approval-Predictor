"""Data-cleaning and feature-pipeline guarantees.

These run against the real workbook in data/raw/ (the file is small and public),
so they double as an integration check on the loader.
"""

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import CleaningReport, load_completed, load_under_review
from src.features import FrequencyEncoder, build_features, build_preprocessor, output_feature_names


@pytest.fixture(scope="module")
def clean():
    return load_completed(CleaningReport())


def test_no_missing_dates(clean):
    assert clean["date_accepted"].isna().sum() == 0
    assert clean["date_concluded"].isna().sum() == 0


def test_no_negative_review_days(clean):
    assert (clean["review_days"] < 0).sum() == 0


def test_target_is_binary(clean):
    assert set(clean["approved"].unique()) <= {0, 1}


def test_class_flags_are_binary(clean):
    flags = clean[C.CLASS_FLAG_FEATURES]
    assert flags.isin([0, 1]).all().all()


def test_negative_row_is_reported_and_dropped():
    report = CleaningReport()
    load_completed(report)
    assert report.rows_negative_review_days >= 1
    assert 300215 in report.negative_control_numbers


def test_snds_rows_flagged_supplemental(clean):
    snds = clean[clean["submission_type"] == "SNDS"]
    assert (snds["is_supplemental"] == 1).all()
    assert (clean[clean["submission_type"] == "NDS"]["is_supplemental"] == 0).all()


def test_feature_matrix_has_expected_columns(clean):
    X = build_features(clean)
    assert list(X.columns) == C.FEATURE_COLUMNS


def test_preprocessor_output_names_match(clean):
    X = build_features(clean)
    prep = build_preprocessor().fit(X, clean["review_days"])
    assert list(prep.get_feature_names_out()) == output_feature_names()


def test_frequency_encoder_no_leak_on_unseen_category():
    train = pd.DataFrame({"c": ["a", "a", "b"]})
    enc = FrequencyEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "zzz"]}))
    assert out[0, 0] == pytest.approx(2 / 3)
    assert out[1, 0] == pytest.approx(1 / 3)
    assert out[2, 0] == 0.0  # unseen -> 0


def test_under_review_loads_with_same_feature_columns():
    pending = load_under_review()
    assert len(pending) > 0
    X = build_features(pending)
    assert list(X.columns) == C.FEATURE_COLUMNS
    assert np.isfinite(X["accept_year"]).mean() > 0.9
