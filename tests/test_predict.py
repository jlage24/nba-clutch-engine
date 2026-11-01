"""Unit tests for production prediction API (src/predict.py)."""

import numpy as np
import pandas as pd
import pytest

from src.features import SRC, build_features, split
from src.predict import expected_points, load_artifacts, predict_make_probability


def test_load_artifacts():
    """Model artifacts should load cleanly with correct types and non-empty feature list."""
    model, platt_params, features = load_artifacts()
    assert model is not None
    assert isinstance(features, list)
    assert len(features) == 11
    assert "SHOT_DISTANCE_FT" in features
    assert "ACTION_GROUP" in features
    assert "IS_PUTBACK" in features
    assert "LOC_PLACEHOLDER" not in features
    assert "IS_PLAYOFFS" not in features
    if platt_params is not None:
        assert "coef" in platt_params
        assert "intercept" in platt_params


def test_predict_single_shot_bounds():
    """Single-shot inference returns valid probability between 0 and 1."""
    shot_2pt = {
        "LOC_X": 0.0,
        "LOC_Y": 10.0,
        "ACTION_TYPE": "Driving Dunk Shot",
        "SHOT_TYPE": "2PT Field Goal",
        "TIME_REMAINING_SEC": 90.0,
        "SCORE_MARGIN": 0.0,
        "PERIOD": 4,
        "location": "H",
    }
    p_raw = predict_make_probability(shot_2pt, calibrated=False)
    p_cal = predict_make_probability(shot_2pt, calibrated=True)
    xp_raw = expected_points(shot_2pt, calibrated=False)
    xp_cal = expected_points(shot_2pt, calibrated=True)

    assert isinstance(p_raw, float)
    assert isinstance(p_cal, float)
    assert 0.0 <= p_raw <= 1.0
    assert 0.0 <= p_cal <= 1.0
    assert np.isclose(xp_raw, p_raw * 2.0)
    assert np.isclose(xp_cal, p_cal * 2.0)


def test_single_dict_equals_batch_predictions():
    """Predictions for individual dicts must match batch predictions on a DataFrame."""
    raw_df = pd.read_parquet(SRC)
    feats = build_features(raw_df)
    _, test_df = split(feats)

    # Take first 10 shots from the held-out test set
    sample_batch = test_df.head(10).copy()

    # Batch predictions
    batch_raw = predict_make_probability(sample_batch, calibrated=False)
    batch_cal = predict_make_probability(sample_batch, calibrated=True)
    batch_xp = expected_points(sample_batch, calibrated=False)

    assert isinstance(batch_raw, np.ndarray)
    assert len(batch_raw) == 10

    # Single-row dict predictions
    for i in range(10):
        single_row = sample_batch.iloc[i].to_dict()
        single_p_raw = predict_make_probability(single_row, calibrated=False)
        single_p_cal = predict_make_probability(single_row, calibrated=True)
        single_xp = expected_points(single_row, calibrated=False)

        assert np.isclose(single_p_raw, batch_raw[i], atol=1e-6), f"Mismatch at index {i} (raw)"
        assert np.isclose(single_p_cal, batch_cal[i], atol=1e-6), f"Mismatch at index {i} (cal)"
        assert np.isclose(single_xp, batch_xp[i], atol=1e-6), f"Mismatch at index {i} (xp)"


def test_enforce_min_distance_non_putback():
    """Verify that non-putback shots with distance < 1.0 ft are clamped to 1.0 ft."""
    # 1. Non-putback layup specified with distance 0 ft should match one specified at 1.0 ft
    shot_zero_layup = {
        "LOC_X": 0.0,
        "LOC_Y": 0.0,
        "SHOT_DISTANCE_FT": 0.0,
        "SHOT_ANGLE": 0.0,
        "ACTION_TYPE": "Driving Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
        "TIME_REMAINING_SEC": 60.0,
        "SCORE_MARGIN": 0.0,
        "PERIOD": 4,
        "location": "H",
    }
    shot_one_layup = {
        "LOC_X": 0.0,
        "LOC_Y": 0.0,
        "SHOT_DISTANCE_FT": 1.0,
        "SHOT_ANGLE": 0.0,
        "ACTION_TYPE": "Driving Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
        "TIME_REMAINING_SEC": 60.0,
        "SCORE_MARGIN": 0.0,
        "PERIOD": 4,
        "location": "H",
    }
    p_zero = predict_make_probability(shot_zero_layup, calibrated=False)
    p_one = predict_make_probability(shot_one_layup, calibrated=False)
    assert np.isclose(p_zero, p_one, atol=1e-5), "Non-putback shot at 0 ft should be clamped to 1.0 ft"

    # 2. Putback shot at 0 ft should NOT be clamped to 1.0 ft (and has lower make prob)
    shot_putback = {
        "LOC_X": 0.0,
        "LOC_Y": 0.0,
        "SHOT_DISTANCE_FT": 0.0,
        "SHOT_ANGLE": 0.0,
        "ACTION_TYPE": "Tip Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
        "TIME_REMAINING_SEC": 60.0,
        "SCORE_MARGIN": 0.0,
        "PERIOD": 4,
        "location": "H",
    }
    p_putback = predict_make_probability(shot_putback, calibrated=False)
    assert not np.isclose(p_putback, p_one, atol=1e-3), "Putback shot should keep distance 0.0 and putback attribution"
    assert p_putback < p_one, "Putbacks under rim should have lower probability than controlled layups"
