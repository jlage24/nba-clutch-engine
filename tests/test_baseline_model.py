"""Unit tests for baseline_model.py (Phase 3b Step 2).

Tests safeguards:
- assert_no_clutch_in_training raises if CLUTCH or LATE_BLOWOUT rows are present
- NaN SCORE_MARGIN does not contaminate baseline features or break prediction
- GroupKFold split guarantees no game appears in both train and validation splits
- Artifact-dependent tests skip cleanly when artifacts are not generated
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.baseline_model import (
    BASELINE_FEATURES,
    assert_no_clutch_in_training,
    make_fold_assignment,
    _X,
)
from src.features import ACTION_DTYPE


def test_assert_no_clutch_in_training_raises():
    """assert_no_clutch_in_training must raise ValueError if any CLUTCH shot is present."""
    df = pd.DataFrame({
        "GAME_ID": ["0022500001", "0022500002"],
        "CLUTCH": [True, False],
        "LATE_BLOWOUT": [False, False],
        "REGULAR": [False, True],
    })
    with pytest.raises(ValueError, match="CLUTCH"):
        assert_no_clutch_in_training(df, fold_k=0, phase="test-train")


def test_assert_no_clutch_in_training_raises_for_blowout():
    """assert_no_clutch_in_training must raise ValueError if any LATE_BLOWOUT shot is present."""
    df = pd.DataFrame({
        "GAME_ID": ["0022500001", "0022500002"],
        "CLUTCH": [False, False],
        "LATE_BLOWOUT": [True, False],
        "REGULAR": [False, True],
    })
    with pytest.raises(ValueError, match="LATE_BLOWOUT"):
        assert_no_clutch_in_training(df, fold_k=0, phase="test-train")


def test_assert_no_clutch_in_training_passes_regular_only():
    """assert_no_clutch_in_training must pass silently when all shots are REGULAR."""
    df = pd.DataFrame({
        "GAME_ID": ["0022500001", "0022500002"],
        "CLUTCH": [False, False],
        "LATE_BLOWOUT": [False, False],
        "REGULAR": [True, True],
    })
    # Should not raise
    assert_no_clutch_in_training(df, fold_k=0, phase="test-train")


def test_nan_margin_does_not_break_features():
    """Model A features exclude SCORE_MARGIN; a shot with NaN margin has valid features."""
    shot_with_nan_margin = pd.DataFrame([{
        "GAME_ID": "0022500792",
        "GAME_EVENT_ID": 403,
        "LOC_X": 10.0,
        "LOC_Y": 50.0,
        "SHOT_DISTANCE_FT": 5.1,
        "SHOT_ANGLE": 11.3,
        "SHOT_VALUE": 2,
        "ACTION_GROUP": "layup",
        "IS_PUTBACK": 0,
        "SCORE_MARGIN": np.nan,  # NaN margin from unmatched PBP
        "CLUTCH": False,
        "LATE_BLOWOUT": False,
        "REGULAR": True,
    }])
    X = _X(shot_with_nan_margin)
    assert not X.isna().any().any(), "Baseline feature matrix contains unexpected NaNs!"
    assert "SCORE_MARGIN" not in X.columns, "SCORE_MARGIN leaked into Model A features!"
    assert list(X.columns) == BASELINE_FEATURES


def test_no_game_in_both_train_and_predict_set():
    """For every fold, no GAME_ID appears in both training games and validation games."""
    # Synthetic dataset of 20 games
    games = [f"002250000{i:02d}" for i in range(20)]
    all_shots = pd.DataFrame({
        "GAME_ID": np.repeat(games, 5),
    })
    fold_assign = make_fold_assignment(all_shots)
    merged = all_shots.merge(fold_assign, on="GAME_ID", how="left")

    for k in range(5):
        train_games = set(merged[merged["fold"] != k]["GAME_ID"])
        val_games = set(merged[merged["fold"] == k]["GAME_ID"])
        overlap = train_games.intersection(val_games)
        assert len(overlap) == 0, f"Fold {k} has overlapping games between train and val: {overlap}"


def test_fold_assignment_file_validity():
    """If fold_assignment.parquet exists, verify game disjointness across all 5 folds."""
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "fold_assignment.parquet"
    if not path.exists():
        pytest.skip("data/processed/fold_assignment.parquet not found — run baseline_model.py first")

    df = pd.read_parquet(path)
    assert len(df) == 1230, f"Expected 1,230 games, got {len(df)}"
    assert sorted(df["fold"].unique()) == [0, 1, 2, 3, 4]
    assert df["GAME_ID"].duplicated().sum() == 0, "Duplicate GAME_IDs in fold assignment"


def test_oof_predictions_file_validity():
    """If oof_predictions.parquet exists, verify row counts, column presence, and no NaNs in p_A."""
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "oof_predictions.parquet"
    if not path.exists():
        pytest.skip("data/processed/oof_predictions.parquet not found — run baseline_model.py first")

    df = pd.read_parquet(path)
    assert len(df) == 219160, f"Expected 219,160 shots, got {len(df):,}"
    expected_cols = [
        "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME",
        "SET_LABEL", "SHOT_VALUE", "PERIOD", "fold",
        "SHOT_MADE_FLAG", "p_A",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in oof_predictions.parquet"

    assert df["p_A"].isna().sum() == 0, f"Found {df['p_A'].isna().sum()} NaN p_A predictions"
    assert (df["p_A"] >= 0.0).all() and (df["p_A"] <= 1.0).all(), "p_A out of bounds [0, 1]"
