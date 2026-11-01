import numpy as np
import pandas as pd
import pytest

from src.features import (
    ACTION_CATEGORIES,
    FEATURES,
    action_group,
    build_features,
    split,
)


def test_action_group_mapping():
    """Verify action types are mapped to expected shot families."""
    raw_actions = pd.Series([
        "Dunk Shot",
        "Cutting Dunk Shot",
        "Driving Layup Shot",
        "Driving Finger Roll Layup Shot",
        "Turnaround Hook Shot",
        "Driving Floating Jump Shot",
        "Step Back Jump shot",
        "Running Pull-Up Jump Shot",
        "Turnaround Fadeaway shot",
        "Jump Shot",
        "Jump Bank Shot",
        "Some Rare Unseen Trick Shot",
    ])
    result = action_group(raw_actions)

    expected = [
        "dunk",
        "dunk",
        "layup",
        "layup",
        "hook",
        "floater",
        "stepback",
        "pullup",
        "fadeaway",
        "jump_shot",
        "jump_shot",
        "other",
    ]
    assert list(result) == expected
    assert set(result.cat.categories) == set(ACTION_CATEGORIES)


def test_polar_coordinates():
    """Verify conversion of LOC_X and LOC_Y to distance and angle."""
    df = pd.DataFrame([
        # Straight on, 10 ft away
        {"LOC_X": 0, "LOC_Y": 100, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1, "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "2PT Field Goal", "ACTION_TYPE": "Jump Shot"},
        # Right corner / baseline, 22 ft away
        {"LOC_X": 220, "LOC_Y": 0, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1, "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "3PT Field Goal", "ACTION_TYPE": "Jump Shot"},
        # Left corner / baseline, 22 ft away
        {"LOC_X": -220, "LOC_Y": 0, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1, "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "3PT Field Goal", "ACTION_TYPE": "Jump Shot"},
        # Behind the backboard, 5 ft away
        {"LOC_X": 0, "LOC_Y": -50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1, "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "2PT Field Goal", "ACTION_TYPE": "Jump Shot"},
    ])

    feats = build_features(df)

    # 1. Straight on
    assert feats.loc[0, "SHOT_DISTANCE_FT"] == pytest.approx(10.0)
    assert feats.loc[0, "SHOT_ANGLE"] == pytest.approx(0.0)

    # 2. Right baseline (+90 degrees)
    assert feats.loc[1, "SHOT_DISTANCE_FT"] == pytest.approx(22.0)
    assert feats.loc[1, "SHOT_ANGLE"] == pytest.approx(90.0)

    # 3. Left baseline (-90 degrees)
    assert feats.loc[2, "SHOT_DISTANCE_FT"] == pytest.approx(22.0)
    assert feats.loc[2, "SHOT_ANGLE"] == pytest.approx(-90.0)

    # 4. Behind backboard (180 degrees)
    assert feats.loc[3, "SHOT_DISTANCE_FT"] == pytest.approx(5.0)
    assert feats.loc[3, "SHOT_ANGLE"] == pytest.approx(180.0)


def test_shot_value():
    """Verify 2PT vs 3PT value assignment."""
    df_2pt = pd.DataFrame([{
        "LOC_X": 0, "LOC_Y": 50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1,
        "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "2PT Field Goal",
        "ACTION_TYPE": "Layup Shot",
    }])
    df_3pt = pd.DataFrame([{
        "LOC_X": 0, "LOC_Y": 250, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1,
        "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "3PT Field Goal",
        "ACTION_TYPE": "Jump Shot",
    }])

    assert build_features(df_2pt)["SHOT_VALUE"].iloc[0] == 2
    assert build_features(df_3pt)["SHOT_VALUE"].iloc[0] == 3


def test_is_ot():
    """Verify overtime indicator for regulation (period <= 4) and OT (period > 4)."""
    df = pd.DataFrame([
        {"PERIOD": 4, "LOC_X": 0, "LOC_Y": 50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 2, "SECONDS_REMAINING": 0, "SHOT_TYPE": "2PT", "ACTION_TYPE": "Jump Shot"},
        {"PERIOD": 5, "LOC_X": 0, "LOC_Y": 50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 2, "SECONDS_REMAINING": 0, "SHOT_TYPE": "2PT", "ACTION_TYPE": "Jump Shot"},
        {"PERIOD": 6, "LOC_X": 0, "LOC_Y": 50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 2, "SECONDS_REMAINING": 0, "SHOT_TYPE": "2PT", "ACTION_TYPE": "Jump Shot"},
    ])
    feats = build_features(df)
    assert feats["IS_OT"].tolist() == [0, 1, 1]


def test_game_id_parsing():
    """Verify season extraction and playoff flag from GAME_ID."""
    df = pd.DataFrame([
        {"GAME_ID": "0022500001", "LOC_X": 0, "LOC_Y": 50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1, "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "2PT", "ACTION_TYPE": "Jump Shot"},
        {"GAME_ID": "0042400101", "LOC_X": 0, "LOC_Y": 50, "SCORE_MARGIN": 0, "MINUTES_REMAINING": 1, "SECONDS_REMAINING": 0, "PERIOD": 4, "SHOT_TYPE": "2PT", "ACTION_TYPE": "Jump Shot"},
    ])
    feats = build_features(df)
    assert feats.loc[0, "SEASON"] == 2025
    assert feats.loc[0, "IS_PLAYOFFS"] == 0
    assert feats.loc[1, "SEASON"] == 2024
    assert feats.loc[1, "IS_PLAYOFFS"] == 1


def test_build_features_single_dict():
    """Verify that build_features handles a single dict input for live/single inference."""
    shot_dict = {
        "PERIOD": 4,
        "MINUTES_REMAINING": 1,
        "SECONDS_REMAINING": 30,
        "SCORE_MARGIN": -2,
        "ACTION_TYPE": "Driving Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
        "LOC_X": 10,
        "LOC_Y": 20,
        "location": "h",
        "GAME_ID": "0022500123",
    }
    result = build_features(shot_dict)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    for feat in FEATURES:
        assert feat in result.columns
        assert not result[feat].isna().any()

    assert result["TIME_REMAINING_SEC"].iloc[0] == 90.0
    assert result["SCORE_MARGIN"].iloc[0] == -2
    assert result["IS_HOME"].iloc[0] == 1
    assert result["IS_OT"].iloc[0] == 0
    assert result["SHOT_VALUE"].iloc[0] == 2
    assert result["ACTION_GROUP"].iloc[0] == "layup"


def test_split_grouped_no_leakage():
    """Verify that grouped split does not leak games across train and test sets."""
    df = pd.DataFrame({
        "GAME_ID": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4", "G5", "G5"],
        "SEASON": [2025] * 10,
        "TIME_REMAINING_SEC": [100.0] * 10,
        "SCORE_MARGIN": [0.0] * 10,
        "SHOT_DISTANCE_FT": [15.0] * 10,
        "SHOT_ANGLE": [0.0] * 10,
        "LOC_X": [0.0] * 10,
        "LOC_Y": [150.0] * 10,
        "SHOT_VALUE": [2] * 10,
        "IS_OT": [0] * 10,
        "IS_HOME": [1] * 10,
        "IS_PLAYOFFS": [0] * 10,
        "ACTION_GROUP": pd.Categorical(["jump_shot"] * 10, categories=ACTION_CATEGORIES),
        "SHOT_MADE_FLAG": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    train, test = split(df)
    train_games = set(train["GAME_ID"])
    test_games = set(test["GAME_ID"])
    assert len(train_games.intersection(test_games)) == 0
    assert len(train) + len(test) == len(df)
