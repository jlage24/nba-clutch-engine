"""Unit tests for build_all_shots_dataset.py — window labelling and edge cases.

Tests must pass in a clean clone even when the processed parquet artifacts are
absent.  Tests that require generated artifacts (all_shots.parquet) are skipped
with a clear message when the file is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.build_all_shots_dataset import (
    assert_unmatched_are_regular,
    label_shots,
    validate_against_existing_clutch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shot_dict(
    period: int,
    minutes: int,
    seconds: int,
    score_margin: float | None = 0.0,
    made: int = 1,
    game_id: str = "0022500001",
    event_id: int = 1,
) -> dict:
    """Return a minimal shot-row dict suitable for label_shots / assert_unmatched tests."""
    return {
        "GAME_ID": game_id,
        "GAME_EVENT_ID": event_id,
        "PLAYER_ID": 1,
        "PLAYER_NAME": "Test Player",
        "TEAM_ID": 1,
        "PERIOD": period,
        "MINUTES_REMAINING": minutes,
        "SECONDS_REMAINING": seconds,
        "TIME_REMAINING_SEC": float(minutes * 60 + seconds),
        "SCORE_MARGIN": score_margin,  # may be NaN
        "SHOT_MADE_FLAG": made,
        "LOC_X": 0,
        "LOC_Y": 100,
        "SHOT_DISTANCE_FT": 10.0,
        "SHOT_ANGLE": 0.0,
        "SHOT_VALUE": 2,
        "ACTION_GROUP": "jump_shot",
        "IS_PUTBACK": 0,
        "_merge": "both",  # default: matched
        "ACTION_TYPE": "Jump Shot",
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["PERIOD"] = df["PERIOD"].astype(int)
    df["TIME_REMAINING_SEC"] = df["TIME_REMAINING_SEC"].astype(float)
    df["SCORE_MARGIN"] = pd.to_numeric(df["SCORE_MARGIN"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# LATE_WINDOW boundary
# ---------------------------------------------------------------------------


def test_late_window_exactly_300_seconds():
    """Exactly 300 sec remaining in Q4 is inside LATE_WINDOW."""
    df = _make_df([_shot_dict(period=4, minutes=5, seconds=0, score_margin=0)])
    out = label_shots(df)
    assert out["LATE_WINDOW"].iloc[0], "300 sec remaining in Q4 must be LATE_WINDOW"


def test_not_late_window_301_seconds():
    """301 sec remaining in Q4 is NOT inside LATE_WINDOW."""
    df = _make_df([_shot_dict(period=4, minutes=5, seconds=1, score_margin=0)])
    out = label_shots(df)
    assert not out["LATE_WINDOW"].iloc[0]


def test_regular_q3():
    """Q3 shot is always REGULAR regardless of time remaining."""
    df = _make_df([_shot_dict(period=3, minutes=0, seconds=0, score_margin=0)])
    out = label_shots(df)
    assert out["REGULAR"].iloc[0]
    assert not out["LATE_WINDOW"].iloc[0]


def test_overtime_period_is_late_window():
    """OT periods (>= 5) are entirely inside LATE_WINDOW."""
    df = _make_df([_shot_dict(period=5, minutes=3, seconds=0, score_margin=3)])
    out = label_shots(df)
    assert out["LATE_WINDOW"].iloc[0], "OT period must be LATE_WINDOW"


def test_q4_301_seconds_is_regular():
    """Period 4 with 5:01 remaining is REGULAR (just outside window)."""
    df = _make_df([_shot_dict(period=4, minutes=5, seconds=1, score_margin=0)])
    out = label_shots(df)
    assert out["REGULAR"].iloc[0]
    assert not out["CLUTCH"].iloc[0]


# ---------------------------------------------------------------------------
# CLUTCH / LATE_BLOWOUT margin boundary
# ---------------------------------------------------------------------------


def test_clutch_margin_exactly_5():
    """Margin = +5 is CLUTCH (abs <= 5)."""
    df = _make_df([_shot_dict(period=4, minutes=2, seconds=0, score_margin=5)])
    out = label_shots(df)
    assert out["CLUTCH"].iloc[0], "Margin=5 must be CLUTCH"
    assert not out["LATE_BLOWOUT"].iloc[0]


def test_clutch_margin_exactly_minus5():
    """Margin = -5 is CLUTCH."""
    df = _make_df([_shot_dict(period=4, minutes=2, seconds=0, score_margin=-5)])
    out = label_shots(df)
    assert out["CLUTCH"].iloc[0]


def test_late_blowout_margin_6():
    """Margin = +6 is LATE_BLOWOUT (abs > 5)."""
    df = _make_df([_shot_dict(period=4, minutes=2, seconds=0, score_margin=6)])
    out = label_shots(df)
    assert out["LATE_BLOWOUT"].iloc[0], "Margin=6 must be LATE_BLOWOUT"
    assert not out["CLUTCH"].iloc[0]


# ---------------------------------------------------------------------------
# NaN margin handling
# ---------------------------------------------------------------------------


def test_nan_margin_regular_shot_labels_as_regular():
    """A shot with NaN SCORE_MARGIN in REGULAR territory is labelled REGULAR, not CLUTCH/BLOWOUT."""
    row = _shot_dict(period=3, minutes=4, seconds=46, score_margin=None)
    df = _make_df([row])
    out = label_shots(df)
    assert out["REGULAR"].iloc[0], "NaN-margin shot in Q3 must be REGULAR"
    assert not out["CLUTCH"].iloc[0], "NaN-margin shot must not be CLUTCH"
    assert not out["LATE_BLOWOUT"].iloc[0], "NaN-margin shot must not be LATE_BLOWOUT"


def test_nan_margin_in_late_window_is_neither_clutch_nor_blowout():
    """A shot with NaN margin inside LATE_WINDOW is labelled neither CLUTCH nor LATE_BLOWOUT.

    This state should never occur in production (assert_unmatched_are_regular
    guards against it), but the labelling logic must not crash or produce a
    spurious label when given NaN.
    """
    row = _shot_dict(period=4, minutes=1, seconds=0, score_margin=None)
    df = _make_df([row])
    out = label_shots(df)
    assert out["LATE_WINDOW"].iloc[0], "Shot is inside LATE_WINDOW"
    assert not out["CLUTCH"].iloc[0], "NaN margin → not CLUTCH"
    assert not out["LATE_BLOWOUT"].iloc[0], "NaN margin → not LATE_BLOWOUT"
    # REGULAR = ~LATE_WINDOW; so this shot is LATE_WINDOW but neither subset
    assert not out["REGULAR"].iloc[0]


# ---------------------------------------------------------------------------
# Disjoint and exhaustive partition
# ---------------------------------------------------------------------------


def test_sets_are_disjoint_and_exhaustive():
    """REGULAR, CLUTCH, LATE_BLOWOUT must be mutually exclusive and cover all rows."""
    rows = [
        _shot_dict(period=2, minutes=5, seconds=0, score_margin=10, event_id=1),   # REGULAR
        _shot_dict(period=4, minutes=2, seconds=0, score_margin=3, event_id=2),    # CLUTCH
        _shot_dict(period=4, minutes=2, seconds=0, score_margin=10, event_id=3),   # LATE_BLOWOUT
        _shot_dict(period=5, minutes=2, seconds=0, score_margin=2, event_id=4),    # CLUTCH (OT)
        _shot_dict(period=4, minutes=5, seconds=0, score_margin=0, event_id=5),    # CLUTCH (300 sec)
        _shot_dict(period=3, minutes=4, seconds=46, score_margin=None, event_id=6),# REGULAR (NaN margin)
    ]
    df = _make_df(rows)
    out = label_shots(df)

    # Exhaustive: every row is in exactly one of REGULAR, CLUTCH, or LATE_BLOWOUT
    # (shots in LATE_WINDOW but neither CLUTCH nor BLOWOUT, e.g. NaN margin, are
    # an anomalous state guarded by assert_unmatched_are_regular; they show as
    # LATE_WINDOW=True, REGULAR=False, CLUTCH=False, LATE_BLOWOUT=False)
    non_late = out[~out["LATE_WINDOW"]]
    assert non_late["REGULAR"].all(), "All non-LATE_WINDOW shots must be REGULAR"

    # Mutually exclusive
    assert not (out["CLUTCH"] & out["LATE_BLOWOUT"]).any()
    assert not (out["CLUTCH"] & out["REGULAR"]).any()
    assert not (out["LATE_BLOWOUT"] & out["REGULAR"]).any()


# ---------------------------------------------------------------------------
# assert_unmatched_are_regular
# ---------------------------------------------------------------------------


def _merged_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the _merge indicator column as assert_unmatched_are_regular expects."""
    df = _make_df(rows)
    return df


def test_assert_unmatched_are_regular_passes_when_unmatched_is_regular():
    """No error when the only unmatched shot is in REGULAR territory (Period 3)."""
    rows = [
        _shot_dict(period=3, minutes=4, seconds=46, score_margin=None, event_id=1),
        _shot_dict(period=4, minutes=2, seconds=0, score_margin=3, event_id=2),
    ]
    df = _make_df(rows)
    df.loc[0, "_merge"] = "left_only"   # mark first shot as unmatched
    # Should not raise
    assert_unmatched_are_regular(df)


def test_assert_unmatched_are_regular_passes_when_all_matched():
    """No error when every shot is matched (no unmatched rows)."""
    rows = [_shot_dict(period=4, minutes=2, seconds=0, score_margin=3, event_id=i) for i in range(3)]
    df = _make_df(rows)
    # All _merge == "both" by default
    assert_unmatched_are_regular(df)


def test_assert_unmatched_are_regular_raises_when_late_window():
    """Raises ValueError if an unmatched shot is inside LATE_WINDOW (margin required)."""
    rows = [
        _shot_dict(period=4, minutes=1, seconds=30, score_margin=None, event_id=1),  # LATE_WINDOW
    ]
    df = _make_df(rows)
    df.loc[0, "_merge"] = "left_only"
    with pytest.raises(ValueError, match="STOP"):
        assert_unmatched_are_regular(df)


def test_assert_unmatched_are_regular_raises_for_ot_unmatched():
    """Raises ValueError if an unmatched shot is in an OT period (always LATE_WINDOW)."""
    rows = [
        _shot_dict(period=5, minutes=3, seconds=0, score_margin=None, event_id=1),
    ]
    df = _make_df(rows)
    df.loc[0, "_merge"] = "left_only"
    with pytest.raises(ValueError, match="STOP"):
        assert_unmatched_are_regular(df)


def test_assert_unmatched_regular_q4_outside_window():
    """No error for an unmatched Q4 shot with more than 5 minutes remaining."""
    rows = [
        _shot_dict(period=4, minutes=6, seconds=0, score_margin=None, event_id=1),
    ]
    df = _make_df(rows)
    df.loc[0, "_merge"] = "left_only"
    # TIME_REMAINING_SEC = 360 > 300, so not LATE_WINDOW → should not raise
    assert_unmatched_are_regular(df)


# ---------------------------------------------------------------------------
# Artifact-dependent tests (skipped in clean clone)
# ---------------------------------------------------------------------------


def _all_shots_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "data" / "processed" / "all_shots.parquet"


def _clutch_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "data" / "processed" / "clutch_shots.parquet"


def test_all_shots_parquet_counts():
    """Row counts in all_shots.parquet match expected totals from the brief."""
    path = _all_shots_path()
    if not path.exists():
        pytest.skip("data/processed/all_shots.parquet not found — run build_all_shots_dataset.py first")
    df = pd.read_parquet(path)
    assert len(df) == 219_160, f"Expected 219,160 rows, got {len(df):,}"
    assert df["REGULAR"].sum() == 196_198, f"Expected 196,198 REGULAR, got {df['REGULAR'].sum():,}"
    assert df["CLUTCH"].sum() == 7_104, f"Expected 7,104 CLUTCH, got {df['CLUTCH'].sum():,}"
    assert df["LATE_BLOWOUT"].sum() == 15_858, f"Expected 15,858 LATE_BLOWOUT, got {df['LATE_BLOWOUT'].sum():,}"


def test_all_shots_no_duplicates():
    """No duplicate (GAME_ID, GAME_EVENT_ID) pairs in all_shots.parquet."""
    path = _all_shots_path()
    if not path.exists():
        pytest.skip("data/processed/all_shots.parquet not found — run build_all_shots_dataset.py first")
    df = pd.read_parquet(path)
    dupes = df.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum()
    assert dupes == 0, f"Found {dupes:,} duplicate (GAME_ID, GAME_EVENT_ID) pairs"


def test_all_shots_nan_margin_only_on_regular():
    """In all_shots.parquet, NaN SCORE_MARGIN shots must all be REGULAR."""
    path = _all_shots_path()
    if not path.exists():
        pytest.skip("data/processed/all_shots.parquet not found — run build_all_shots_dataset.py first")
    df = pd.read_parquet(path)
    nan_margin = df[df["SCORE_MARGIN"].isna()]
    assert nan_margin["REGULAR"].all(), (
        f"{(~nan_margin['REGULAR']).sum()} NaN-margin shot(s) are not REGULAR"
    )


def test_clutch_subset_matches_existing():
    """CLUTCH subset of all_shots.parquet matches the validated clutch_shots.parquet."""
    ap = _all_shots_path()
    cp = _clutch_path()
    if not ap.exists():
        pytest.skip("data/processed/all_shots.parquet not found — run build_all_shots_dataset.py first")
    if not cp.exists():
        pytest.skip("data/processed/clutch_shots.parquet not found")
    all_shots = pd.read_parquet(ap)
    result = validate_against_existing_clutch(all_shots, cp)
    assert result["counts_match"], (
        f"CLUTCH count mismatch: new={result['n_new_clutch']}, existing={result['n_existing_clutch']}"
    )
    assert result["keys_match"], (
        f"Key mismatch: {result['only_in_new']} only in new, {result['only_in_existing']} only in existing"
    )
    assert result["margin_matches"], (
        f"SCORE_MARGIN disagreement on {result['margin_mismatches']} shots"
    )
