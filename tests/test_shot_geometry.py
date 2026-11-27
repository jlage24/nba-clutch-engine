"""Tests for court-coordinate geometry helpers in src/shot_geometry.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shot_geometry import (
    compute_shot_distance_and_angle,
    compute_shot_value,
)
from src.simulator_model import (
    DEFAULT_EXPORT_DIR,
    load_export,
    predict_make_probability,
)

ROOT = Path(__file__).resolve().parents[1]
ALL_SHOTS_PATH = ROOT / "data" / "processed" / "all_shots.parquet"
ALL_SHOTS_PRESENT = ALL_SHOTS_PATH.exists()
EXPORT_PRESENT = (DEFAULT_EXPORT_DIR / "model_a.json").exists()


def test_distance_truncation_and_origin_clamp() -> None:
    """Distance truncates to integer feet and clamps non-putbacks below 1 ft to 1.0."""
    # (175, 202) -> hypot/10 = 26.726... -> floor 26.0
    dist, ang = compute_shot_distance_and_angle(175.0, 202.0, is_putback=0)
    assert dist == 26.0
    assert ang == pytest.approx(np.degrees(np.arctan2(175.0, 202.0)))

    # Origin non-putback clamps to 1.0; putback stays 0.0
    dist_non_pb, ang_zero = compute_shot_distance_and_angle(-0.0, 0.0, is_putback=0)
    assert dist_non_pb == 1.0
    assert ang_zero == 0.0

    dist_pb, _ = compute_shot_distance_and_angle(0.0, 0.0, is_putback=1)
    assert dist_pb == 0.0

    # Unclamped non-putback stays 0.0
    dist_raw, _ = compute_shot_distance_and_angle(5.0, 5.0, is_putback=0, clamp_distance=False)
    assert dist_raw == 0.0


def test_shot_value_boundary_cases() -> None:
    """Three-point rule uses 22 ft in corners and 23.75 ft on the arc."""
    assert compute_shot_value(220.0, 0.0) == 3
    assert compute_shot_value(-220.0, 50.0) == 3
    assert compute_shot_value(219.0, 0.0) == 2
    assert compute_shot_value(0.0, 238.0) == 3
    assert compute_shot_value(0.0, 237.0) == 2
    assert compute_shot_value(0.0, 15.0) == 2


def test_vectorized_matches_scalar() -> None:
    """Vectorized calls match element-wise scalar outputs."""
    xs = np.array([0.0, -220.0, 175.0, 10.0])
    ys = np.array([0.0, 20.0, 202.0, 238.0])
    pbs = np.array([1, 0, 0, 0])

    dists, angs = compute_shot_distance_and_angle(xs, ys, is_putback=pbs)
    vals = compute_shot_value(xs, ys)

    for i in range(len(xs)):
        d_i, a_i = compute_shot_distance_and_angle(float(xs[i]), float(ys[i]), is_putback=int(pbs[i]))
        v_i = compute_shot_value(float(xs[i]), float(ys[i]))
        assert dists[i] == d_i
        assert angs[i] == a_i
        assert vals[i] == v_i


@pytest.mark.skipif(
    not ALL_SHOTS_PRESENT,
    reason="data/processed/all_shots.parquet not found; skipping data geometry agreement tests.",
)
def test_geometry_agreement_on_all_shots() -> None:
    """Helper reproduces stored distance, angle, and shot value on all_shots.parquet."""
    df = pd.read_parquet(ALL_SHOTS_PATH)

    dist_raw, ang = compute_shot_distance_and_angle(
        df["LOC_X"].to_numpy(),
        df["LOC_Y"].to_numpy(),
        is_putback=df["IS_PUTBACK"].to_numpy(),
        clamp_distance=False,
    )
    stored_dist = df["SHOT_DISTANCE_FT"].to_numpy(dtype=float)
    stored_ang = df["SHOT_ANGLE"].to_numpy(dtype=float)

    assert np.array_equal(dist_raw, stored_dist)
    assert np.max(np.abs(dist_raw - stored_dist)) == 0.0
    assert np.array_equal(ang, stored_ang)
    assert np.max(np.abs(ang - stored_ang)) == 0.0

    pred_val = compute_shot_value(df["LOC_X"].to_numpy(), df["LOC_Y"].to_numpy())
    stored_val = df["SHOT_VALUE"].to_numpy(dtype=np.int64)
    mismatch_rate = float(np.mean(pred_val != stored_val))
    # Only shots right on the 3PT boundary (~20 of 219,160) should differ
    assert mismatch_rate < 0.0005


@pytest.mark.skipif(
    not (ALL_SHOTS_PRESENT and EXPORT_PRESENT),
    reason="data/processed/all_shots.parquet or export/ not found; skipping probability-effect test.",
)
def test_geometry_probability_effect_on_20k_sample() -> None:
    """Recomputing features via shot_geometry has negligible effect on predicted p_make."""
    df = pd.read_parquet(ALL_SHOTS_PATH)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=20_000, replace=False)
    sample = df.iloc[idx].copy().reset_index(drop=True)

    dist_clamped, ang = compute_shot_distance_and_angle(
        sample["LOC_X"].to_numpy(),
        sample["LOC_Y"].to_numpy(),
        is_putback=sample["IS_PUTBACK"].to_numpy(),
        clamp_distance=True,
    )
    val = compute_shot_value(sample["LOC_X"].to_numpy(), sample["LOC_Y"].to_numpy())

    recomputed = sample.copy()
    recomputed["SHOT_DISTANCE_FT"] = dist_clamped
    recomputed["SHOT_ANGLE"] = ang
    recomputed["SHOT_VALUE"] = val

    model = load_export(DEFAULT_EXPORT_DIR)
    p_stored = predict_make_probability(model, sample, clutch_shift=False)
    p_helper = predict_make_probability(model, recomputed, clutch_shift=False)

    abs_diff = np.abs(p_stored - p_helper)
    mean_diff = float(np.mean(abs_diff))
    max_diff = float(np.max(abs_diff))

    assert mean_diff < 1e-4
    assert max_diff < 0.05
