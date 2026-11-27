"""Unit tests for src/decision_layer.py and loader shift extensions."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import src.decision_layer as dl
from src.decision_layer import GameState, evaluate_options, load_zone_presets
from src.simulator_model import (
    DEFAULT_EXPORT_DIR,
    ExportedModel,
    load_export,
    predict_make_probability,
)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_REQUIRED = [
    DEFAULT_EXPORT_DIR / "model_a.json",
    DEFAULT_EXPORT_DIR / "player_offsets.csv",
    DEFAULT_EXPORT_DIR / "clutch_shift.json",
    DEFAULT_EXPORT_DIR / "manifest.json",
    DEFAULT_EXPORT_DIR / "zone_presets.json",
]
EXPORT_PRESENT = all(p.exists() for p in EXPORT_REQUIRED)

pytestmark = pytest.mark.skipif(
    not EXPORT_PRESENT,
    reason="export/ artifacts (including zone_presets.json) not found.",
)

SHOT_2PT = {
    "label": "Custom Mid 2PT",
    "LOC_X": 0.0,
    "LOC_Y": 160.0,
    "SHOT_DISTANCE_FT": 16,
    "SHOT_ANGLE": 0.0,
    "SHOT_VALUE": 2,
    "ACTION_GROUP": "pullup",
    "IS_PUTBACK": 0,
}

SHOT_3PT = {
    "label": "Custom Top 3PT",
    "LOC_X": 0.0,
    "LOC_Y": 260.0,
    "SHOT_DISTANCE_FT": 26,
    "SHOT_ANGLE": 0.0,
    "SHOT_VALUE": 3,
    "ACTION_GROUP": "jump_shot",
    "IS_PUTBACK": 0,
}


@pytest.fixture(scope="module")
def model() -> ExportedModel:
    return load_export(DEFAULT_EXPORT_DIR)


@pytest.fixture(scope="module")
def presets() -> dict:
    return load_zone_presets(DEFAULT_EXPORT_DIR / "zone_presets.json")


def _fake_predict_factory(p_by_val: dict[int, float]):
    def _fake(model_arg, shots, player_id=None, clutch_shift=True, shift_mode="overall", shift_bound="point"):
        vals = [int(s["SHOT_VALUE"]) for s in shots]
        return np.array([p_by_val[v] for v in vals], dtype=float)

    return _fake


def test_hand_checked_end_state_cases(monkeypatch, model, presets) -> None:
    """Hand-checked V and rankings across down 3, down 2, down 1, tied, and down 5."""
    monkeypatch.setattr(dl, "predict_make_probability", _fake_predict_factory({2: 0.55, 3: 0.35}))
    opts = [SHOT_2PT, SHOT_3PT]

    # Down 3, 6s left: V(3PT) = 0.175, V(2PT) = 0 -> 3PT ranks first
    res_d3 = evaluate_options({"time_left_sec": 6.0, "margin": -3}, opts, model=model, presets=presets)
    t3 = res_d3.table.set_index("label")
    assert t3.loc["Custom Top 3PT", "v"] == pytest.approx(0.175)
    assert t3.loc["Custom Mid 2PT", "v"] == pytest.approx(0.0)
    assert t3.loc["Custom Top 3PT", "rank_v"] == 1
    assert t3.loc["Custom Mid 2PT", "rank_v"] == 2
    assert "no_option_changes_end_state" not in res_d3.meta["flags"]

    # Down 2: 2PT V = 0.275, 3PT V = 0.35 -> 3PT ranks first
    res_d2 = evaluate_options({"time_left_sec": 6.0, "margin": -2}, opts, model=model, presets=presets)
    t2 = res_d2.table.set_index("label")
    assert t2.loc["Custom Mid 2PT", "v"] == pytest.approx(0.275)
    assert t2.loc["Custom Top 3PT", "v"] == pytest.approx(0.35)
    assert t2.loc["Custom Top 3PT", "rank_v"] == 1

    # Down 1: V = p for both
    res_d1 = evaluate_options({"time_left_sec": 6.0, "margin": -1}, opts, model=model, presets=presets)
    t1 = res_d1.table.set_index("label")
    assert t1.loc["Custom Mid 2PT", "v"] == pytest.approx(0.55)
    assert t1.loc["Custom Top 3PT", "v"] == pytest.approx(0.35)
    assert t1.loc["Custom Mid 2PT", "rank_v"] == 1

    # Tied: V = 0.5 + 0.5 * p
    res_tied = evaluate_options({"time_left_sec": 6.0, "margin": 0}, opts, model=model, presets=presets)
    tt = res_tied.table.set_index("label")
    assert tt.loc["Custom Mid 2PT", "v"] == pytest.approx(0.5 + 0.5 * 0.55)
    assert tt.loc["Custom Top 3PT", "v"] == pytest.approx(0.5 + 0.5 * 0.35)

    # Down 5: all V = 0 -> no_option_changes_end_state set and ranked by xP (2PT: 1.10 > 3PT: 1.05)
    res_d5 = evaluate_options({"time_left_sec": 6.0, "margin": -5}, opts, model=model, presets=presets)
    t5 = res_d5.table.set_index("label")
    assert t5.loc["Custom Mid 2PT", "v"] == pytest.approx(0.0)
    assert t5.loc["Custom Top 3PT", "v"] == pytest.approx(0.0)
    assert "no_option_changes_end_state" in res_d5.meta["flags"]
    assert res_d5.meta["no_option_changes_end_state"] is True
    assert t5.loc["Custom Mid 2PT", "rank_v"] == 1


def test_outcome_probabilities_sum_to_one(model, presets) -> None:
    """P_ahead + P_tied + P_behind == 1 across random clutch states and options."""
    rng = np.random.default_rng(42)
    opts = ["above_the_break_3", "corner_three", "restricted_area", SHOT_2PT]
    for _ in range(25):
        t = float(rng.uniform(0.0, 24.0))
        m = int(rng.integers(-6, 7))
        p = int(rng.integers(4, 7))
        res = evaluate_options(
            GameState(time_left_sec=t, margin=m, period=p),
            opts,
            model=model,
            presets=presets,
        )
        sums = res.table["p_ahead"] + res.table["p_tied"] + res.table["p_behind"]
        np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-12)


def test_time_gate_open_at_24_and_closed_above(model, presets) -> None:
    """At 24.0s decision columns are populated; at 24.01s they are NaN/NA and flag is set."""
    opts = ["corner_three", "restricted_area"]
    res_open = evaluate_options({"time_left_sec": 24.0, "margin": -2}, opts, model=model, presets=presets)
    assert res_open.meta["gate_open"] is True
    assert "decision_gate_closed" not in res_open.meta["flags"]
    assert res_open.table["v"].notna().all()
    assert res_open.table["rank_v"].notna().all()
    assert res_open.table["rank_xp"].notna().all()

    res_closed = evaluate_options({"time_left_sec": 24.01, "margin": -2}, opts, model=model, presets=presets)
    assert res_closed.meta["gate_open"] is False
    assert "decision_gate_closed" in res_closed.meta["flags"]
    assert res_closed.meta["ranking_robust"] is None
    for col in ("p_ahead", "p_tied", "p_behind", "v", "v_no_shift", "v_shift_low", "v_shift_high"):
        assert res_closed.table[col].isna().all()
        assert res_closed.table[col].dtype == np.float64
    for col in ("rank_v", "rank_xp"):
        assert res_closed.table[col].isna().all()
        assert str(res_closed.table[col].dtype) == "Int64"
    assert res_closed.table["p_make"].notna().all()
    assert res_closed.table["xp"].notna().all()


def test_outside_clutch_state_applies_no_shift_and_sets_flag(model, presets) -> None:
    """Outside clutch (e.g. margin > 5 with period 4, or time > 300), shift is not applied."""
    opts = ["corner_three", "restricted_area"]
    for state in ({"time_left_sec": 10.0, "margin": 8, "period": 4}, {"time_left_sec": 350.0, "margin": 0, "period": 4}):
        res = evaluate_options(state, opts, model=model, presets=presets)
        assert res.meta["is_clutch_state"] is False
        assert res.meta["shift_applied"] is False
        assert "shift_not_applied_outside_clutch" in res.meta["flags"]
        np.testing.assert_allclose(res.table["p_make"], res.table["p_make_no_shift"], atol=1e-12)


def test_shift_modes_none_overall_and_by_shot_type(model, presets) -> None:
    """shift_mode='none' gives baseline; 'overall' lowers it; 'by_shot_type' uses 2PT/3PT shifts."""
    state = {"time_left_sec": 10.0, "margin": -2}
    opts = [SHOT_2PT, SHOT_3PT]

    res_none = evaluate_options(state, opts, shift_mode="none", model=model, presets=presets)
    res_over = evaluate_options(state, opts, shift_mode="overall", model=model, presets=presets)
    res_type = evaluate_options(state, opts, shift_mode="by_shot_type", model=model, presets=presets)

    np.testing.assert_allclose(res_none.table["p_make"], res_none.table["p_make_no_shift"], atol=1e-12)
    assert np.all(res_over.table["p_make"] < res_none.table["p_make"])

    # Direct comparison against predict_make_probability with explicit shift_mode
    p_2pt_type = predict_make_probability(model, [SHOT_2PT], clutch_shift=True, shift_mode="by_shot_type")[0]
    p_3pt_type = predict_make_probability(model, [SHOT_3PT], clutch_shift=True, shift_mode="by_shot_type")[0]
    np.testing.assert_allclose(res_type.table["p_make"].to_numpy(), [p_2pt_type, p_3pt_type], atol=1e-12)

    # Exploratory 3PT shift is more negative than overall, and 2PT is less negative than overall
    assert res_type.table.loc[1, "p_make"] < res_over.table.loc[1, "p_make"]
    assert res_type.table.loc[0, "p_make"] > res_over.table.loc[0, "p_make"]


def test_sensitivity_bounds_ordering(model, presets) -> None:
    """p_make_shift_low < p_make < p_make_shift_high < p_make_no_shift in clutch state."""
    state = {"time_left_sec": 8.0, "margin": -2}
    res = evaluate_options(
        state,
        ["above_the_break_3", "corner_three", "restricted_area"],
        model=model,
        presets=presets,
    )
    assert np.all(res.table["p_make_shift_low"] < res.table["p_make"])
    assert np.all(res.table["p_make"] < res.table["p_make_shift_high"])
    assert np.all(res.table["p_make_shift_high"] < res.table["p_make_no_shift"])


def test_unknown_and_known_player_behavior(model, presets) -> None:
    """Unknown player matches no-player result + flag; known player shifts p in offset direction."""
    state = {"time_left_sec": 8.0, "margin": -2}
    opts = ["corner_three", SHOT_2PT]

    res_none = evaluate_options(state, opts, player_id=None, model=model, presets=presets)
    res_unk = evaluate_options(state, opts, player_id="unknown_999999", model=model, presets=presets)
    assert "unknown_player_offset_zero" in res_unk.meta["flags"]
    assert res_unk.meta["player_offset"] == 0.0
    pd.testing.assert_frame_equal(res_unk.table, res_none.table)

    # Pick high-volume positive and negative offset players without naming them
    pos_row = model.player_table[model.player_table["offset"] > 0.03].iloc[0]
    neg_row = model.player_table[model.player_table["offset"] < -0.03].iloc[0]

    res_pos = evaluate_options(state, opts, player_id=int(pos_row["PLAYER_ID"]), model=model, presets=presets)
    res_neg = evaluate_options(state, opts, player_id=int(neg_row["PLAYER_ID"]), model=model, presets=presets)
    assert "unknown_player_offset_zero" not in res_pos.meta["flags"]
    assert np.all(res_pos.table["p_make_no_shift"] > res_none.table["p_make_no_shift"])
    assert np.all(res_neg.table["p_make_no_shift"] < res_none.table["p_make_no_shift"])


def test_decision_layer_matches_predict_make_probability(model, presets) -> None:
    """Decision layer p_make matches predict_make_probability for custom and preset options."""
    state = {"time_left_sec": 8.0, "margin": -3}
    pid = int(model.player_table.sort_values("n_regular_shots", ascending=False).iloc[0]["PLAYER_ID"])

    for chosen_pid in (None, pid):
        for mode in ("overall", "by_shot_type", "none"):
            res = evaluate_options(
                state,
                ["corner_three", SHOT_2PT],
                player_id=chosen_pid,
                shift_mode=mode,
                model=model,
                presets=presets,
            )
            p_corner = float(
                np.mean(
                    predict_make_probability(
                        model,
                        presets["corner_three"]["shots"],
                        player_id=chosen_pid,
                        clutch_shift=(mode != "none"),
                        shift_mode=mode,
                    )
                )
            )
            p_custom = float(
                predict_make_probability(
                    model,
                    [SHOT_2PT],
                    player_id=chosen_pid,
                    clutch_shift=(mode != "none"),
                    shift_mode=mode,
                )[0]
            )
            assert res.table.loc[0, "n_rows"] == presets["corner_three"]["n_sample"]
            assert res.table.loc[1, "n_rows"] == 1
            assert res.table.loc[0, "p_make"] == pytest.approx(p_corner, abs=1e-12)
            assert res.table.loc[1, "p_make"] == pytest.approx(p_custom, abs=1e-12)


def test_predict_make_probability_default_behavior_unchanged(model) -> None:
    """New optional args preserve exact default outputs of predict_make_probability."""
    shots = [
        {k: v for k, v in SHOT_2PT.items() if k != "label"},
        {k: v for k, v in SHOT_3PT.items() if k != "label"},
    ]
    pid = int(model.player_table.iloc[0]["PLAYER_ID"])
    for p in (None, pid):
        np.testing.assert_array_equal(
            predict_make_probability(model, shots, player_id=p),
            predict_make_probability(model, shots, player_id=p, clutch_shift=True, shift_mode="overall", shift_bound="point"),
        )
        np.testing.assert_array_equal(
            predict_make_probability(model, shots, player_id=p, clutch_shift=False),
            predict_make_probability(model, shots, player_id=p, clutch_shift=True, shift_mode="none"),
        )


def test_invalid_inputs_raise_value_error(model, presets) -> None:
    """Invalid states, empty options, unknown presets, and unknown ACTION_GROUP raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": -1.0, "margin": 0}, ["corner_three"], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 3001.0, "margin": 0}, ["corner_three"], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": -1.5}, ["corner_three"], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": True}, ["corner_three"], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": 0, "period": 3}, ["corner_three"], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": 0}, [], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": 0}, ["nonexistent_zone"], model=model, presets=presets)
    bad_shot = dict(SHOT_2PT, ACTION_GROUP="skyhook_special")
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": 0}, [bad_shot], model=model, presets=presets)
    with pytest.raises(ValueError):
        evaluate_options({"time_left_sec": 10.0, "margin": 0}, ["corner_three"], shift_mode="bad", model=model, presets=presets)


def test_runs_in_copy_without_data_and_models(tmp_path: Path) -> None:
    """Decision layer imports and runs in an isolated copy containing only src/ and export/."""
    copy_root = tmp_path / "clean_repo"
    shutil.copytree(ROOT / "src", copy_root / "src")
    shutil.copytree(ROOT / "export", copy_root / "export")
    assert not (copy_root / "data").exists()
    assert not (copy_root / "models").exists()

    code = (
        "from src.decision_layer import evaluate_options; "
        "res = evaluate_options({'time_left_sec': 8.0, 'margin': -3}, ['above_the_break_3', 'restricted_area']); "
        "assert len(res.table) == 2; "
        "assert res.table.loc[0, 'rank_v'] == 1"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=copy_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
