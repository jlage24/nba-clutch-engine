"""Step 4: Equivalence and unit tests for the portable simulator export."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.simulator_model import (
    DEFAULT_EXPORT_DIR,
    ExportedModel,
    load_export,
    predict_make_probability,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILES = [
    DEFAULT_EXPORT_DIR / "model_a.json",
    DEFAULT_EXPORT_DIR / "player_offsets.csv",
    DEFAULT_EXPORT_DIR / "clutch_shift.json",
    DEFAULT_EXPORT_DIR / "manifest.json",
]
EXPORT_PRESENT = all(p.exists() for p in EXPORT_FILES)
ALL_SHOTS_PATH = ROOT / "data" / "processed" / "all_shots.parquet"
ALL_SHOTS_PRESENT = ALL_SHOTS_PATH.exists()
RAW_SHOTS_PRESENT = bool(sorted((ROOT / "data").glob("raw_shots_*.parquet")))

pytestmark = pytest.mark.skipif(
    not EXPORT_PRESENT,
    reason="Portable export not found under export/. Run 'python src/export_model.py' first.",
)


@pytest.fixture(scope="module")
def exported_model() -> ExportedModel:
    return load_export(DEFAULT_EXPORT_DIR)


def test_load_export_exposes_shift_and_schema(exported_model: ExportedModel) -> None:
    """load_export returns ExportedModel with shifts, CIs, offsets, and matching feature names."""
    assert len(exported_model.player_table) == 572
    assert exported_model.manifest["player_offsets"]["n_players_absent_zero_regular_shots"] == 10
    assert exported_model.model_a.get_booster().feature_names == exported_model.features

    lo_a, hi_a = exported_model.clutch_shift_a_ci
    lo_b, hi_b = exported_model.clutch_shift_b_ci
    assert lo_a < exported_model.clutch_shift_a < hi_a < 0.0
    assert lo_b < exported_model.clutch_shift_b < hi_b < 0.0

    diff_entry = exported_model.clutch_shift_info["exploratory"]["model_b_3pt_minus_2pt"]
    assert diff_entry["ci_lower_logit"] < diff_entry["diff_logit"] < diff_entry["ci_upper_logit"]
    assert "no detectable difference between 2-point and 3-point shifts at this sample size" in (
        exported_model.clutch_shift_info["exploratory"]["warning"]
    )
    assert "no detectable difference between 2-point and 3-point shifts at this sample size" in (
        exported_model.manifest["prediction_formula"]["shift_scope_note"]
    )


def test_load_export_rejects_feature_mismatch(tmp_path: Path) -> None:
    """load_export raises ValueError if model_a.json feature order disagrees with manifest.json."""
    export_copy = tmp_path / "export"
    shutil.copytree(DEFAULT_EXPORT_DIR, export_copy)

    manifest_path = export_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["features"] = list(reversed(manifest["features"]))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Feature order mismatch"):
        load_export(export_copy)


def test_action_group_categorical_and_unknown_raises(exported_model: ExportedModel) -> None:
    """ACTION_GROUP uses exact manifest categories; unknown values raise ValueError."""
    shot = {
        "LOC_X": 0.0,
        "LOC_Y": 250.0,
        "SHOT_VALUE": 3,
        "ACTION_GROUP": "jump_shot",
        "IS_PUTBACK": 0,
    }
    X = prepare_features([shot], exported_model)
    assert isinstance(X["ACTION_GROUP"].dtype, pd.CategoricalDtype)
    assert list(X["ACTION_GROUP"].cat.categories) == exported_model.action_group_categories

    bad_shot = dict(shot, ACTION_GROUP="skyhook_special")
    with pytest.raises(ValueError, match="Unknown ACTION_GROUP"):
        prepare_features([bad_shot], exported_model)

    nan_shot = dict(shot, ACTION_GROUP=None)
    with pytest.raises(ValueError, match="Unknown ACTION_GROUP"):
        prepare_features([nan_shot], exported_model)


def test_dataframe_and_list_of_dicts_agree(exported_model: ExportedModel) -> None:
    """predict_make_probability accepts both DataFrame and list of dicts with identical output."""
    shots_list = [
        {
            "LOC_X": 0.0,
            "LOC_Y": 15.0,
            "ACTION_TYPE": "Driving Layup Shot",
            "SHOT_TYPE": "2PT Field Goal",
        },
        {
            "LOC_X": 220.0,
            "LOC_Y": 20.0,
            "ACTION_TYPE": "Jump Shot",
            "SHOT_TYPE": "3PT Field Goal",
        },
    ]
    shots_df = pd.DataFrame(shots_list)
    p_list = predict_make_probability(exported_model, shots_list, clutch_shift=True)
    p_df = predict_make_probability(exported_model, shots_df, clutch_shift=True)
    assert isinstance(p_list, np.ndarray)
    assert isinstance(p_df, np.ndarray)
    assert p_list.shape == (2,)
    np.testing.assert_allclose(p_list, p_df, atol=1e-12)


def test_unknown_player_gives_p_b_equals_p_a(exported_model: ExportedModel) -> None:
    """An unknown player (and players absent from player_offsets.csv) get offset 0 so p_B == p_A."""
    shots = [
        {"LOC_X": 0.0, "LOC_Y": 245.0, "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": 10.0, "LOC_Y": 20.0, "ACTION_TYPE": "Driving Layup Shot", "SHOT_TYPE": "2PT Field Goal"},
    ]
    p_a = predict_make_probability(exported_model, shots, player_id=None, clutch_shift=False)
    p_unknown = predict_make_probability(
        exported_model, shots, player_id="unknown_9999999", clutch_shift=False
    )
    np.testing.assert_array_equal(p_unknown, p_a)


def test_known_player_moves_probability_by_offset(exported_model: ExportedModel) -> None:
    """A known player shifts p_A by exact offset (before clipping to [0.01, 0.99])."""
    top_row = exported_model.player_table.sort_values("n_regular_shots", ascending=False).iloc[0]
    pid = int(top_row["PLAYER_ID"])
    offset = float(top_row["offset"])
    assert offset != 0.0

    shot = [{"LOC_X": 0.0, "LOC_Y": 160.0, "ACTION_TYPE": "Pullup Jump Shot", "SHOT_TYPE": "2PT Field Goal"}]
    p_a = predict_make_probability(exported_model, shot, player_id=None, clutch_shift=False)
    p_b = predict_make_probability(exported_model, shot, player_id=pid, clutch_shift=False)
    expected = np.clip(p_a + offset, 0.01, 0.99)
    np.testing.assert_allclose(p_b, expected, atol=1e-12)


def test_clutch_shift_toggle_and_direction(exported_model: ExportedModel) -> None:
    """clutch_shift=False gives baseline; clutch_shift=True lowers probabilities for negative shift."""
    shots = [
        {"LOC_X": 0.0, "LOC_Y": 10.0, "ACTION_TYPE": "Driving Dunk Shot", "SHOT_TYPE": "2PT Field Goal"},
        {"LOC_X": 0.0, "LOC_Y": 250.0, "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "3PT Field Goal"},
    ]
    pid = int(exported_model.player_table.iloc[0]["PLAYER_ID"])

    p_a_base = predict_make_probability(exported_model, shots, player_id=None, clutch_shift=False)
    p_a_shift = predict_make_probability(exported_model, shots, player_id=None, clutch_shift=True)
    assert np.all(p_a_shift < p_a_base)

    p_b_base = predict_make_probability(exported_model, shots, player_id=pid, clutch_shift=False)
    p_b_shift = predict_make_probability(exported_model, shots, player_id=pid, clutch_shift=True)
    assert np.all(p_b_shift < p_b_base)

    # Exact logit formula check
    eps = 1e-6
    p_clip = np.clip(p_b_base, eps, 1.0 - eps)
    expected_logit = np.log(p_clip / (1.0 - p_clip)) + exported_model.clutch_shift_b
    expected = 1.0 / (1.0 + np.exp(-expected_logit))
    np.testing.assert_allclose(p_b_shift, expected, atol=1e-12)


def test_origin_clamp_non_putback_vs_putback(exported_model: ExportedModel) -> None:
    """Coordinates at (0, 0) without IS_PUTBACK are clamped to 1 ft; putbacks are not clamped."""
    non_putback_zero = {
        "LOC_X": 0.0,
        "LOC_Y": 0.0,
        "ACTION_TYPE": "Driving Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
    }
    non_putback_one = {
        "LOC_X": 0.0,
        "LOC_Y": 0.0,
        "SHOT_DISTANCE_FT": 1.0,
        "SHOT_ANGLE": 0.0,
        "ACTION_TYPE": "Driving Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
    }
    X_zero = prepare_features([non_putback_zero], exported_model, clamp_distance=True)
    assert X_zero["IS_PUTBACK"].iloc[0] == 0
    assert X_zero["SHOT_DISTANCE_FT"].iloc[0] == 1.0

    p_zero = predict_make_probability(exported_model, [non_putback_zero], clutch_shift=False)
    p_one = predict_make_probability(exported_model, [non_putback_one], clutch_shift=False)
    np.testing.assert_array_equal(p_zero, p_one)

    putback_zero = {
        "LOC_X": 0.0,
        "LOC_Y": 0.0,
        "ACTION_TYPE": "Putback Layup Shot",
        "SHOT_TYPE": "2PT Field Goal",
    }
    X_putback = prepare_features([putback_zero], exported_model, clamp_distance=True)
    assert X_putback["IS_PUTBACK"].iloc[0] == 1
    assert X_putback["SHOT_DISTANCE_FT"].iloc[0] == 0.0


def test_outputs_always_strictly_in_unit_interval(exported_model: ExportedModel) -> None:
    """Output probabilities are always strictly inside (0, 1)."""
    shots = [
        {"LOC_X": 0.0, "LOC_Y": 5.0, "ACTION_TYPE": "Alley Oop Dunk Shot", "SHOT_TYPE": "2PT Field Goal"},
        {"LOC_X": 0.0, "LOC_Y": 450.0, "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": -220.0, "LOC_Y": 10.0, "ACTION_TYPE": "Step Back Jump shot", "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": 50.0, "LOC_Y": 120.0, "ACTION_TYPE": "Floating Jump Shot", "SHOT_TYPE": "2PT Field Goal"},
    ]
    sample_pids = [None, "unknown_id"] + [
        int(pid) for pid in exported_model.player_table["PLAYER_ID"].iloc[:3]
    ]
    for pid in sample_pids:
        for shift in (False, True):
            probs = predict_make_probability(exported_model, shots, player_id=pid, clutch_shift=shift)
            assert np.all(np.isfinite(probs))
            assert np.all((probs > 0.0) & (probs < 1.0))


@pytest.mark.skipif(
    not ALL_SHOTS_PRESENT,
    reason="data/processed/all_shots.parquet not found; skipping full-dataset Model A equivalence test.",
)
def test_exported_model_a_reproduces_in_memory_model_on_all_shots(
    exported_model: ExportedModel,
) -> None:
    """Exported Model A must match in-memory final Model A within 1e-6 on all 219,160 shots."""
    from src.baseline_model import BASELINE_FEATURES, RNG_SEED, _X
    from src.export_model import fit_final_model_a, load_phase3b_hyperparameters
    from src.features import ACTION_DTYPE

    all_shots = pd.read_parquet(ALL_SHOTS_PATH)
    all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(str)
    all_shots["ACTION_GROUP"] = pd.Categorical(all_shots["ACTION_GROUP"], dtype=ACTION_DTYPE)
    regular = all_shots[all_shots["REGULAR"]].copy()

    best_cfg, n_estimators, _ = load_phase3b_hyperparameters()
    in_memory_model = fit_final_model_a(regular, best_cfg, n_estimators, seed=RNG_SEED)

    X_mem = _X(all_shots)
    X_exp = prepare_features(all_shots, exported_model, clamp_distance=False)
    assert list(X_exp.columns) == BASELINE_FEATURES

    p_mem = in_memory_model.predict_proba(X_mem)[:, 1]
    p_exp = exported_model.model_a.predict_proba(X_exp)[:, 1]
    max_diff = float(np.max(np.abs(p_mem - p_exp)))
    assert max_diff < 1e-6, f"Max absolute prediction diff {max_diff} >= 1e-6"


@pytest.mark.skipif(
    not ALL_SHOTS_PRESENT,
    reason="data/processed/all_shots.parquet not found; skipping build_features equivalence test.",
)
def test_loader_matches_build_features_on_5000_all_shots(exported_model: ExportedModel) -> None:
    """prepare_features matches build_features on 5,000 sampled shots from all_shots.parquet."""
    from src.baseline_model import BASELINE_FEATURES
    from src.features import build_features

    all_shots = pd.read_parquet(ALL_SHOTS_PATH)
    sample = all_shots.sample(n=5000, random_state=42).reset_index(drop=True)

    ref = build_features(sample)[BASELINE_FEATURES]
    got = prepare_features(sample, exported_model, clamp_distance=False)
    pd.testing.assert_frame_equal(got, ref, check_dtype=True)


@pytest.mark.skipif(
    not RAW_SHOTS_PRESENT,
    reason="data/raw_shots_*.parquet not found; skipping raw build_features equivalence test.",
)
def test_loader_matches_build_features_on_5000_raw_shots(exported_model: ExportedModel) -> None:
    """prepare_features matches build_features from raw columns on 5,000 sampled shots."""
    from src.baseline_model import BASELINE_FEATURES
    from src.features import build_features

    raw_frames = [pd.read_parquet(p) for p in sorted((ROOT / "data").glob("raw_shots_*.parquet"))]
    raw_all = pd.concat(raw_frames, ignore_index=True)
    sample = raw_all.sample(n=5000, random_state=42).reset_index(drop=True)
    # build_features requires SCORE_MARGIN even though Model A baseline features omit it
    sample["SCORE_MARGIN"] = 0.0

    ref = build_features(sample)[BASELINE_FEATURES]
    got = prepare_features(sample, exported_model, clamp_distance=False)
    pd.testing.assert_frame_equal(got, ref, check_dtype=True)
