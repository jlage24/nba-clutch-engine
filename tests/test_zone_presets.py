"""Tests for export/zone_presets.json and src/export_zone_presets.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.simulator_model import (
    DEFAULT_EXPORT_DIR,
    load_export,
    predict_make_probability,
)

ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = DEFAULT_EXPORT_DIR / "zone_presets.json"
ALL_SHOTS_PATH = ROOT / "data" / "processed" / "all_shots.parquet"
RAW_SHOTS_PRESENT = bool(sorted((ROOT / "data").glob("raw_shots_*.parquet")))

EXPECTED_PRESETS = {
    "restricted_area_layup",
    "putback",
    "paint",
    "short_mid_range",
    "long_mid_range",
    "corner_three",
    "above_the_break_3",
}
EXPECTED_SHOT_KEYS = {
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE_FT",
    "SHOT_ANGLE",
    "SHOT_VALUE",
    "ACTION_GROUP",
    "IS_PUTBACK",
}
FORBIDDEN_KEYS = {
    "SHOT_MADE_FLAG",
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "SCORE_MARGIN",
    "TIME_REMAINING_SEC",
    "PERIOD",
    "CLUTCH",
}

pytestmark = pytest.mark.skipif(
    not (PRESETS_PATH.exists() and (DEFAULT_EXPORT_DIR / "model_a.json").exists()),
    reason="export/zone_presets.json or export/model_a.json not found.",
)


def _load_presets() -> dict:
    return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))


def test_zone_presets_schema_and_no_outcomes_or_ids() -> None:
    """Presets contain only baseline features, no outcome or ID columns, and stay under 1 MB."""
    assert PRESETS_PATH.stat().st_size <= 1_000_000
    payload = _load_presets()
    assert payload["meta"]["no_outcomes_or_ids"] is True
    assert payload["meta"]["seed"] == 42

    presets = payload["presets"]
    assert set(presets.keys()) == EXPECTED_PRESETS

    for name, entry in presets.items():
        shots = entry["shots"]
        assert len(shots) == entry["n_sample"]
        assert entry["n_sample"] in {300, 1000}
        assert entry["n_zone_shots"] >= entry["n_sample"]

        expected_val = 3 if name in {"corner_three", "above_the_break_3"} else 2
        expected_pb = 1 if name == "putback" else 0

        for row in shots:
            keys = set(row.keys())
            assert keys == EXPECTED_SHOT_KEYS
            assert keys.isdisjoint(FORBIDDEN_KEYS)
            assert row["SHOT_VALUE"] == expected_val
            assert row["IS_PUTBACK"] == expected_pb

        for side_dict in (entry["side_shares_zone"], entry["side_shares_sample"]):
            assert pytest.approx(sum(side_dict.values()), abs=1e-4) == 1.0

        assert "abs_mirror_diff" in entry["mirror_diagnostic"]


def test_zone_presets_sample_means_match_model_and_targets() -> None:
    """Sample mean p_A matches freshly computed Model A predictions and zone targets."""
    payload = _load_presets()
    model = load_export(DEFAULT_EXPORT_DIR)

    for name, entry in payload["presets"].items():
        p_sample = predict_make_probability(model, entry["shots"], clutch_shift=False)
        mean_sample = float(np.mean(p_sample))
        assert mean_sample == pytest.approx(entry["mean_p_a_sample"], abs=1e-6)

        diff = abs(mean_sample - entry["mean_p_a_zone"])
        assert diff == pytest.approx(entry["abs_mean_p_a_diff"], abs=1e-6)
        if not entry["redrawn_to_1000"]:
            assert diff <= 0.003, f"{name} exceeded 0.003 with 300 rows"
        else:
            assert diff <= 0.005, f"{name} exceeded 0.005 even after 1,000-row redraw"


@pytest.mark.skipif(
    not (ALL_SHOTS_PATH.exists() and RAW_SHOTS_PRESENT),
    reason="data/processed/all_shots.parquet or raw_shots not found; skipping full-zone check.",
)
def test_zone_presets_match_regular_zone_statistics_on_data() -> None:
    """Stored zone counts, side shares, and zone mean p_A match REGULAR data on disk."""
    from src.export_zone_presets import BASELINE_FEATURES, _side_shares, _zone_masks, load_regular_shots_with_zones

    payload = _load_presets()
    model = load_export(DEFAULT_EXPORT_DIR)
    reg = load_regular_shots_with_zones()
    masks = _zone_masks(reg)

    for name, entry in payload["presets"].items():
        zone_df = reg[masks[name]]
        assert len(zone_df) == entry["n_zone_shots"]
        mean_zone = float(
            np.mean(predict_make_probability(model, zone_df[BASELINE_FEATURES], clutch_shift=False))
        )
        assert mean_zone == pytest.approx(entry["mean_p_a_zone"], abs=1e-6)
        shares = _side_shares(zone_df)
        for side in ("left", "right", "center"):
            assert shares[side] == pytest.approx(entry["side_shares_zone"][side], abs=1e-6)
