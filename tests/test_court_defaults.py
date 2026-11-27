"""Tests for export/court_defaults.json and src/export_court_defaults.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.export_court_defaults import assign_2ft_grid_indices, largest_remainder_weights
from src.simulator_model import DEFAULT_EXPORT_DIR

ROOT = Path(__file__).resolve().parents[1]
COURT_DEFAULTS_PATH = DEFAULT_EXPORT_DIR / "court_defaults.json"
MANIFEST_PATH = DEFAULT_EXPORT_DIR / "manifest.json"
ALL_SHOTS_PATH = ROOT / "data" / "processed" / "all_shots.parquet"

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
    not (COURT_DEFAULTS_PATH.exists() and MANIFEST_PATH.exists()),
    reason="export/court_defaults.json or export/manifest.json not found.",
)


def _load_defaults() -> dict[str, Any]:
    return json.loads(COURT_DEFAULTS_PATH.read_text(encoding="utf-8"))


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _collect_all_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_all_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys(item))
    return keys


def test_court_defaults_schema_size_and_no_outcomes_or_ids() -> None:
    """Exported JSON is under 1 MB, contains no outcome/ID keys, and uses manifest categories."""
    assert COURT_DEFAULTS_PATH.stat().st_size <= 1_000_000
    payload = _load_defaults()
    manifest = _load_manifest()
    categories = list(manifest["action_group_categories"])

    assert payload["meta"]["no_outcomes_or_ids"] is True
    assert payload["meta"]["seed"] == 42
    assert payload["meta"]["action_group_categories"] == categories
    active = payload["meta"]["active_action_groups"]
    assert len(active) == 8
    assert "other" not in active
    assert set(active).issubset(set(categories))
    assert _collect_all_keys(payload).isdisjoint(FORBIDDEN_KEYS)

    mix = payload["action_group_mix"]
    assert mix["categories"] == categories

    pool = mix["pooled_ge_22ft"]
    assert set(pool["group_counts"].keys()) == set(categories)
    assert set(pool["weights_per_1000"].keys()) == set(categories)
    assert sum(pool["group_counts"].values()) == pool["n_shots"]
    assert sum(pool["weights_per_1000"].values()) == 1_000
    assert all(isinstance(w, int) and w >= 0 for w in pool["weights_per_1000"].values())

    for dist_key, entry in mix["by_distance_ft"].items():
        assert int(dist_key) == entry["distance_ft"]
        assert set(entry["group_counts"].keys()) == set(categories)
        assert set(entry["weights_per_1000"].keys()) == set(categories)
        assert sum(entry["group_counts"].values()) == entry["n_shots"]
        weights = entry["weights_per_1000"]
        assert all(isinstance(w, int) and w >= 0 for w in weights.values())
        assert sum(weights.values()) == 1_000
        if entry["n_shots"] >= 100:
            assert entry["source"] == "per_foot"
        else:
            assert entry["source"] == "pooled_ge_22ft"
            assert weights == pool["weights_per_1000"]

    modal = payload["checks"]["modal_comparison"]["modal_by_distance_ft"]
    allowed = set(categories)
    assert all(group in allowed for group in modal.values())


def test_density_mask_schema_coverage_and_checks() -> None:
    """Density mask uses 25x24 2 ft cells, >= 99% coverage, and well-formed check sections."""
    payload = _load_defaults()
    dm = payload["density_mask"]

    assert dm["nx"] == 25
    assert dm["ny"] == 24
    assert dm["n_cells_total"] == 600
    assert dm["x_edges_tenths"] == [float(v) for v in range(-250, 251, 20)]
    assert dm["y_edges_tenths"] == [float(v) for v in range(-50, 431, 20)]
    assert dm["min_shots_threshold"] == 10

    counts = np.asarray(dm["counts"], dtype=int)
    mask = np.asarray(dm["mask"], dtype=bool)
    assert counts.shape == (25, 24)
    assert mask.shape == (25, 24)
    np.testing.assert_array_equal(mask, counts >= 10)

    assert dm["n_cells_in_mask"] == int(mask.sum())
    assert dm["n_regular_shots_in_mask"] == int(counts[mask].sum())
    assert dm["share_regular_shots_in_mask"] >= 0.99

    pc = payload["checks"]["cell_level_typical_mix_vs_actual"]
    assert pc["subset"] == "REGULAR & IS_PUTBACK == 0"
    assert pc["weighted_mean_abs_diff"] <= 0.01
    assert pc["n_cells_ge_threshold"] == len(pc["cells_ge_threshold"])
    assert all(item["abs_diff"] >= pc["diff_threshold"] for item in pc["cells_ge_threshold"])
    assert pc["max_diff_cell"]["abs_diff"] == pc["max_abs_diff"]
    assert pc["max_diff_cell"]["n_shots"] >= pc["min_cell_shots"]

    cd = payload["checks"]["cell_centre_diagnostic"]
    assert "geometry inside the cell" in cd["note"]
    assert cd["n_cells_ge_threshold"] == len(cd["cells_ge_threshold"])
    assert all(item["abs_diff"] >= cd["diff_threshold"] for item in cd["cells_ge_threshold"])


def test_largest_remainder_and_grid_assignment_helpers() -> None:
    """Pure rounding and grid-indexing helpers behave deterministically at boundaries."""
    cats = ["a", "b", "c"]
    weights = largest_remainder_weights({"a": 1, "b": 1, "c": 1}, cats, scale=1_000)
    assert sum(weights.values()) == 1_000
    assert sorted(weights.values()) == [333, 333, 334]

    x_edges = np.arange(-250.0, 251.0, 20.0)
    y_edges = np.arange(-50.0, 431.0, 20.0)
    ix, iy, in_box = assign_2ft_grid_indices(
        np.array([-250.0, 0.0, 250.0, 0.0]),
        np.array([-50.0, 0.0, 430.0, 500.0]),
        x_edges,
        y_edges,
    )
    assert list(in_box) == [True, True, True, False]
    assert list(ix[:3]) == [0, 12, 24]
    assert list(iy[:3]) == [0, 2, 23]


@pytest.mark.skipif(
    not ALL_SHOTS_PATH.exists(),
    reason="data/processed/all_shots.parquet not found; skipping full-data court defaults check.",
)
def test_court_defaults_match_regular_data_on_disk() -> None:
    """Stored counts, mix weights, mask counts, and cell checks match REGULAR data on disk."""
    from src.export_court_defaults import build_court_defaults, load_regular_shots

    payload = _load_defaults()
    rebuilt = build_court_defaults()
    reg = load_regular_shots()
    non_pb = reg[reg["IS_PUTBACK"] == 0]

    assert payload["meta"]["n_regular_shots"] == len(reg)
    assert payload["meta"]["n_regular_non_putback_shots"] == len(non_pb)
    assert payload["meta"]["n_regular_putback_shots"] == len(reg) - len(non_pb)

    assert payload["action_group_mix"] == rebuilt["action_group_mix"]
    assert payload["density_mask"]["counts"] == rebuilt["density_mask"]["counts"]
    assert payload["density_mask"]["mask"] == rebuilt["density_mask"]["mask"]
    assert (
        payload["density_mask"]["share_regular_shots_in_mask"]
        == rebuilt["density_mask"]["share_regular_shots_in_mask"]
    )

    for key in ("cell_level_typical_mix_vs_actual", "cell_centre_diagnostic"):
        disk = payload["checks"][key]
        new = rebuilt["checks"][key]
        assert disk["n_cells_checked"] == new["n_cells_checked"]
        assert disk["n_shots_covered"] == new["n_shots_covered"]
        assert disk["weighted_mean_abs_diff"] == pytest.approx(
            new["weighted_mean_abs_diff"], abs=1e-6
        )
        assert disk["max_abs_diff"] == pytest.approx(new["max_abs_diff"], abs=1e-6)
        assert disk["n_cells_ge_threshold"] == new["n_cells_ge_threshold"]

    mc_disk = payload["checks"]["modal_comparison"]
    mc_new = rebuilt["checks"]["modal_comparison"]
    assert mc_disk["agreement_share_all_regular"] == pytest.approx(
        mc_new["agreement_share_all_regular"], abs=1e-6
    )
    assert mc_disk["sample_mean_abs_p_a_diff"] == pytest.approx(
        mc_new["sample_mean_abs_p_a_diff"], abs=1e-6
    )
    assert mc_disk["sample_max_abs_p_a_diff"] == pytest.approx(
        mc_new["sample_max_abs_p_a_diff"], abs=1e-6
    )
