"""Unit, cross-check, and clean-copy runtime tests for src/heatmap.py."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.decision_layer import GameState, evaluate_options
from src.heatmap import (
    TYPICAL_MIX_KEY,
    build_custom_shot_preset,
    compute_heatmap,
    get_active_shot_types,
    load_court_defaults,
    lookup_cell,
    render_court_heatmap,
)
from src.shot_geometry import compute_shot_value
from src.simulator_model import DEFAULT_EXPORT_DIR, ExportedModel, load_export

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXPORTS = [
    DEFAULT_EXPORT_DIR / "model_a.json",
    DEFAULT_EXPORT_DIR / "player_offsets.csv",
    DEFAULT_EXPORT_DIR / "clutch_shift.json",
    DEFAULT_EXPORT_DIR / "manifest.json",
    DEFAULT_EXPORT_DIR / "zone_presets.json",
    DEFAULT_EXPORT_DIR / "court_defaults.json",
]
EXPORTS_PRESENT = all(p.exists() for p in REQUIRED_EXPORTS)

pytestmark = pytest.mark.skipif(
    not EXPORTS_PRESENT,
    reason="export/ artifacts (including court_defaults.json) not found.",
)


@pytest.fixture(scope="module")
def model() -> ExportedModel:
    return load_export(DEFAULT_EXPORT_DIR)


@pytest.fixture(scope="module")
def court_defaults() -> dict:
    return load_court_defaults(DEFAULT_EXPORT_DIR / "court_defaults.json")


def test_smoothing_symmetry_and_unsmoothed_asymmetry(model, court_defaults) -> None:
    """With smoothing on, every map is exactly left-right symmetric; with it off, Model A differs."""
    state = GameState(time_left_sec=8.0, margin=-3, period=4)
    hm_smooth = compute_heatmap(state, smooth=True, model=model, court_defaults=court_defaults)
    hm_raw = compute_heatmap(state, smooth=False, model=model, court_defaults=court_defaults)

    np.testing.assert_array_equal(hm_smooth.mask, hm_smooth.mask[::-1, :])
    for arr in (hm_smooth.p_make, hm_smooth.xp, hm_smooth.v):
        np.testing.assert_array_equal(np.isnan(arr), np.isnan(arr[::-1, :]))
        valid = np.isfinite(arr)
        np.testing.assert_array_equal(arr[valid], arr[::-1, :][valid])

    both_valid = np.isfinite(hm_raw.p_make) & np.isfinite(hm_raw.p_make[::-1, :])
    max_asym = float(np.max(np.abs(hm_raw.p_make[both_valid] - hm_raw.p_make[::-1, :][both_valid])))
    assert max_asym > 1e-3


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("shot_type", [TYPICAL_MIX_KEY, "jump_shot"])
def test_crosscheck_against_decision_layer_within_1e9(
    model, court_defaults, smooth: bool, shot_type: str
) -> None:
    """Map values match evaluate_options with repeated-weight custom presets within 1e-9."""
    state = GameState(time_left_sec=8.0, margin=-2, period=4)
    hm = compute_heatmap(
        state,
        shift_mode="overall",
        shot_type=shot_type,
        smooth=smooth,
        tie_value=0.5,
        model=model,
        court_defaults=court_defaults,
    )

    # At least 12 active cells spanning multiple zones and both sides of the court.
    candidate_coords = [
        (-5.0, 5.0),
        (5.0, 5.0),
        (-35.0, 55.0),
        (35.0, 55.0),
        (-95.0, 115.0),
        (95.0, 115.0),
        (-145.0, 145.0),
        (145.0, 145.0),
        (-235.0, 25.0),
        (235.0, 25.0),
        (-135.0, 225.0),
        (135.0, 225.0),
        (-5.0, 255.0),
        (5.0, 255.0),
    ]
    checked = 0
    presets: dict[str, dict] = {}
    option_keys: list[str] = []
    expected_cells: list[dict] = []
    for idx, (cx, cy) in enumerate(candidate_coords):
        cell = lookup_cell(hm, cx, cy)
        if not cell["in_mask"]:
            continue
        key = f"cell_{idx}"
        presets[key] = build_custom_shot_preset(
            cell["cell_x"],
            cell["cell_y"],
            shot_type=shot_type,
            smooth=smooth,
            court_defaults=court_defaults,
            label=key,
        )
        option_keys.append(key)
        expected_cells.append(cell)
        checked += 1

    assert checked >= 10
    res = evaluate_options(
        state=state,
        options=option_keys,
        shift_mode="overall",
        tie_value=0.5,
        model=model,
        presets=presets,
    )
    by_label = res.table.set_index("label")
    for key, cell in zip(option_keys, expected_cells):
        row = by_label.loc[key]
        assert cell["p_make"] == pytest.approx(float(row["p_make"]), abs=1e-9)
        assert cell["xp"] == pytest.approx(float(row["xp"]), abs=1e-9)
        assert cell["v"] == pytest.approx(float(row["v"]), abs=1e-9)


def test_mask_bounds_and_shot_type_filtering(model, court_defaults) -> None:
    """Masked cells are NaN, active cells are bounded, and group masking drops infeasible spots."""
    active_types = get_active_shot_types(court_defaults)
    assert len(active_types) == 8
    assert "other" not in active_types

    with pytest.raises(ValueError, match="Unknown shot_type"):
        compute_heatmap(
            GameState(8.0, -3),
            shot_type="other",
            model=model,
            court_defaults=court_defaults,
        )

    hm = compute_heatmap(
        GameState(8.0, -3),
        shot_type=TYPICAL_MIX_KEY,
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    outside = ~hm.mask
    assert np.all(np.isnan(hm.p_make[outside]))
    assert np.all(np.isnan(hm.xp[outside]))
    assert np.all(np.isnan(hm.v[outside]))

    inside = hm.mask
    assert np.all((hm.p_make[inside] > 0.0) & (hm.p_make[inside] < 1.0))
    assert np.all((hm.xp[inside] > 0.0) & (hm.xp[inside] < 3.0))
    assert np.all((hm.v[inside] >= 0.0) & (hm.v[inside] <= 1.0))

    # Amendment A3: dunks have weight < 10 at 25 ft, so 25 ft cells are masked for shot_type='dunk'.
    hm_dunk = compute_heatmap(
        GameState(8.0, -3),
        shot_type="dunk",
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    deep_cell = lookup_cell(hm_dunk, 5.0, 255.0)
    rim_cell = lookup_cell(hm_dunk, 5.0, 5.0)
    assert not deep_cell["in_mask"]
    assert np.isnan(deep_cell["p_make"])
    assert rim_cell["in_mask"]
    assert 0.5 < rim_cell["p_make"] < 1.0


def test_shot_value_matches_three_point_line(model, court_defaults) -> None:
    """Grid shot_value matches compute_shot_value across 2PT and 3PT regions."""
    hm = compute_heatmap(
        GameState(8.0, -3),
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    xx, yy = np.meshgrid(hm.x_centers, hm.y_centers, indexing="ij")
    expected_sv = compute_shot_value(xx, yy)
    np.testing.assert_array_equal(hm.shot_value, expected_sv)

    assert lookup_cell(hm, 225.0, 5.0)["shot_value"] == 3
    assert lookup_cell(hm, -225.0, 5.0)["shot_value"] == 3
    assert lookup_cell(hm, 215.0, 5.0)["shot_value"] == 2
    assert lookup_cell(hm, 5.0, 245.0)["shot_value"] == 3
    assert lookup_cell(hm, 5.0, 225.0)["shot_value"] == 2


def test_v_unavailable_when_gate_closed(model, court_defaults) -> None:
    """When time_left_sec > time_gate_sec, V is all NaN and plotting V raises ValueError."""
    hm_closed = compute_heatmap(
        GameState(time_left_sec=40.0, margin=-3, period=4),
        time_gate_sec=24.0,
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    assert not hm_closed.gate_open
    assert np.all(np.isnan(hm_closed.v))
    assert np.all(np.isfinite(hm_closed.p_make[hm_closed.mask]))
    assert np.all(np.isfinite(hm_closed.xp[hm_closed.mask]))

    with pytest.raises(ValueError, match="unavailable"):
        render_court_heatmap(hm_closed, metric="v")


def test_shift_inside_and_outside_clutch(model, court_defaults) -> None:
    """Shift 'none' > 'overall' inside clutch, and identical outside clutch."""
    clutch_state = GameState(time_left_sec=8.0, margin=-3, period=4)
    hm_off = compute_heatmap(
        clutch_state, shift_mode="none", smooth=True, model=model, court_defaults=court_defaults
    )
    hm_on = compute_heatmap(
        clutch_state, shift_mode="overall", smooth=True, model=model, court_defaults=court_defaults
    )
    assert np.all(hm_off.p_make[hm_off.mask] > hm_on.p_make[hm_on.mask])

    non_clutch_state = GameState(time_left_sec=8.0, margin=-8, period=4)
    hm_nc_off = compute_heatmap(
        non_clutch_state, shift_mode="none", smooth=True, model=model, court_defaults=court_defaults
    )
    hm_nc_on = compute_heatmap(
        non_clutch_state, shift_mode="overall", smooth=True, model=model, court_defaults=court_defaults
    )
    np.testing.assert_allclose(hm_nc_off.p_make, hm_nc_on.p_make, equal_nan=True, atol=1e-12)


def test_shooter_changes_map_in_offset_direction(model, court_defaults) -> None:
    """Positive/negative shooter offsets shift p_make in the direction of the offset."""
    eligible = model.player_table[model.player_table["n_regular_shots"] >= 200]
    pos_pid = int(eligible.sort_values("offset", ascending=False).iloc[0]["PLAYER_ID"])
    neg_pid = int(eligible.sort_values("offset", ascending=True).iloc[0]["PLAYER_ID"])
    assert model.player_offsets[str(pos_pid)] > 0.03
    assert model.player_offsets[str(neg_pid)] < -0.03

    state = GameState(time_left_sec=8.0, margin=-3, period=4)
    hm_none_noshift = compute_heatmap(
        state, player_id=None, shift_mode="none", model=model, court_defaults=court_defaults
    )
    hm_pos_noshift = compute_heatmap(
        state, player_id=pos_pid, shift_mode="none", model=model, court_defaults=court_defaults
    )
    hm_neg_noshift = compute_heatmap(
        state, player_id=neg_pid, shift_mode="none", model=model, court_defaults=court_defaults
    )
    mask = hm_none_noshift.mask
    assert np.all(hm_pos_noshift.p_make[mask] > hm_none_noshift.p_make[mask])
    assert np.all(hm_none_noshift.p_make[mask] > hm_neg_noshift.p_make[mask])

    hm_pos_overall = compute_heatmap(
        state, player_id=pos_pid, shift_mode="overall", model=model, court_defaults=court_defaults
    )
    hm_neg_overall = compute_heatmap(
        state, player_id=neg_pid, shift_mode="overall", model=model, court_defaults=court_defaults
    )
    assert np.all(hm_pos_overall.p_make[mask] > hm_neg_overall.p_make[mask])


def test_render_court_heatmap_figures(model, court_defaults) -> None:
    """Matplotlib figure rendering succeeds for p_make, xp, and v."""
    hm = compute_heatmap(
        GameState(8.0, -3),
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    for metric in ("p_make", "xp", "v"):
        fig = render_court_heatmap(hm, metric=metric, highlight_xy=(225.0, 5.0))
        assert len(fig.axes) >= 2
        plt.close(fig)


def test_heatmap_runs_and_fast_in_clean_copy(tmp_path: Path) -> None:
    """Heatmap computes in < 2s inside an isolated copy with only src/ and export/."""
    copy_root = tmp_path / "clean_repo"
    shutil.copytree(ROOT / "src", copy_root / "src")
    shutil.copytree(ROOT / "export", copy_root / "export")
    assert not (copy_root / "data").exists()
    assert not (copy_root / "models").exists()

    code = (
        "import time; "
        "from src.simulator_model import load_export; "
        "from src.heatmap import load_court_defaults, compute_heatmap; "
        "m = load_export(); d = load_court_defaults(); "
        "t0 = time.perf_counter(); "
        "hm = compute_heatmap({'time_left_sec': 8.0, 'margin': -3}, model=m, court_defaults=d); "
        "dt = time.perf_counter() - t0; "
        "assert hm.gate_open is True; "
        "assert dt < 2.0, f'Too slow: {dt:.3f}s'; "
        "print(f'clean_copy_runtime_sec={dt:.4f}')"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=copy_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed_total = time.perf_counter() - t0
    assert proc.returncode == 0
    assert "clean_copy_runtime_sec=" in proc.stdout
    assert elapsed_total < 10.0
