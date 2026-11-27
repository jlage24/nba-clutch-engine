"""Headless Streamlit AppTest and wording/formatting tests for app/streamlit_app.py."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from app.formatting import (
    APP_DISCLAIMER,
    CUSTOM_SHOT_RARE_WARNING,
    MAP_TYPICAL_MIX_CAPTION,
    NO_SHOOTER_LABEL,
    SHOOTER_NOTE,
    build_shooter_options,
)
from src.decision_layer import GameState, evaluate_options, load_zone_presets
from src.heatmap import build_custom_shot_preset, load_court_defaults
from src.simulator_model import DEFAULT_EXPORT_DIR, load_export

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"
DEMO_PATH = ROOT / "docs" / "decision_demo.json"

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
    not (EXPORTS_PRESENT and APP_PATH.exists()),
    reason="export/ artifacts or app/streamlit_app.py not found.",
)


def _run_app(timeout: float = 20.0) -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"
    return at


def _collect_all_rendered_text(at: AppTest) -> str:
    parts: list[str] = []
    for group in (
        at.title,
        at.header,
        at.subheader,
        at.markdown,
        at.caption,
        at.info,
        at.warning,
        at.error,
        at.success,
        at.text,
    ):
        for el in group:
            val = getattr(el, "value", None)
            if val is not None:
                parts.append(str(val))
    for el in at.metric:
        parts.append(f"{el.label} {el.value}")
    for el in at.radio:
        parts.append(f"{el.label} {' '.join(map(str, el.options))}")
    for el in at.selectbox:
        parts.append(str(el.label))
    for el in at.slider:
        parts.append(str(el.label))
    for el in at.checkbox:
        parts.append(str(el.label))
    for el in at.dataframe:
        df = el.value
        parts.append(" ".join(map(str, df.columns)))
        if "Option" in df.columns:
            parts.append(" ".join(map(str, df["Option"].tolist())))
    return "\n".join(parts)


def _assert_no_forbidden_wording(text: str) -> None:
    lowered = text.lower()
    for banned in ("win probab", "pressure effect", "chok", "clutch gene"):
        assert banned not in lowered, f"Forbidden phrase {banned!r} found in text."
    for match in re.finditer(r"confidence interval", lowered):
        start = max(0, match.start() - 12)
        prefix = lowered[start : match.start()]
        assert "not a " in prefix, (
            f"Un-negated 'confidence interval' found in: {lowered[start:match.end() + 20]!r}"
        )


def test_1_defaults_load_and_match_demo_scenario_1() -> None:
    """App starts cleanly with defaults, 4 options, disclaimer, and Scenario 1 rank-1 option."""
    at = _run_app()
    df = at.dataframe[0].value
    assert len(df) == 4
    assert list(df.columns) == [
        "Option",
        "Make prob.",
        "xP",
        "Ahead",
        "Tied",
        "Behind",
        "V",
        "Rank",
    ]

    if DEMO_PATH.exists():
        demo = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
        sc1_rows = demo["primary_scenarios"][0]["rows"]
        expected_top = min(sc1_rows, key=lambda r: r["rank_v"])["label"]
        assert df.iloc[0]["Option"] == expected_top
    else:
        assert df.iloc[0]["Option"] == "Corner Three"

    assert any(APP_DISCLAIMER in c.value for c in at.caption)


def test_2_gate_closed_hides_v_and_shows_message() -> None:
    """At 40 s left, gate-closed message appears and no V values are shown."""
    at = _run_app()
    at.sidebar.slider(key="time_left_sec").set_value(40).run()
    assert not at.exception

    infos = " ".join(el.value for el in at.info)
    assert "Gate closed" in infos

    for df_el in at.dataframe:
        cols = [str(c) for c in df_el.value.columns]
        assert "V" not in cols
        assert not any(c.startswith("V ") for c in cols)


def test_3_margin_minus_5_10s_shows_no_option_changes_end_state() -> None:
    """At margin -5 and 10 s left, no-option-changes message appears."""
    at = _run_app()
    at.sidebar.slider(key="margin").set_value(-5)
    at.sidebar.slider(key="time_left_sec").set_value(10)
    at.run()
    assert not at.exception

    warnings = " ".join(el.value for el in at.warning)
    assert "No option changes the end state" in warnings


def test_4_shift_off_vs_overall_changes_probabilities_in_clutch() -> None:
    """Switching clutch shift between 'Off' and 'Overall (default)' changes make probabilities."""
    at = _run_app()
    df_overall = at.dataframe[0].value.copy()

    at.sidebar.radio(key="shift_mode_label").set_value("Off").run()
    assert not at.exception
    df_off = at.dataframe[0].value.copy()

    p_overall = [float(s.rstrip("%")) for s in df_overall.sort_values("Option")["Make prob."]]
    p_off = [float(s.rstrip("%")) for s in df_off.sort_values("Option")["Make prob."]]
    assert all(off > on for off, on in zip(p_off, p_overall))


def test_5_shooter_selection_changes_probs_and_shows_note() -> None:
    """Choosing a shooter alphabetically changes probabilities and displays the shrinkage note."""
    model = load_export(DEFAULT_EXPORT_DIR)
    labels, name_to_id = build_shooter_options(model.player_table)
    assert labels[0] == NO_SHOOTER_LABEL
    assert labels[1:] == sorted(labels[1:], key=lambda s: (s.casefold(), s))

    # Pick the player with the largest |offset| dynamically (never hardcode names).
    max_row = model.player_table.loc[model.player_table["offset"].abs().idxmax()]
    chosen_name = str(max_row["PLAYER_NAME"])
    assert chosen_name in name_to_id

    at = _run_app()
    assert at.sidebar.selectbox(key="shooter").options == labels
    df_none = at.dataframe[0].value.copy()

    at.sidebar.selectbox(key="shooter").set_value(chosen_name).run()
    assert not at.exception
    df_shooter = at.dataframe[0].value.copy()

    p_none = [float(s.rstrip("%")) for s in df_none.sort_values("Option")["Make prob."]]
    p_shooter = [float(s.rstrip("%")) for s in df_shooter.sort_values("Option")["Make prob."]]
    assert p_none != p_shooter

    rendered = _collect_all_rendered_text(at)
    assert SHOOTER_NOTE in rendered
    assert f"{float(max_row['offset']):.4f}" not in rendered


def test_6_custom_shot_smoothing_asymmetric_center_and_a4_crosscheck() -> None:
    """Custom shot smoothing differs off-centre, matches on LOC_X=0, and matches A4 presets."""
    model = load_export(DEFAULT_EXPORT_DIR)
    presets = load_zone_presets()
    court_defaults = load_court_defaults()

    at = _run_app()
    at.checkbox(key="use_custom_shot").check().run()
    at.number_input(key="custom_loc_x").set_value(15)
    at.number_input(key="custom_loc_y").set_value(15)
    at.run()
    assert not at.exception

    df_smooth = at.dataframe[0].value.copy()
    row_smooth = df_smooth[df_smooth["Option"].str.startswith("Custom")].iloc[0]

    at.sidebar.checkbox(key="smooth_lr").uncheck().run()
    assert not at.exception
    df_raw = at.dataframe[0].value.copy()
    row_raw = df_raw[df_raw["Option"].str.startswith("Custom")].iloc[0]

    assert row_smooth["Make prob."] != row_raw["Make prob."]

    # Amendment A4 cross-check against evaluate_options with weighted repeated rows.
    for is_smooth, disp_row in [(True, row_smooth), (False, row_raw)]:
        custom_preset = build_custom_shot_preset(
            15.0,
            15.0,
            shot_type="typical_mix",
            smooth=is_smooth,
            court_defaults=court_defaults,
            label="Custom (Typical mix)",
        )
        res = evaluate_options(
            state=GameState(8.0, -3, 4),
            options=["__custom__"],
            shift_mode="overall",
            model=model,
            presets={**presets, "__custom__": custom_preset},
        )
        expected_p = f"{float(res.table.iloc[0]['p_make']):.1%}"
        expected_xp = f"{float(res.table.iloc[0]['xp']):.2f}"
        assert disp_row["Make prob."] == expected_p
        assert disp_row["xP"] == expected_xp

    # On centre line (LOC_X = 0), smoothing on and off produce identical results.
    at.number_input(key="custom_loc_x").set_value(0).run()
    df_center_off = at.dataframe[0].value.copy()
    row_center_off = df_center_off[df_center_off["Option"].str.startswith("Custom")].iloc[0]

    at.sidebar.checkbox(key="smooth_lr").check().run()
    df_center_on = at.dataframe[0].value.copy()
    row_center_on = df_center_on[df_center_on["Option"].str.startswith("Custom")].iloc[0]

    assert row_center_on["Make prob."] == row_center_off["Make prob."]
    assert row_center_on["xP"] == row_center_off["xP"]
    assert row_center_on["V"] == row_center_off["V"]


def test_custom_shot_rare_distance_warning() -> None:
    """Rare shot type at a distance shows extrapolation caption; common shot type does not."""
    at = _run_app()
    at.checkbox(key="use_custom_shot").check().run()
    at.number_input(key="custom_loc_x").set_value(0)
    at.number_input(key="custom_loc_y").set_value(250)
    at.selectbox(key="custom_shot_type").set_value("dunk").run()
    assert not at.exception

    captions = " ".join(c.value for c in at.caption)
    assert CUSTOM_SHOT_RARE_WARNING in captions

    at.selectbox(key="custom_shot_type").set_value("jump_shot").run()
    assert not at.exception
    captions_after = " ".join(c.value for c in at.caption)
    assert CUSTOM_SHOT_RARE_WARNING not in captions_after


def test_7_map_tab_renders_each_metric_and_disables_v_when_gate_closed() -> None:
    """Map tab renders for all three metrics, shows captions, and disables V when gate closed."""
    at = _run_app()
    metric_radio = at.radio(key="map_metric")
    assert metric_radio.options == [
        "Make probability",
        "Expected points (xP)",
        "End-state value (V)",
    ]

    for metric_label in ("Make probability", "Expected points (xP)", "End-state value (V)"):
        at.radio(key="map_metric").set_value(metric_label).run()
        assert not at.exception

    captions = " ".join(c.value for c in at.caption)
    assert MAP_TYPICAL_MIX_CAPTION in captions
    assert "Within about 4 ft of the rim" in captions

    at.sidebar.slider(key="time_left_sec").set_value(40).run()
    assert not at.exception
    closed_radio = at.radio(key="map_metric")
    assert "End-state value (V)" not in closed_radio.options
    assert "disabled when the decision gate is closed" in " ".join(i.value for i in at.info)


def test_8_wording_scan_across_states_and_source_literals() -> None:
    """Every rendered string and every string literal in app/ obeys wording rules."""
    for file_path in (ROOT / "app" / "formatting.py", APP_PATH):
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        _assert_no_forbidden_wording("\n".join(literals))

    at = _run_app()
    _assert_no_forbidden_wording(_collect_all_rendered_text(at))

    at.sidebar.radio(key="shift_mode_label").set_value("By shot type (exploratory)")
    at.sidebar.slider(key="time_left_sec").set_value(40)
    at.run()
    _assert_no_forbidden_wording(_collect_all_rendered_text(at))

    at.sidebar.slider(key="time_left_sec").set_value(10)
    at.sidebar.slider(key="margin").set_value(-5)
    at.run()
    _assert_no_forbidden_wording(_collect_all_rendered_text(at))


def test_9_app_runs_in_clean_copy_without_data_and_models(tmp_path: Path) -> None:
    """AppTest boots and evaluates defaults cleanly without data/, models/, or docs/."""
    copy_root = tmp_path / "clean_app"
    shutil.copytree(ROOT / "app", copy_root / "app")
    shutil.copytree(ROOT / "src", copy_root / "src")
    shutil.copytree(ROOT / "export", copy_root / "export")
    shutil.copytree(ROOT / ".streamlit", copy_root / ".streamlit")
    assert not (copy_root / "data").exists()
    assert not (copy_root / "models").exists()

    code = (
        "import time; "
        "from streamlit.testing.v1 import AppTest; "
        "at = AppTest.from_file('app/streamlit_app.py', default_timeout=20); "
        "at.run(); "
        "assert not at.exception, at.exception; "
        "assert len(at.dataframe[0].value) == 4; "
        "assert at.dataframe[0].value.iloc[0]['Option'] == 'Corner Three'; "
        "t0 = time.perf_counter(); "
        "at.sidebar.slider(key='margin').set_value(-2).run(); "
        "dt = time.perf_counter() - t0; "
        "assert not at.exception, at.exception; "
        "print(f'clean_app_update_sec={dt:.4f}')"
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
    assert "clean_app_update_sec=" in proc.stdout
