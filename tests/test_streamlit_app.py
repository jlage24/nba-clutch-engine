"""Headless Streamlit AppTest, layout, wording, and contrast tests for app/streamlit_app.py."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.formatting import (
    APP_DISCLAIMER,
    CUSTOM_SHOT_RARE_WARNING,
    END_STATE_CAPTION,
    MAP_MODEL_STEPS_CAPTION,
    MAP_TYPICAL_MIX_CAPTION,
    NO_SHOOTER_LABEL,
    SHOOTER_NOTE,
    build_shooter_options,
    classify_summary_variant,
    format_decision_summary,
)
from src.decision_layer import DecisionResult, GameState, evaluate_options, load_zone_presets
from src.heatmap import build_custom_shot_preset, load_court_defaults
from src.simulator_model import DEFAULT_EXPORT_DIR, load_export

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"
LOGO_PATH = ROOT / "app" / "assets" / "nba-clutch.png"
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

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
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
    for el in at.segmented_control:
        parts.append(f"{el.label} {' '.join(map(str, el.options))}")
    for el in at.selectbox:
        parts.append(str(el.label))
    for el in at.slider:
        parts.append(str(el.label))
    for el in at.toggle:
        parts.append(str(el.label))
    for el in at.button:
        parts.append(str(el.label))
    for el in at.dataframe:
        df = el.value
        parts.append(" ".join(map(str, df.columns)))
        if "Option" in df.columns:
            parts.append(" ".join(map(str, df["Option"].tolist())))
    return "\n".join(parts)


def _assert_clean_wording(text: str) -> None:
    lowered = text.lower()
    for banned in ("win probab", "pressure effect", "chok", "clutch gene"):
        assert banned not in lowered, f"Forbidden phrase {banned!r} found in text."
    for match in re.finditer(r"confidence interval", lowered):
        start = max(0, match.start() - 12)
        prefix = lowered[start : match.start()]
        assert "not a " in prefix, (
            f"Un-negated 'confidence interval' found in: {lowered[start:match.end() + 20]!r}"
        )
    assert not _EMOJI_RE.search(text), "Emoji characters are not allowed in the app UI."
    assert "**" not in text or not re.search(r"\*\*[^*]+\*\*\s*\*\*[^*]+\*\*", text), (
        "Stacked bold markdown found in text."
    )


def _relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast_ratio(hex_a: str, hex_b: str) -> float:
    la = _relative_luminance(hex_a)
    lb = _relative_luminance(hex_b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_1_defaults_headline_examples_and_demo_scenario_1() -> None:
    """App loads cleanly with state headline, 4 example buttons, 5-col table, and Scenario 1."""
    assert LOGO_PATH.exists() and LOGO_PATH.stat().st_size <= 200_000

    at = _run_app()
    assert any(h.value == "Down 3 · 8 s left · Period 4" for h in at.header)
    assert any(APP_DISCLAIMER in c.value for c in at.caption)
    assert any(END_STATE_CAPTION in c.value for c in at.caption)

    df = at.dataframe[0].value
    assert len(df) == 4
    assert list(df.columns) == ["Option", "Make prob.", "xP", "Outcome score", "Rank"]

    if DEMO_PATH.exists():
        demo = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
        sc1_rows = demo["primary_scenarios"][0]["rows"]
        expected_top = min(sc1_rows, key=lambda r: r["rank_v"])["label"]
        assert df.iloc[0]["Option"] == expected_top
    else:
        assert df.iloc[0]["Option"] == "Corner Three"

    assert at.metric[0].label == "Best option"
    assert at.metric[0].value == df.iloc[0]["Option"]

    # Clicking an example scenario button updates state and headline cleanly.
    at.button(key="example_4_6_-1").click().run()
    assert not at.exception
    assert any(h.value == "Down 1 · 6 s left · Period 4" for h in at.header)
    assert at.dataframe[0].value.iloc[0]["Option"] == "Restricted Area"


def test_2_gate_closed_hides_outcome_score_and_shows_message() -> None:
    """At 40 s left, gate-closed message appears and no outcome score / V values appear."""
    at = _run_app()
    at.sidebar.slider(key="time_left_sec").set_value(40).run()
    assert not at.exception

    infos = " ".join(el.value for el in at.info)
    assert "Gate closed" in infos

    for df_el in at.dataframe:
        cols = [str(c) for c in df_el.value.columns]
        assert "Outcome score" not in cols
        assert "V" not in cols
        assert not any(c.startswith("Score") for c in cols)


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
    df_overall = at.dataframe[0].value.sort_values("Option").reset_index(drop=True)

    at.sidebar.radio(key="shift_mode_label").set_value("Off").run()
    assert not at.exception
    df_off = at.dataframe[0].value.sort_values("Option").reset_index(drop=True)

    assert all(
        float(off) > float(on)
        for off, on in zip(df_off["Make prob."], df_overall["Make prob."])
    )


def test_5_shooter_selection_changes_probs_and_shows_note() -> None:
    """Choosing a shooter alphabetically changes probabilities and displays the shrinkage note."""
    model = load_export(DEFAULT_EXPORT_DIR)
    labels, name_to_id = build_shooter_options(model.player_table)
    assert labels[0] == NO_SHOOTER_LABEL
    assert labels[1:] == sorted(labels[1:], key=lambda s: (s.casefold(), s))

    max_row = model.player_table.loc[model.player_table["offset"].abs().idxmax()]
    chosen_name = str(max_row["PLAYER_NAME"])
    assert chosen_name in name_to_id

    at = _run_app()
    assert at.sidebar.selectbox(key="shooter").options == labels
    df_none = at.dataframe[0].value.sort_values("Option").reset_index(drop=True)

    at.sidebar.selectbox(key="shooter").set_value(chosen_name).run()
    assert not at.exception
    df_shooter = at.dataframe[0].value.sort_values("Option").reset_index(drop=True)

    assert not df_none["Make prob."].equals(df_shooter["Make prob."])
    rendered = _collect_all_rendered_text(at)
    assert SHOOTER_NOTE in rendered
    assert f"{float(max_row['offset']):.4f}" not in rendered


def test_6_custom_shot_smoothing_asymmetric_center_and_a4_crosscheck() -> None:
    """Custom shot smoothing differs off-centre, matches on LOC_X=0, and matches A4 within 1e-9."""
    model = load_export(DEFAULT_EXPORT_DIR)
    presets = load_zone_presets()
    court_defaults = load_court_defaults()

    at = _run_app()
    at.toggle(key="use_custom_shot").set_value(True).run()
    at.number_input(key="custom_loc_x").set_value(15)
    at.number_input(key="custom_loc_y").set_value(15)
    at.run()
    assert not at.exception

    df_smooth = at.dataframe[0].value.copy()
    row_smooth = df_smooth[df_smooth["Option"].str.startswith("Custom")].iloc[0]

    at.sidebar.toggle(key="smooth_lr").set_value(False).run()
    assert not at.exception
    df_raw = at.dataframe[0].value.copy()
    row_raw = df_raw[df_raw["Option"].str.startswith("Custom")].iloc[0]

    assert float(row_smooth["Make prob."]) != pytest.approx(float(row_raw["Make prob."]), abs=1e-4)

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
        assert float(disp_row["Make prob."]) == pytest.approx(
            float(res.table.iloc[0]["p_make"]) * 100.0, abs=1e-9
        )
        assert float(disp_row["xP"]) == pytest.approx(float(res.table.iloc[0]["xp"]), abs=1e-9)
        assert float(disp_row["Outcome score"]) == pytest.approx(
            float(res.table.iloc[0]["v"]), abs=1e-9
        )

    at.number_input(key="custom_loc_x").set_value(0).run()
    df_center_off = at.dataframe[0].value.copy()
    row_center_off = df_center_off[df_center_off["Option"].str.startswith("Custom")].iloc[0]

    at.sidebar.toggle(key="smooth_lr").set_value(True).run()
    df_center_on = at.dataframe[0].value.copy()
    row_center_on = df_center_on[df_center_on["Option"].str.startswith("Custom")].iloc[0]

    assert float(row_center_on["Make prob."]) == pytest.approx(
        float(row_center_off["Make prob."]), abs=1e-12
    )
    assert float(row_center_on["xP"]) == pytest.approx(float(row_center_off["xP"]), abs=1e-12)
    assert float(row_center_on["Outcome score"]) == pytest.approx(
        float(row_center_off["Outcome score"]), abs=1e-12
    )


def test_custom_shot_rare_distance_warning() -> None:
    """Rare shot type at a distance shows extrapolation caption; common shot type does not."""
    at = _run_app()
    at.toggle(key="use_custom_shot").set_value(True).run()
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


def test_all_five_summary_sentence_variants() -> None:
    """All summary variants (can_tie, can_lead, neither, no_change, closed_gate) work cleanly."""
    model = load_export(DEFAULT_EXPORT_DIR)
    presets = load_zone_presets()
    opts = ["corner_three", "above_the_break_3", "long_mid_range", "restricted_area"]

    # 1. can_tie (Down 3, 8 s)
    res_tie = evaluate_options(GameState(8.0, -3, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_tie) == "can_tie"
    s_tie = format_decision_summary(res_tie)
    assert "to tie" in s_tie and "outcome score" in s_tie and "top option unchanged" in s_tie

    # 2. can_lead (Down 1, 6 s)
    res_lead = evaluate_options(GameState(6.0, -1, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_lead) == "can_lead"
    s_lead = format_decision_summary(res_lead)
    assert "to go ahead" in s_lead and "outcome score" in s_lead

    # 3. no_option_changes_end_state (Down 5, 10 s)
    res_nc = evaluate_options(GameState(10.0, -5, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_nc) == "no_option_changes_end_state"
    s_nc = format_decision_summary(res_nc)
    assert "same outcome score" in s_nc

    # 4. closed_gate (Down 3, 40 s)
    res_closed = evaluate_options(GameState(40.0, -3, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_closed) == "closed_gate"
    s_closed = format_decision_summary(res_closed)
    assert "outcome scores are hidden" in s_closed

    # 5. neither (best option has p_ahead == 0 and p_tied == 0 without no_option_changes flag)
    synthetic = DecisionResult(
        table=pd.DataFrame(
            [
                {
                    "label": "Option A",
                    "p_make": 0.40,
                    "xp": 0.80,
                    "p_ahead": 0.0,
                    "p_tied": 0.0,
                    "p_behind": 1.0,
                    "v": 0.0,
                    "rank_v": 1,
                }
            ]
        ),
        meta={
            "gate_open": True,
            "no_option_changes_end_state": False,
            "ranking_robust": True,
            "v_gap_first_second": None,
            "state": {"time_left_sec": 8.0, "margin": -4, "period": 4},
        },
    )
    assert classify_summary_variant(synthetic) == "neither"
    s_neither = format_decision_summary(synthetic)
    assert "100.0% behind" in s_neither and "outcome score" in s_neither


def test_7_map_tab_segmented_control_captions_and_gate_disabled() -> None:
    """Map tab renders all three metrics via segmented_control and disables Outcome score when closed."""
    at = _run_app()
    seg = at.segmented_control(key="map_metric")
    assert seg.options == ["Make probability", "Expected points (xP)", "Outcome score"]

    for metric_label in ("Make probability", "Expected points (xP)", "Outcome score"):
        at.segmented_control(key="map_metric").set_value(metric_label).run()
        assert not at.exception

    captions = " ".join(c.value for c in at.caption)
    assert MAP_TYPICAL_MIX_CAPTION in captions
    assert MAP_MODEL_STEPS_CAPTION in captions
    assert "Within about 4 ft of the rim" in captions

    at.sidebar.slider(key="time_left_sec").set_value(40).run()
    assert not at.exception
    closed_seg = at.segmented_control(key="map_metric")
    assert "Outcome score" not in closed_seg.options
    assert "Outcome score is disabled" in " ".join(i.value for i in at.info)


def test_8_wording_scan_and_theme_contrast_ratios() -> None:
    """Rendered text and literals pass wording/emoji/bold checks and theme contrast >= 4.5:1."""
    for file_path in (ROOT / "app" / "formatting.py", APP_PATH):
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        _assert_clean_wording("\n".join(literals))

    at = _run_app()
    _assert_clean_wording(_collect_all_rendered_text(at))

    at.sidebar.radio(key="shift_mode_label").set_value("By shot type (exploratory)")
    at.sidebar.slider(key="time_left_sec").set_value(40)
    at.run()
    _assert_clean_wording(_collect_all_rendered_text(at))

    at.sidebar.slider(key="time_left_sec").set_value(10)
    at.sidebar.slider(key="margin").set_value(-5)
    at.run()
    _assert_clean_wording(_collect_all_rendered_text(at))

    cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    theme = cfg["theme"]
    assert _contrast_ratio(theme["textColor"], theme["backgroundColor"]) >= 4.5
    assert _contrast_ratio(theme["textColor"], theme["secondaryBackgroundColor"]) >= 4.5
    assert _contrast_ratio("#ffffff", theme["primaryColor"]) >= 4.5
    assert _contrast_ratio(theme["linkColor"], theme["backgroundColor"]) >= 4.5


def test_9_app_runs_in_clean_copy_without_data_and_models(tmp_path: Path) -> None:
    """AppTest boots, renders logo/defaults, and runs example buttons without data/ or models/."""
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
        "at.button(key='example_4_8_-2').click().run(); "
        "dt = time.perf_counter() - t0; "
        "assert not at.exception, at.exception; "
        "assert any(h.value == 'Down 2 · 8 s left · Period 4' for h in at.header); "
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
