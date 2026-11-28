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
from types import SimpleNamespace

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
    PERIOD_OPTIONS,
    SHOOTER_NOTE,
    build_altair_court_chart,
    build_shooter_options,
    classify_summary_variant,
    format_decision_answer_parts,
    format_decision_summary,
    format_spot_headline,
    format_spot_position,
    parse_map_selection,
)
from src.decision_layer import DecisionResult, GameState, evaluate_options, load_zone_presets
from src.heatmap import (
    HEATMAP_X_MAX,
    HEATMAP_X_MIN,
    HEATMAP_Y_MAX,
    HEATMAP_Y_MIN,
    build_custom_shot_preset,
    compute_heatmap,
    load_court_defaults,
    lookup_cell,
)
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
    for el in at.number_input:
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


def _assert_no_visitor_jargon(text: str) -> None:
    """Assert visitor-visible text contains no internal terms, 'depth', or position 'left-right'."""
    lowered = text.lower()
    assert "loc_x" not in lowered, "Visitor UI text must not mention LOC_X."
    assert "loc_y" not in lowered, "Visitor UI text must not mention LOC_Y."
    assert not re.search(r"\bcells?\b", lowered), "Visitor UI text must not mention 'cell' or 'cells'."
    assert "tenths of a foot" not in lowered, "Visitor UI text must not mention 'tenths of a foot'."
    assert not re.search(r"\btenths?\b", lowered), "Visitor UI text must not mention 'tenths'."
    assert "shift sensitivity" not in lowered, "Visitor UI text must not mention 'Shift sensitivity'."
    assert "depth" not in lowered, "Visitor UI text must not mention 'depth'."
    without_smoothing = (
        lowered.replace("left-right differences", "").replace(
            "left-right coordinate differences", ""
        )
    )
    assert "left-right" not in without_smoothing, (
        "Visitor position UI text must not use 'left-right'."
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
    """App loads cleanly with state headline, sidebar A8 controls, answer block, and Scenario 1."""
    assert LOGO_PATH.exists() and LOGO_PATH.stat().st_size <= 200_000

    at = _run_app()
    assert any(h.value == "Down 3 · 8 s left · Period 4" for h in at.header)
    assert any(APP_DISCLAIMER in c.value for c in at.caption)
    assert any(END_STATE_CAPTION in c.value for c in at.caption)

    period_ctrl = at.sidebar.segmented_control(key="period_label")
    assert period_ctrl.options == list(PERIOD_OPTIONS.keys())
    assert period_ctrl.value == "4th quarter"
    assert int(at.sidebar.number_input(key="time_left_sec").value) == 8
    margin_slider = at.sidebar.slider(key="margin")
    assert margin_slider.label == "Score margin (offense minus defense)"
    assert int(margin_slider.value) == -3

    rendered = _collect_all_rendered_text(at)
    assert "ties the game" in rendered
    assert "Next:" in rendered
    assert "The best option stays the same with or without the clutch adjustment." in rendered
    assert "+0." not in rendered

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
    at.button(key="example_4th quarter_6_-1").click().run()
    assert not at.exception
    assert any(h.value == "Down 1 · 6 s left · Period 4" for h in at.header)
    assert at.dataframe[0].value.iloc[0]["Option"] == "Restricted Area"

    # Switching Period segmented control to Overtime updates the headline cleanly.
    at.sidebar.segmented_control(key="period_label").set_value("Overtime").run()
    assert not at.exception
    assert any("Overtime" in h.value for h in at.header)


def test_2_gate_closed_hides_outcome_score_and_shows_message() -> None:
    """At 40 s left, gate-closed message appears and no outcome score / V values appear."""
    at = _run_app()
    at.sidebar.number_input(key="time_left_sec").set_value(40).run()
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
    at.sidebar.number_input(key="time_left_sec").set_value(10)
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
    assert NO_SHOOTER_LABEL == "No player"
    assert any(h.value == "Player (optional)" for h in at.sidebar.subheader)
    assert at.sidebar.selectbox(key="shooter").label == "Player (optional)"
    assert at.sidebar.selectbox(key="shooter").options == labels
    sidebar_labels = " ".join(
        [h.value for h in at.sidebar.subheader]
        + [s.label for s in at.sidebar.selectbox]
        + [labels[0]]
    ).lower()
    assert "shooter" not in sidebar_labels
    df_none = at.dataframe[0].value.sort_values("Option").reset_index(drop=True)

    at.sidebar.selectbox(key="shooter").set_value(chosen_name).run()
    assert not at.exception
    df_shooter = at.dataframe[0].value.sort_values("Option").reset_index(drop=True)

    assert not df_none["Make prob."].equals(df_shooter["Make prob."])
    rendered = _collect_all_rendered_text(at)
    assert SHOOTER_NOTE in rendered
    assert "shooter adjustments" not in rendered.lower()
    assert f"{float(max_row['offset']):.4f}" not in rendered


def test_6_custom_shot_smoothing_asymmetric_center_and_a4_crosscheck() -> None:
    """Custom shot smoothing in feet differs off-centre, matches on center, and matches A4 within 1e-9."""
    model = load_export(DEFAULT_EXPORT_DIR)
    presets = load_zone_presets()
    court_defaults = load_court_defaults()

    at = _run_app()
    at.toggle(key="use_custom_shot").set_value(True).run()
    at.number_input(key="custom_x_ft").set_value(1.5)
    at.number_input(key="custom_y_ft").set_value(1.5)
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

    at.number_input(key="custom_x_ft").set_value(0.0).run()
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
    at.number_input(key="custom_x_ft").set_value(0.0)
    at.number_input(key="custom_y_ft").set_value(25.0)
    at.selectbox(key="custom_shot_type").set_value("dunk").run()
    assert not at.exception

    captions = " ".join(c.value for c in at.caption)
    assert CUSTOM_SHOT_RARE_WARNING in captions

    at.selectbox(key="custom_shot_type").set_value("jump_shot").run()
    assert not at.exception
    captions_after = " ".join(c.value for c in at.caption)
    assert CUSTOM_SHOT_RARE_WARNING not in captions_after


def test_all_five_summary_sentence_variants() -> None:
    """All A8 answer-block variants (can_tie, can_lead, neither, no_change, closed_gate) work cleanly."""
    model = load_export(DEFAULT_EXPORT_DIR)
    presets = load_zone_presets()
    opts = ["corner_three", "above_the_break_3", "long_mid_range", "restricted_area"]

    # 1. can_tie (Down 3, 8 s)
    res_tie = evaluate_options(GameState(8.0, -3, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_tie) == "can_tie"
    m_tie, n_tie, st_tie = format_decision_answer_parts(res_tie)
    assert "is the best option. It ties the game" in m_tie
    assert "outcome score" in m_tie
    assert "+0." not in m_tie and " over " not in m_tie
    assert n_tie is not None and n_tie.startswith("Next:")
    assert st_tie == "The best option stays the same with or without the clutch adjustment."

    # 2. can_lead (Down 1, 6 s)
    res_lead = evaluate_options(GameState(6.0, -1, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_lead) == "can_lead"
    m_lead, n_lead, st_lead = format_decision_answer_parts(res_lead)
    assert "is the best option. It puts the offense ahead" in m_lead
    assert "outcome score" in m_lead
    assert n_lead is not None and n_lead.startswith("Next:")
    assert st_lead is not None

    # 3. no_option_changes_end_state (Down 5, 10 s)
    res_nc = evaluate_options(GameState(10.0, -5, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_nc) == "no_option_changes_end_state"
    m_nc, n_nc, st_nc = format_decision_answer_parts(res_nc)
    assert "No selected shot can tie or take the lead" in m_nc
    assert "leads in xP" in m_nc
    assert n_nc is not None and n_nc.startswith("Next:")
    assert st_nc is not None

    # 4. closed_gate (Down 3, 40 s)
    res_closed = evaluate_options(GameState(40.0, -3, 4), opts, model=model, presets=presets)
    assert classify_summary_variant(res_closed) == "closed_gate"
    m_closed, n_closed, st_closed = format_decision_answer_parts(res_closed)
    assert "outcome scores are hidden" in m_closed
    assert "leads in xP" in m_closed
    assert n_closed is not None and n_closed.startswith("Next:")
    assert st_closed is None

    # 5. neither (p_ahead == 0 and p_tied == 0 without no_option_changes flag, plus unstable ranking branch)
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
                },
                {
                    "label": "Option B",
                    "p_make": 0.35,
                    "xp": 0.70,
                    "p_ahead": 0.0,
                    "p_tied": 0.0,
                    "p_behind": 1.0,
                    "v": 0.0,
                    "rank_v": 2,
                },
            ]
        ),
        meta={
            "gate_open": True,
            "no_option_changes_end_state": False,
            "ranking_robust": False,
            "v_gap_first_second": 0.0,
            "state": {"time_left_sec": 8.0, "margin": -4, "period": 4},
        },
    )
    assert classify_summary_variant(synthetic) == "neither"
    m_n, n_n, st_n = format_decision_answer_parts(synthetic)
    assert "leaves the offense behind 100.0% of the time" in m_n
    assert "outcome score 0.00" in m_n
    assert n_n == "Next: Option B, outcome score 0.00 against 0.00."
    assert st_n == "The best option changes with the clutch adjustment."
    assert "Option A is the best option." in format_decision_summary(synthetic)


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

    at.sidebar.number_input(key="time_left_sec").set_value(40).run()
    assert not at.exception
    closed_seg = at.segmented_control(key="map_metric")
    assert "Outcome score" not in closed_seg.options
    assert "Outcome score is disabled" in " ".join(i.value for i in at.info)


def test_map_selection_parser_click_inspect_and_use_spot_flow() -> None:
    """Altair click payloads, plain spot position copy, exact-position edits, and Use-this-spot stay synchronized."""
    assert parse_map_selection(None) is None
    assert parse_map_selection({}) is None
    assert parse_map_selection({"selection": {"spot": []}}) is None
    assert parse_map_selection({"selection": {"spot": [{"x_ft": "bad", "y_ft": 1.0}]}}) is None
    assert parse_map_selection({"selection": {"spot": [{"x_ft": 22.0, "y_ft": 1.0}]}}) == (220.0, 10.0)
    assert parse_map_selection(SimpleNamespace(selection={"spot": [{"x_ft": -99.0, "y_ft": 99.0}]})) == (
        HEATMAP_X_MIN,
        HEATMAP_Y_MAX,
    )
    assert parse_map_selection({"spot": [{"x_ft": 99.0, "y_ft": -99.0}]}) == (
        HEATMAP_X_MAX,
        HEATMAP_Y_MIN,
    )

    assert format_spot_headline(22.0, 3) == "22 ft from the basket · three\u2011pointer"
    assert format_spot_headline(10.0, 2) == "10 ft from the basket · two\u2011pointer"
    assert "-" not in format_spot_headline(22.0, 3)
    assert "-" not in format_spot_headline(10.0, 2)

    assert format_spot_position(-9.5, 4.5) == "9.5 ft left of the basket · 4.5 ft in front of it"
    assert format_spot_position(22.5, -2.0) == "22.5 ft right of the basket · 2.0 ft behind it"
    assert format_spot_position(0.0, 25.0) == "in line with the basket · 25.0 ft in front of it"
    assert format_spot_position(-22.5, 0.0) == "22.5 ft left of the basket · in line with the hoop"
    assert format_spot_position(0.0, 0.0) == "in line with the basket · in line with the hoop"

    model = load_export(DEFAULT_EXPORT_DIR)
    presets = load_zone_presets()
    court_defaults = load_court_defaults()

    at = _run_app()
    assert (
        at.number_input(key="inspect_x_ft").label
        == "Left or right of the basket (ft, negative = left)"
    )
    assert (
        at.number_input(key="inspect_y_ft").label
        == "In front of the basket (ft, negative = behind it)"
    )

    # Simulated click on above-the-break three (0.5 ft, 25.5 ft) via map_chart session state.
    at.session_state["map_chart"] = {"selection": {"spot": [{"x_ft": 0.5, "y_ft": 25.5}]}}
    at.run()
    assert not at.exception
    assert float(at.number_input(key="inspect_x_ft").value) == pytest.approx(0.5)
    assert float(at.number_input(key="inspect_y_ft").value) == pytest.approx(25.5)
    assert any(
        h.value == format_spot_headline(25.0, 3) for h in at.subheader
    ), [h.value for h in at.subheader]
    assert any(
        c.value == "0.5 ft right of the basket · 25.5 ft in front of it" for c in at.caption
    )

    # Exact-position edit overrides inspected coordinates and checks plain relative wording.
    at.number_input(key="inspect_x_ft").set_value(-9.5)
    at.number_input(key="inspect_y_ft").set_value(4.5)
    at.run()
    assert not at.exception
    assert any(
        c.value == "9.5 ft left of the basket · 4.5 ft in front of it" for c in at.caption
    )

    at.number_input(key="inspect_x_ft").set_value(-22.5)
    at.number_input(key="inspect_y_ft").set_value(0.5)
    at.run()
    assert not at.exception
    hm = compute_heatmap(
        GameState(8.0, -3, 4),
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    expected_spot = lookup_cell(hm, -225.0, 5.0)
    metric_pairs = {(m.label, m.value) for m in at.metric}
    assert ("Make probability", f"{expected_spot['p_make']:.1%}") in metric_pairs
    assert ("Outcome score", f"{expected_spot['v']:.2f}") in metric_pairs

    # Clicking "Use this spot as a custom shot" turns on custom shot with matching coordinates.
    at.button(key="use_spot_custom").click().run()
    assert not at.exception
    assert bool(at.toggle(key="use_custom_shot").value) is True
    assert float(at.number_input(key="custom_x_ft").value) == pytest.approx(-22.5)
    assert float(at.number_input(key="custom_y_ft").value) == pytest.approx(0.5)

    custom_row = at.dataframe[0].value
    custom_row = custom_row[custom_row["Option"].str.startswith("Custom")].iloc[0]
    custom_preset = build_custom_shot_preset(
        -225.0,
        5.0,
        shot_type="typical_mix",
        smooth=True,
        court_defaults=court_defaults,
        label="Custom (Typical mix)",
    )
    expected_res = evaluate_options(
        state=GameState(8.0, -3, 4),
        options=["__custom__"],
        shift_mode="overall",
        model=model,
        presets={**presets, "__custom__": custom_preset},
    )
    assert float(custom_row["Make prob."]) == pytest.approx(
        float(expected_res.table.iloc[0]["p_make"]) * 100.0, abs=1e-9
    )


def test_altair_chart_spec_and_layout_properties() -> None:
    """Altair court spec validates equal 470/500 x/y aspect, exact court domains, no cell borders, and selection."""
    model = load_export(DEFAULT_EXPORT_DIR)
    court_defaults = load_court_defaults()
    hm = compute_heatmap(
        GameState(8.0, -3, 4),
        smooth=True,
        model=model,
        court_defaults=court_defaults,
    )
    chart = build_altair_court_chart(hm, metric="v", selected_xy_tenths=(220.0, 10.0))
    spec = chart.to_dict(validate=True)
    assert spec["width"] == 600
    assert spec["height"] == 564
    assert spec["height"] / spec["width"] == pytest.approx(470.0 / 500.0, rel=0.0, abs=1e-12)
    assert spec["autosize"] == {"type": "pad", "contains": "padding"}
    assert len(spec["layer"]) == 6

    heat_layer = spec["layer"][0]
    assert heat_layer["mark"]["strokeWidth"] == 0
    assert heat_layer["mark"]["clip"] is True
    assert heat_layer["encoding"]["x"]["scale"]["domain"] == [-25.0, 25.0]
    assert heat_layer["encoding"]["y"]["scale"]["domain"] == [-4.75, 42.25]
    assert heat_layer["encoding"]["color"]["legend"]["orient"] == "bottom"
    assert heat_layer["encoding"]["color"]["scale"]["scheme"] == "viridis"

    for layer in spec["layer"][1:]:
        assert layer["encoding"]["x"]["scale"]["domain"] == [-25.0, 25.0]
        assert layer["encoding"]["y"]["scale"]["domain"] == [-4.75, 42.25]

    tooltip_titles = [t["title"].lower() for t in spec["layer"][-1]["encoding"]["tooltip"]]
    assert all("depth" not in t and "left-right" not in t for t in tooltip_titles)
    assert spec["params"][0]["name"] == "spot"
    assert spec["params"][0]["select"]["on"] == "click"
    assert spec["params"][0]["views"] == [spec["layer"][-1]["name"]]


def test_8_wording_scan_and_theme_contrast_ratios() -> None:
    """Rendered text passes wording/jargon/emoji/bold checks and theme contrast >= 4.5:1."""
    tree_fmt = ast.parse((ROOT / "app" / "formatting.py").read_text(encoding="utf-8"))
    fmt_literals = [
        node.value
        for node in ast.walk(tree_fmt)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    _assert_clean_wording("\n".join(fmt_literals))

    tree_app = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    app_literals = [
        node.value
        for node in ast.walk(tree_app)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    _assert_clean_wording("\n".join(app_literals))

    at = _run_app()
    rendered_default = _collect_all_rendered_text(at)
    _assert_clean_wording(rendered_default)
    _assert_no_visitor_jargon(rendered_default)

    at.toggle(key="use_custom_shot").set_value(True)
    at.sidebar.segmented_control(key="period_label").set_value("Overtime")
    at.sidebar.radio(key="shift_mode_label").set_value("By shot type (exploratory)")
    at.sidebar.number_input(key="time_left_sec").set_value(40)
    at.run()
    rendered_closed = _collect_all_rendered_text(at)
    _assert_clean_wording(rendered_closed)
    _assert_no_visitor_jargon(rendered_closed)

    at.sidebar.number_input(key="time_left_sec").set_value(10)
    at.sidebar.slider(key="margin").set_value(-5)
    at.run()
    rendered_nc = _collect_all_rendered_text(at)
    _assert_clean_wording(rendered_nc)
    _assert_no_visitor_jargon(rendered_nc)

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
        "at.button(key='example_4th quarter_8_-2').click().run(); "
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
