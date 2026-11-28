"""Phase 4c: Interactive Streamlit Clutch Decision & Half-Court Simulator App.

Run locally from the repository root with::

    venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.formatting import (  # noqa: E402
    APP_DISCLAIMER,
    CUSTOM_SHOT_RARE_WARNING,
    DEFAULT_PRESET_KEYS,
    END_STATE_CAPTION,
    EXAMPLE_SCENARIOS,
    MAP_MODEL_STEPS_CAPTION,
    MAP_TYPICAL_MIX_CAPTION,
    OUTCOME_SCORE_TOOLTIP,
    PERIOD_OPTIONS,
    REPO_URL,
    SENSITIVITY_CAPTION,
    SHIFT_MODE_OPTIONS,
    SHOOTER_NOTE,
    build_altair_court_chart,
    build_preset_catalog,
    build_shooter_options,
    format_decision_answer_parts,
    format_decision_table,
    format_exploratory_shift_note,
    format_flag_messages,
    format_map_limitation_note,
    format_sensitivity_table,
    format_spot_headline,
    format_spot_position,
    format_state_headline,
    is_rare_shot_type,
    parse_map_selection,
    render_end_state_bar,
)
from src.decision_layer import GameState, evaluate_options, load_zone_presets  # noqa: E402
from src.heatmap import (  # noqa: E402
    ACTION_GROUP_LABELS,
    HeatmapResult,
    TYPICAL_MIX_KEY,
    build_custom_shot_preset,
    compute_heatmap,
    get_active_shot_types,
    load_court_defaults,
    lookup_cell,
)
from src.shot_geometry import compute_shot_distance_and_angle, compute_shot_value  # noqa: E402
from src.simulator_model import ExportedModel, load_export  # noqa: E402

LOGO_PATH = ROOT / "app" / "assets" / "nba-clutch.png"

st.set_page_config(
    page_title="NBA Clutch Decision Simulator",
    page_icon=":material/sports_basketball:",
    layout="wide",
)
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")


def _init_session_defaults() -> None:
    """Initialize widget defaults in session_state once so callbacks stay synchronized."""
    defaults: dict[str, Any] = {
        "period_label": "4th quarter",
        "time_left_sec": 8,
        "margin": -3,
        "shift_mode_label": "Overall (default)",
        "tie_value": 0.5,
        "time_gate_sec": 24,
        "smooth_lr": True,
        "map_metric": "Outcome score",
        "use_custom_shot": False,
        "custom_x_ft": 0.0,
        "custom_y_ft": 25.0,
        "inspect_x_ft": 22.5,
        "inspect_y_ft": 0.5,
        "_last_map_selection_xy": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _apply_example_state(period_label: str, time_left_sec: int, margin: int) -> None:
    """Update situation inputs from an example scenario button before rerun."""
    st.session_state["period_label"] = period_label
    st.session_state["time_left_sec"] = time_left_sec
    st.session_state["margin"] = margin


def _apply_spot_to_custom_shot(x_ft: float, y_ft: float) -> None:
    """Copy the inspected court spot onto the Decision tab's custom shot."""
    st.session_state["use_custom_shot"] = True
    st.session_state["custom_x_ft"] = round(float(x_ft), 1)
    st.session_state["custom_y_ft"] = round(float(y_ft), 1)


def _sync_map_selection(force: bool = False) -> None:
    """Sync Altair click-selection payload into inspected foot coordinates."""
    xy = parse_map_selection(st.session_state.get("map_chart"))
    if xy is None:
        return
    if not force and xy == st.session_state.get("_last_map_selection_xy"):
        return
    st.session_state["_last_map_selection_xy"] = xy
    st.session_state["inspect_x_ft"] = round(xy[0] / 10.0, 1)
    st.session_state["inspect_y_ft"] = round(xy[1] / 10.0, 1)


def _on_map_select() -> None:
    """Callback invoked when the user clicks a spot on the Altair court map."""
    _sync_map_selection(force=True)


@st.cache_resource
def load_app_resources() -> tuple[ExportedModel, dict[str, Any], dict[str, Any]]:
    """Load exported model, zone presets, and court defaults once per server process."""
    model = load_export()
    presets = load_zone_presets()
    court_defaults = load_court_defaults()
    return model, presets, court_defaults


@st.cache_data(max_entries=64)
def get_cached_heatmap(
    _model: ExportedModel,
    _court_defaults: dict[str, Any],
    period: int,
    time_left_sec: int,
    margin: int,
    player_id: str | None,
    shift_mode: str,
    shot_type: str,
    smooth: bool,
    time_gate_sec: int,
    tie_value: float,
) -> HeatmapResult:
    """Compute and cache half-court grids so switching metric does not recompute."""
    state = GameState(
        time_left_sec=float(time_left_sec),
        margin=int(margin),
        period=int(period),
    )
    return compute_heatmap(
        state=state,
        player_id=player_id,
        shift_mode=shift_mode,  # type: ignore[arg-type]
        shot_type=shot_type,
        smooth=smooth,
        time_gate_sec=float(time_gate_sec),
        tie_value=float(tie_value),
        model=_model,
        court_defaults=_court_defaults,
    )


def _render_sidebar(
    model: ExportedModel,
) -> tuple[GameState, str | None, str, bool, float, int]:
    st.sidebar.subheader("Situation")
    period_label = st.sidebar.segmented_control(
        "Period",
        options=list(PERIOD_OPTIONS.keys()),
        key="period_label",
        wrap=True,
    )
    if period_label not in PERIOD_OPTIONS:
        period_label = "4th quarter"
    period = PERIOD_OPTIONS[period_label]
    time_left_sec = st.sidebar.number_input(
        "Time left (seconds)",
        min_value=0,
        max_value=300,
        step=1,
        key="time_left_sec",
    )
    margin = st.sidebar.slider(
        "Score margin (offense minus defense)",
        min_value=-10,
        max_value=10,
        step=1,
        key="margin",
        help="Negative means the offense trails before taking the shot.",
    )

    st.sidebar.subheader("Player (optional)")
    shooter_labels, name_to_id = build_shooter_options(model.player_table)
    selected_shooter = st.sidebar.selectbox(
        "Player (optional)",
        options=shooter_labels,
        index=0,
        key="shooter",
        label_visibility="collapsed",
    )
    player_id = name_to_id.get(selected_shooter)
    if player_id is not None:
        st.sidebar.caption(SHOOTER_NOTE)

    with st.sidebar.expander("Model settings", expanded=False):
        shift_label = st.radio(
            "Clutch shift",
            options=list(SHIFT_MODE_OPTIONS.keys()),
            key="shift_mode_label",
            help=(
                "Descriptive gap on close late-game shots bundling defense, fatigue, "
                "shot selection within a type, and player mix."
            ),
        )
        shift_mode = SHIFT_MODE_OPTIONS[shift_label]
        if shift_mode == "by_shot_type":
            st.caption(format_exploratory_shift_note(model.clutch_shift_info))

        tie_value = st.slider(
            "Overtime value",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="tie_value",
            help="Weight given to ending tied in outcome score. Default 0.5 treats overtime as a coin flip.",
        )
        time_gate_sec = st.number_input(
            "Decision time gate (seconds)",
            min_value=0,
            max_value=300,
            step=1,
            key="time_gate_sec",
            help="Outcome scores appear only up to this clock limit (default 24 s).",
        )
        smooth = st.toggle(
            "Smooth left-right differences",
            key="smooth_lr",
            help="Averages each court point with its mirror across the centre line.",
        )

    state = GameState(
        time_left_sec=float(time_left_sec),
        margin=int(margin),
        period=int(period),
    )
    return state, player_id, shift_mode, bool(smooth), float(tie_value), int(time_gate_sec)


def _decision_column_config(gate_open: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "Option": st.column_config.TextColumn("Option", width="medium"),
        "Make prob.": st.column_config.NumberColumn("Make prob.", format="%.1f%%"),
        "xP": st.column_config.NumberColumn("xP", format="%.2f"),
    }
    if gate_open:
        config["Outcome score"] = st.column_config.NumberColumn(
            "Outcome score",
            format="%.2f",
            help=OUTCOME_SCORE_TOOLTIP,
        )
        config["Rank"] = st.column_config.NumberColumn("Rank", format="%d", width="small")
    return config


def _sensitivity_column_config(gate_open: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "Option": st.column_config.TextColumn("Option"),
        "Make (no shift)": st.column_config.NumberColumn("Make (no shift)", format="%.1f%%"),
        "Make (low)": st.column_config.NumberColumn("Make (low)", format="%.1f%%"),
        "Make (point)": st.column_config.NumberColumn("Make (point)", format="%.1f%%"),
        "Make (high)": st.column_config.NumberColumn("Make (high)", format="%.1f%%"),
    }
    if gate_open:
        config["Score (no shift)"] = st.column_config.NumberColumn(
            "Score (no shift)", format="%.2f"
        )
        config["Score (low)"] = st.column_config.NumberColumn("Score (low)", format="%.2f")
        config["Score (point)"] = st.column_config.NumberColumn("Score (point)", format="%.2f")
        config["Score (high)"] = st.column_config.NumberColumn("Score (high)", format="%.2f")
    return config


def _render_decision_tab(
    model: ExportedModel,
    presets: dict[str, Any],
    court_defaults: dict[str, Any],
    state: GameState,
    player_id: str | None,
    shift_mode: str,
    smooth: bool,
    tie_value: float,
    time_gate_sec: int,
) -> tuple[bool, tuple[float, float] | None]:
    st.caption(APP_DISCLAIMER)
    st.header(format_state_headline(state))

    with st.container(horizontal=True, wrap=True):
        for btn_label, ex_period, ex_time, ex_margin in EXAMPLE_SCENARIOS:
            st.button(
                btn_label,
                key=f"example_{ex_period}_{ex_time}_{ex_margin}",
                on_click=_apply_example_state,
                args=(ex_period, ex_time, ex_margin),
            )

    preset_labels, label_to_key = build_preset_catalog(presets)
    default_labels = [
        str(presets[k]["label"]) for k in DEFAULT_PRESET_KEYS if k in presets
    ]
    selected_labels = st.multiselect(
        "Shot options",
        options=preset_labels,
        default=default_labels,
        key="selected_presets",
        help="Each zone preset averages predictions over real normal-play shots.",
    )
    selected_options: list[str] = [label_to_key[lbl] for lbl in selected_labels]

    shot_type_keys = [TYPICAL_MIX_KEY, *get_active_shot_types(court_defaults)]
    use_custom = st.toggle("Include custom shot", key="use_custom_shot")

    eval_presets = dict(presets)
    custom_xy: tuple[float, float] | None = None
    if use_custom:
        c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
        with c1:
            custom_x_ft = st.number_input(
                "Left or right of the basket (ft, negative = left)",
                min_value=-25.0,
                max_value=25.0,
                step=0.5,
                format="%.1f",
                key="custom_x_ft",
                help="Negative is left of the basket, positive is right, 0.0 ft is in line with it.",
            )
        with c2:
            custom_y_ft = st.number_input(
                "In front of the basket (ft, negative = behind it)",
                min_value=-5.0,
                max_value=42.0,
                step=0.5,
                format="%.1f",
                key="custom_y_ft",
                help="0.0 ft is the centre of the hoop, -4.75 ft is the baseline behind it, and 42.0 ft is near half court.",
            )
        with c3:
            custom_shot_type = st.selectbox(
                "Custom shot type",
                options=shot_type_keys,
                index=0,
                format_func=lambda k: ACTION_GROUP_LABELS[k],
                key="custom_shot_type",
            )

        custom_x = round(float(custom_x_ft) * 10.0, 1)
        custom_y = round(float(custom_y_ft) * 10.0, 1)
        dist_ft, angle_deg = compute_shot_distance_and_angle(
            custom_x, custom_y, is_putback=0, clamp_distance=True
        )
        shot_val = compute_shot_value(custom_x, custom_y)
        custom_xy = (custom_x, custom_y)
        st.caption(f"{format_spot_headline(dist_ft, shot_val)} · {angle_deg:.1f} deg")

        if is_rare_shot_type(court_defaults, dist_ft, custom_shot_type):
            st.caption(CUSTOM_SHOT_RARE_WARNING)

        custom_key = "__custom_shot__"
        custom_label = f"Custom ({ACTION_GROUP_LABELS[custom_shot_type]})"
        eval_presets[custom_key] = build_custom_shot_preset(
            loc_x=custom_x,
            loc_y=custom_y,
            shot_type=custom_shot_type,
            smooth=smooth,
            court_defaults=court_defaults,
            label=custom_label,
        )
        selected_options.append(custom_key)

    if not selected_options:
        st.info("Select at least one shot option or enable a custom shot.")
        return False, custom_xy

    result = evaluate_options(
        state=state,
        options=selected_options,
        player_id=player_id,
        shift_mode=shift_mode,  # type: ignore[arg-type]
        time_gate_sec=float(time_gate_sec),
        tie_value=float(tie_value),
        model=model,
        presets=eval_presets,
    )

    gate_open = bool(result.meta["gate_open"])
    flag_messages = format_flag_messages(result.meta)

    for severity, msg in flag_messages:
        if severity == "warning":
            st.warning(msg)
        elif "Gate closed" in msg:
            st.info(msg)

    display_df = format_decision_table(result)
    top_label = str(display_df.iloc[0]["Option"])

    main_sentence, next_line, stability_line = format_decision_answer_parts(result)
    with st.container(border=True):
        st.metric("Best option", top_label)
        st.write(main_sentence)
        if next_line is not None:
            st.write(next_line)
        if stability_line is not None:
            st.caption(stability_line)
    if gate_open:
        st.caption(END_STATE_CAPTION)

    if gate_open:
        col_tbl, col_bar = st.columns([1.15, 1.0], vertical_alignment="top")
        with col_tbl:
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                column_config=_decision_column_config(gate_open=True),
                alt="Ranked shot comparison table showing make probability, expected points, outcome score, and rank.",
            )
            st.caption(OUTCOME_SCORE_TOOLTIP)
        with col_bar:
            bar_fig = render_end_state_bar(result)
            st.pyplot(
                bar_fig,
                clear_figure=True,
                width="stretch",
                alt="Stacked horizontal bars showing ahead, tied, and behind percentages after the shot.",
            )
            plt.close(bar_fig)
    else:
        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            column_config=_decision_column_config(gate_open=False),
            alt="Shot comparison table showing make probability and expected points.",
        )

    with st.expander("Details", expanded=False):
        st.caption(SENSITIVITY_CAPTION)
        sens_df = format_sensitivity_table(result)
        st.dataframe(
            sens_df,
            hide_index=True,
            width="stretch",
            column_config=_sensitivity_column_config(gate_open=gate_open),
            alt="Clutch-adjustment sensitivity table across no shift, lower bound, point estimate, and upper bound.",
        )
        if player_id is not None:
            st.caption(SHOOTER_NOTE)
        for _, msg in flag_messages:
            st.caption(msg)
        for note in result.meta["assumptions"]:
            st.caption(note)

    return use_custom, custom_xy


def _render_map_tab(
    model: ExportedModel,
    court_defaults: dict[str, Any],
    state: GameState,
    player_id: str | None,
    shift_mode: str,
    smooth: bool,
    tie_value: float,
    time_gate_sec: int,
) -> None:
    st.caption(APP_DISCLAIMER)
    _sync_map_selection(force=False)

    gate_open = state.time_left_sec <= float(time_gate_sec)
    metric_label_to_key = {
        "Make probability": "p_make",
        "Expected points (xP)": "xp",
        "Outcome score": "v",
    }
    if gate_open:
        metric_options = list(metric_label_to_key.keys())
    else:
        metric_options = ["Make probability", "Expected points (xP)"]
        if st.session_state.get("map_metric") not in metric_options:
            st.session_state["map_metric"] = "Make probability"
        st.info(
            f"Outcome score is disabled when the decision gate is closed "
            f"(more than {time_gate_sec} s left)."
        )

    shot_type_keys = [TYPICAL_MIX_KEY, *get_active_shot_types(court_defaults)]

    with st.container(horizontal=True, vertical_alignment="bottom", wrap=True):
        metric_label = st.segmented_control(
            "Map metric",
            options=metric_options,
            key="map_metric",
            wrap=True,
        )
        if metric_label not in metric_options:
            metric_label = metric_options[0]

        map_shot_type = st.selectbox(
            "Shot type on map",
            options=shot_type_keys,
            index=0,
            format_func=lambda k: ACTION_GROUP_LABELS[k],
            key="map_shot_type",
            width=240,
        )

    heatmap = get_cached_heatmap(
        _model=model,
        _court_defaults=court_defaults,
        period=state.period,
        time_left_sec=int(round(state.time_left_sec)),
        margin=state.margin,
        player_id=player_id,
        shift_mode=shift_mode,
        shot_type=map_shot_type,
        smooth=smooth,
        time_gate_sec=time_gate_sec,
        tie_value=tie_value,
    )

    inspect_x_ft = float(st.session_state["inspect_x_ft"])
    inspect_y_ft = float(st.session_state["inspect_y_ft"])
    inspect_x_tenths = round(inspect_x_ft * 10.0, 1)
    inspect_y_tenths = round(inspect_y_ft * 10.0, 1)
    cell_info = lookup_cell(heatmap, inspect_x_tenths, inspect_y_tenths)
    metric_key = metric_label_to_key[metric_label]

    c_plot, c_panel = st.columns([1.5, 1.0], vertical_alignment="top")
    with c_plot:
        chart = build_altair_court_chart(
            heatmap,
            metric=metric_key,  # type: ignore[arg-type]
            selected_xy_tenths=(inspect_x_tenths, inspect_y_tenths),
        )
        with st.container(horizontal=True, wrap=False):
            st.altair_chart(
                chart,
                width="content",
                theme=None,
                on_select=_on_map_select,
                key="map_chart",
                alt=f"Interactive half-court map showing {metric_label.lower()} across unmasked 1 ft spots.",
            )
        with st.expander("How to read this map", expanded=False):
            st.caption(MAP_TYPICAL_MIX_CAPTION)
            st.caption(MAP_MODEL_STEPS_CAPTION)
            st.caption(format_map_limitation_note(court_defaults))

    with c_panel:
        with st.container(border=True):
            st.subheader(
                format_spot_headline(
                    cell_info["shot_distance_ft"], int(cell_info["shot_value"])
                )
            )
            st.caption(format_spot_position(inspect_x_ft, inspect_y_ft))
            if not cell_info["in_mask"]:
                st.warning("Outside historical shot coverage for this selection.")
            else:
                st.metric("Make probability", f"{cell_info['p_make']:.1%}")
                st.metric("Expected points (xP)", f"{cell_info['xp']:.2f}")
                if gate_open:
                    st.metric(
                        "Outcome score",
                        f"{cell_info['v']:.2f}",
                        help=OUTCOME_SCORE_TOOLTIP,
                    )

            st.button(
                "Use this spot as a custom shot",
                key="use_spot_custom",
                on_click=_apply_spot_to_custom_shot,
                args=(inspect_x_ft, inspect_y_ft),
                width="stretch",
                wrap=True,
            )

        with st.expander("Exact position", expanded=False):
            st.number_input(
                "Left or right of the basket (ft, negative = left)",
                min_value=-25.0,
                max_value=25.0,
                step=0.5,
                format="%.1f",
                key="inspect_x_ft",
                help="Negative is left of the basket, positive is right, 0.0 ft is in line with it.",
            )
            st.number_input(
                "In front of the basket (ft, negative = behind it)",
                min_value=-5.0,
                max_value=42.0,
                step=0.5,
                format="%.1f",
                key="inspect_y_ft",
                help="0.0 ft is the centre of the hoop, -4.75 ft is the baseline behind it, and 42.0 ft is near half court.",
            )


def _render_about_tab(
    model: ExportedModel,
    court_defaults: dict[str, Any],
) -> None:
    st.caption(APP_DISCLAIMER)
    st.subheader("About the simulator")

    data_facts = model.manifest["data_facts"]
    versions = model.manifest["versions"]
    sa = model.clutch_shift_info["model_a"]
    sb = model.clutch_shift_info["model_b"]
    dm = court_defaults["density_mask"]

    st.markdown(
        f"""
This app compares late-game NBA field-goal options from an exported shot-make model and
a decision layer. Nothing is refitted at run time, and the app runs without raw data or
training binaries.

### Data and model
- Data: {data_facts['season']} ({data_facts['total_shots']:,} shots in {data_facts['n_games']:,} games: {data_facts['regular_shots']:,} normal-play shots, {data_facts['late_blowout_shots']:,} late-blowout shots, and {data_facts['clutch_shots']:,} clutch shots in {data_facts['n_clutch_games']:,} games).
- Normal-play baseline: Model A (XGBoost) was fitted on the {data_facts['regular_shots']:,} normal-play shots only. Optional player adjustments add shrunk single-season offsets from normal play; players with few shots are pulled toward zero, no player-specific clutch effect is modelled, and this app is not a ranking of players.
- Clutch shift (clutch gap): Clutch outcomes were used only to estimate a descriptive constant logit shift ({sa['shift_logit']:.3f} for Model A, {sb['shift_logit']:.3f} for Model B) when period >= 4, clock <= 300 s, and margin is within +/- 5. It bundles defense, fatigue, shot selection within a type, and shooter mix; it is descriptive, not causal.
- Outcome score (called V in docs): Equals P_ahead + tie_value * P_tied using shares of ending ahead, tied or behind after the shot, if nothing else happens. Default tie_value = 0.5 (coin-flip overtime) and the 24-second gate are modelling choices, not estimates.
- Map defaults: Uses the non-putback typical mix across {len(court_defaults['meta']['active_action_groups'])} shot types and hides 2 ft areas with fewer than {dm['min_shots_threshold']} normal-play shots ({dm['n_regular_shots_in_mask']:,} of all {dm['n_regular_shots_total']:,} normal-play shots, {dm['share_regular_shots_in_mask']:.2%}; {dm['n_regular_shots_outside_box']} fall outside the 2 ft grid).

### Limitations
- Single season only ({data_facts['season']}).
- Field goals only: no free throws, fouls, offensive rebounds, or turnovers.
- No opponent possession after the shot.
- No defender tracking or shot-clock data.
- Baseline make probability does not vary with time left or margin.
- Very short clock times may prevent some shot types in practice.
- The clutch shift covers close late-game shots only; the by-shot-type shift is exploratory (3PT minus 2PT = -0.117 logit, 95% bootstrap interval -0.217 to -0.017, close to zero).
- Sensitivity columns reflect uncertainty in the clutch adjustment only, not a confidence interval for make probability.
- Zone presets average over real-shot samples; custom points and map spots inherit Model A left-right differences unless smoothing is on.
- {format_map_limitation_note(court_defaults)}

### Repository and versions
- GitHub repository: {REPO_URL}
- Export schema: `{model.manifest['schema_version']}`
- Runtime: Python `{versions['python']}`, Streamlit `{st.__version__}`, XGBoost `{versions['xgboost']}`, pandas `{versions['pandas']}`, NumPy `{versions['numpy']}`
"""
    )


def main() -> None:
    """Render the Streamlit app."""
    _init_session_defaults()
    model, presets, court_defaults = load_app_resources()
    state, player_id, shift_mode, smooth, tie_value, time_gate_sec = _render_sidebar(model)

    tab_decision, tab_map, tab_about = st.tabs(["Decision", "Map", "About"])
    with tab_decision:
        _render_decision_tab(
            model=model,
            presets=presets,
            court_defaults=court_defaults,
            state=state,
            player_id=player_id,
            shift_mode=shift_mode,
            smooth=smooth,
            tie_value=tie_value,
            time_gate_sec=time_gate_sec,
        )
    with tab_map:
        _render_map_tab(
            model=model,
            court_defaults=court_defaults,
            state=state,
            player_id=player_id,
            shift_mode=shift_mode,
            smooth=smooth,
            tie_value=tie_value,
            time_gate_sec=time_gate_sec,
        )
    with tab_about:
        _render_about_tab(model=model, court_defaults=court_defaults)


if __name__ == "__main__":
    main()
