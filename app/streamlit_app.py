"""Phase 4c — Step 4: Streamlit Clutch Decision & Half-Court Simulator App.

Run locally from the repository root with::

    streamlit run app/streamlit_app.py
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
    MAP_TYPICAL_MIX_CAPTION,
    NO_SHOOTER_LABEL,
    REPO_URL,
    SENSITIVITY_CAPTION,
    SHIFT_MODE_OPTIONS,
    SHOOTER_NOTE,
    V_DEFINITION_CAPTION,
    build_preset_catalog,
    build_shooter_options,
    format_decision_summary,
    format_decision_table,
    format_exploratory_shift_note,
    format_flag_messages,
    format_map_limitation_note,
    format_sensitivity_table,
    is_rare_shot_type,
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
    render_court_heatmap,
)
from src.shot_geometry import compute_shot_distance_and_angle, compute_shot_value  # noqa: E402
from src.simulator_model import ExportedModel, load_export  # noqa: E402

st.set_page_config(
    page_title="NBA Clutch Decision Simulator",
    layout="wide",
)


@st.cache_resource
def load_app_resources() -> tuple[ExportedModel, dict[str, Any], dict[str, Any]]:
    """Load exported model, zone presets, and court defaults once per server process."""
    model = load_export()
    presets = load_zone_presets()
    court_defaults = load_court_defaults()
    return model, presets, court_defaults


@st.cache_data
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
    """Compute and cache half-court grids so changing metric does not recompute the model."""
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
    st.sidebar.header("Game Situation")

    period = st.sidebar.selectbox(
        "Period",
        options=[4, 5],
        index=0,
        format_func=lambda p: "4 (4th Quarter)" if p == 4 else "5 (Overtime)",
        key="period",
    )
    time_left_sec = st.sidebar.slider(
        "Time left (seconds)",
        min_value=0,
        max_value=300,
        value=8,
        step=1,
        key="time_left_sec",
    )
    margin = st.sidebar.slider(
        "Score margin before shot (offense - defense)",
        min_value=-10,
        max_value=10,
        value=-3,
        step=1,
        key="margin",
        help="Negative means the offense is trailing before taking the shot.",
    )

    shooter_labels, name_to_id = build_shooter_options(model.player_table)
    selected_shooter = st.sidebar.selectbox(
        "Shooter (optional)",
        options=shooter_labels,
        index=0,
        key="shooter",
        help="Alphabetical list of players with regular-season baseline offsets.",
    )
    player_id = name_to_id.get(selected_shooter)

    shift_label = st.sidebar.radio(
        "Clutch shift",
        options=list(SHIFT_MODE_OPTIONS.keys()),
        index=1,
        key="shift_mode_label",
        help=(
            "Descriptive logit-scale gap observed on close late-game shots, bundling "
            "defense, fatigue, shot selection within a type, and shooter composition."
        ),
    )
    shift_mode = SHIFT_MODE_OPTIONS[shift_label]
    if shift_mode == "by_shot_type":
        st.sidebar.caption(format_exploratory_shift_note(model.clutch_shift_info))

    smooth = st.sidebar.checkbox(
        "Smooth left-right differences",
        value=True,
        key="smooth_lr",
        help=(
            "Averages each court point with its mirror image across the centre line "
            "to smooth out Model A's local tree splits between left and right sides."
        ),
    )

    with st.sidebar.expander("Advanced"):
        tie_value = st.slider(
            "Overtime value (tie_value)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="tie_value",
        )
        st.caption(
            "Weight given to ending tied after the shot in V. Default 0.5 treats "
            "overtime as a coin flip (a modelling choice, not estimated from data)."
        )
        time_gate_sec = st.number_input(
            "Decision time gate (seconds)",
            min_value=0,
            max_value=300,
            value=24,
            step=1,
            key="time_gate_sec",
        )
        st.caption(
            "End-state outcomes and V are shown only when time left is at or below "
            "this threshold (default 24 s, about one shot clock; a modelling choice)."
        )

    state = GameState(
        time_left_sec=float(time_left_sec),
        margin=int(margin),
        period=int(period),
    )
    return state, player_id, shift_mode, bool(smooth), float(tie_value), int(time_gate_sec)


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
    st.subheader("Compare Shot Options")

    preset_labels, label_to_key = build_preset_catalog(presets)
    default_labels = [
        str(presets[k]["label"]) for k in DEFAULT_PRESET_KEYS if k in presets
    ]
    selected_labels = st.multiselect(
        "Zone presets",
        options=preset_labels,
        default=default_labels,
        key="selected_presets",
        help="Each preset averages predictions across a fixed sample of real normal-play shots from that zone.",
    )
    selected_options: list[str] = [label_to_key[lbl] for lbl in selected_labels]

    shot_type_keys = [TYPICAL_MIX_KEY, *get_active_shot_types(court_defaults)]
    use_custom = st.checkbox("Include custom shot", value=False, key="use_custom_shot")

    eval_presets = dict(presets)
    custom_xy: tuple[float, float] | None = None
    if use_custom:
        c1, c2, c3 = st.columns(3)
        with c1:
            custom_x = st.number_input(
                "LOC_X (tenths of ft, -250 to 250)",
                min_value=-250,
                max_value=250,
                value=0,
                step=5,
                key="custom_loc_x",
            )
        with c2:
            custom_y = st.number_input(
                "LOC_Y (tenths of ft, -50 to 420)",
                min_value=-50,
                max_value=420,
                value=250,
                step=5,
                key="custom_loc_y",
            )
        with c3:
            custom_shot_type = st.selectbox(
                "Custom shot type",
                options=shot_type_keys,
                index=0,
                format_func=lambda k: ACTION_GROUP_LABELS[k],
                key="custom_shot_type",
            )

        dist_ft, angle_deg = compute_shot_distance_and_angle(
            float(custom_x), float(custom_y), is_putback=0, clamp_distance=True
        )
        shot_val = compute_shot_value(float(custom_x), float(custom_y))
        custom_xy = (float(custom_x), float(custom_y))

        m1, m2, m3 = st.columns(3)
        m1.metric("Derived distance", f"{dist_ft:.0f} ft")
        m2.metric("Derived angle", f"{angle_deg:.1f} deg")
        m3.metric("Shot value", f"{shot_val}PT")

        if is_rare_shot_type(court_defaults, dist_ft, custom_shot_type):
            st.caption(CUSTOM_SHOT_RARE_WARNING)

        custom_key = "__custom_shot__"
        custom_label = f"Custom ({ACTION_GROUP_LABELS[custom_shot_type]})"
        eval_presets[custom_key] = build_custom_shot_preset(
            loc_x=float(custom_x),
            loc_y=float(custom_y),
            shot_type=custom_shot_type,
            smooth=smooth,
            court_defaults=court_defaults,
            label=custom_label,
        )
        selected_options.append(custom_key)

    if player_id is not None:
        st.info(SHOOTER_NOTE)

    if not selected_options:
        st.info("Select at least one zone preset or enable a custom shot to see results.")
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

    for severity, msg in format_flag_messages(result.meta):
        if severity == "warning":
            st.warning(msg)
        else:
            st.info(msg)

    st.write(format_decision_summary(result))

    display_df = format_decision_table(result)
    st.dataframe(display_df, hide_index=True, width="stretch")
    if result.meta["gate_open"]:
        st.caption(V_DEFINITION_CAPTION)

        st.subheader("End-state outcome breakdown")
        bar_fig = render_end_state_bar(result)
        st.pyplot(bar_fig, clear_figure=True, width="content")
        plt.close(bar_fig)

    with st.expander("Sensitivity to the clutch shift"):
        st.caption(SENSITIVITY_CAPTION)
        sens_df = format_sensitivity_table(result)
        st.dataframe(sens_df, hide_index=True, width="stretch")

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
    use_custom: bool,
    custom_xy: tuple[float, float] | None,
) -> None:
    st.caption(APP_DISCLAIMER)
    st.subheader("Half-Court Map")

    gate_open = state.time_left_sec <= float(time_gate_sec)
    metric_label_to_key = {
        "Make probability": "p_make",
        "Expected points (xP)": "xp",
        "End-state value (V)": "v",
    }
    if gate_open:
        metric_options = list(metric_label_to_key.keys())
        default_metric_idx = 2
    else:
        metric_options = ["Make probability", "Expected points (xP)"]
        default_metric_idx = 0
        if st.session_state.get("map_metric") == "End-state value (V)":
            st.session_state["map_metric"] = "Make probability"
        st.info(
            f"End-state value (V) is disabled when the decision gate is closed "
            f"(more than {time_gate_sec} s left)."
        )

    shot_type_keys = [TYPICAL_MIX_KEY, *get_active_shot_types(court_defaults)]

    c_ctrl, c_plot = st.columns([1.0, 2.0])
    with c_ctrl:
        metric_label = st.radio(
            "Map metric",
            options=metric_options,
            index=default_metric_idx if "map_metric" not in st.session_state else (
                metric_options.index(st.session_state["map_metric"])
                if st.session_state["map_metric"] in metric_options
                else default_metric_idx
            ),
            key="map_metric",
        )
        map_shot_type = st.selectbox(
            "Shot type on map",
            options=shot_type_keys,
            index=0,
            format_func=lambda k: ACTION_GROUP_LABELS[k],
            key="map_shot_type",
            help=(
                "Typical mix averages over the non-putback shot types observed at each "
                "distance; choosing one shot type masks distances where its share is "
                "below 10 out of 1,000."
            ),
        )

        st.markdown("**Cell / shot inspector**")
        default_ix = int(round(custom_xy[0])) if (use_custom and custom_xy is not None) else 225
        default_iy = int(round(custom_xy[1])) if (use_custom and custom_xy is not None) else 5
        inspect_x = st.number_input(
            "Inspect LOC_X (tenths of ft)",
            min_value=-250,
            max_value=250,
            value=default_ix,
            step=10,
            key="inspect_loc_x",
        )
        inspect_y = st.number_input(
            "Inspect LOC_Y (tenths of ft)",
            min_value=-50,
            max_value=420,
            value=default_iy,
            step=10,
            key="inspect_loc_y",
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

    highlight_xy = (
        custom_xy
        if (use_custom and custom_xy is not None)
        else (float(inspect_x), float(inspect_y))
    )
    cell_info = lookup_cell(heatmap, float(inspect_x), float(inspect_y))

    with c_ctrl:
        st.caption(
            f"Nearest 1 ft cell centre: ({cell_info['cell_x']:.0f}, {cell_info['cell_y']:.0f}) "
            f"| {cell_info['shot_distance_ft']:.0f} ft, {cell_info['shot_value']}PT"
        )
        if not cell_info["in_mask"]:
            st.warning("Selected cell is outside the density or shot-type mask.")
        else:
            st.metric("Cell make probability", f"{cell_info['p_make']:.1%}")
            st.metric("Cell expected points (xP)", f"{cell_info['xp']:.2f}")
            if gate_open:
                st.metric("Cell end-state value (V)", f"{cell_info['v']:.3f}")

        if player_id is not None:
            st.caption(SHOOTER_NOTE)

    with c_plot:
        metric_key = metric_label_to_key[metric_label]
        fig = render_court_heatmap(
            heatmap,
            metric=metric_key,  # type: ignore[arg-type]
            highlight_xy=highlight_xy,
        )
        st.pyplot(fig, clear_figure=True, width="content")
        plt.close(fig)
        st.caption(MAP_TYPICAL_MIX_CAPTION)
        st.caption(format_map_limitation_note(court_defaults))


def _render_about_tab(
    model: ExportedModel,
    court_defaults: dict[str, Any],
) -> None:
    st.caption(APP_DISCLAIMER)
    st.subheader("About the Simulator")

    data_facts = model.manifest["data_facts"]
    versions = model.manifest["versions"]
    sa = model.clutch_shift_info["model_a"]
    sb = model.clutch_shift_info["model_b"]
    dm = court_defaults["density_mask"]

    st.markdown(
        f"""
This app compares late-game NBA field-goal options using an exported baseline shot-make
model and a pure-Python decision layer. Nothing is refitted at run time, and the app
runs strictly from portable exported artifacts without raw game data or training binaries.

### Data & Model
- **Data:** {data_facts['season']} ({data_facts['total_shots']:,} total shots across {data_facts['n_games']:,} games: {data_facts['regular_shots']:,} normal-play `REGULAR` shots, {data_facts['late_blowout_shots']:,} late-blowout shots, and {data_facts['clutch_shots']:,} `CLUTCH` shots across {data_facts['n_clutch_games']:,} games).
- **Normal-play baseline (Model A & Model B):** Model A is an XGBoost classifier fitted on the {data_facts['regular_shots']:,} normal-play (`REGULAR`) shots only. Optional shooter adjustments (Model B) apply empirical-Bayes shrunk regular-play player offsets; players with few shots are pulled toward zero, no player-specific clutch effect is modelled, and this app is not a ranking of players.
- **Descriptive clutch shift (clutch gap):** Clutch outcomes were used **only** to estimate a descriptive constant logit-scale clutch shift (`{sa['shift_logit']:.3f}` logit for Model A; `{sb['shift_logit']:.3f}` logit for Model B) inside the clutch window (`period >= 4`, `time_left_sec <= 300`, `abs(margin) <= 5`). It bundles defensive intensity, fatigue, shot selection within a shot type, and shooter composition — it is descriptive, not a causal effect.
- **End-state value `V`:** `V = P_ahead + tie_value * P_tied`, using probabilities of ending **ahead, tied or behind after the shot, if nothing else happens**. Default `tie_value = 0.5` treats overtime as a coin flip, and the 24-second decision gate is a modelling choice (neither is estimated from data).
- **Half-court map defaults:** Uses the distance-based typical mix of non-putback `REGULAR` shots across {len(court_defaults['meta']['active_action_groups'])} shot types and masks 2 ft cells with fewer than {dm['min_shots_threshold']} `REGULAR` shots (covering {dm['share_regular_shots_in_mask']:.2%} of `REGULAR` shots).

### Limitations
- Single season only ({data_facts['season']}).
- Field goals only: no free throws, fouls on the shot, offensive rebounds, or turnovers.
- No opponent possession after the shot.
- No defender tracking or shot-clock data.
- Baseline make probability does not depend on time remaining or score margin (Phase 3 found no detectable effect).
- Very short clock times (a few seconds) may not allow some shot types in practice.
- The clutch shift was estimated on close late-game shots only; the by-shot-type shift is exploratory (`3PT minus 2PT = -0.117` logit, `95% bootstrap interval -0.217 to -0.017`, interval close to zero).
- Sensitivity bounds reflect sensitivity to the clutch shift only, not a confidence interval for make probability.
- Zone presets average over real-shot samples; custom points and map cells inherit Model A's local left-right differences unless smoothing is on.
- {format_map_limitation_note(court_defaults)}

### Repository & Versions
- **GitHub repository:** {REPO_URL}
- **Export schema:** `{model.manifest['schema_version']}`
- **Runtime versions:** Python `{versions['python']}`, Streamlit `{st.__version__}`, XGBoost `{versions['xgboost']}`, pandas `{versions['pandas']}`, NumPy `{versions['numpy']}`
"""
    )


def main() -> None:
    """Render the Streamlit app."""
    st.title("NBA Clutch Decision Simulator")
    model, presets, court_defaults = load_app_resources()
    state, player_id, shift_mode, smooth, tie_value, time_gate_sec = _render_sidebar(model)

    tab_decision, tab_map, tab_about = st.tabs(["Decision", "Map", "About"])
    with tab_decision:
        use_custom, custom_xy = _render_decision_tab(
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
            use_custom=use_custom,
            custom_xy=custom_xy,
        )
    with tab_about:
        _render_about_tab(model=model, court_defaults=court_defaults)


if __name__ == "__main__":
    main()
