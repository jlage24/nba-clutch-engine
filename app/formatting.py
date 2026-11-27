"""Pure formatting, wording, selection-parsing, and chart helpers for the Streamlit app.

No Streamlit runtime dependency; all functions are deterministic and unit-testable
without ``data/`` or ``models/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Literal

import altair as alt
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision_layer import DecisionResult, GameState  # noqa: E402
from src.heatmap import (  # noqa: E402
    HEATMAP_X_MAX,
    HEATMAP_X_MIN,
    HEATMAP_Y_MAX,
    HEATMAP_Y_MIN,
    MIN_GROUP_WEIGHT_PER_1000,
    TYPICAL_MIX_KEY,
    HeatmapResult,
    build_heatmap_dataframe,
    get_court_polylines,
    get_distance_weights,
)

REPO_URL = "https://github.com/jlage24/nba-clutch-engine"
APP_DISCLAIMER = (
    "Descriptive model from one season of public data. No defender or shot-clock data. "
    "Field goals only. Not coaching advice."
)
NO_SHOOTER_LABEL = "No shooter"
SHOOTER_NOTE = (
    "Shooter adjustments are shrunk single-season estimates from normal play. "
    "Low-volume players pull toward zero, and this app is not a ranking of players."
)
CUSTOM_SHOT_RARE_WARNING = (
    "This shot type is very rare at this distance; the model is extrapolating."
)
MAP_TYPICAL_MIX_CAPTION = (
    "Typical mix averages the non-putback shot types seen at that distance; real shots "
    "at the same spot differ."
)
MAP_MODEL_STEPS_CAPTION = (
    "Small steps between neighbouring spots come from the model (distance is measured "
    "in whole feet and the model splits on thresholds), not from the data."
)
OUTCOME_SCORE_TOOLTIP = (
    "Ahead counts 1, tied counts the overtime value, behind counts 0 (called V in docs)."
)
END_STATE_CAPTION = (
    "Shares show ahead, tied or behind after the shot, if nothing else happens."
)
SENSITIVITY_CAPTION = (
    "Sensitivity to the clutch shift only, not a confidence interval for make probability."
)

PRESET_ORDER = [
    "restricted_area",
    "putback",
    "paint",
    "short_mid_range",
    "long_mid_range",
    "corner_three",
    "above_the_break_3",
]
DEFAULT_PRESET_KEYS = [
    "corner_three",
    "above_the_break_3",
    "long_mid_range",
    "restricted_area",
]

EXAMPLE_SCENARIOS: list[tuple[str, str, int, int]] = [
    ("Down 3 · 8 s", "4th quarter", 8, -3),
    ("Down 2 · 8 s", "4th quarter", 8, -2),
    ("Down 1 · 6 s", "4th quarter", 6, -1),
    ("Tied · 5 s", "4th quarter", 5, 0),
]

PERIOD_OPTIONS = {
    "4th quarter": 4,
    "Overtime": 5,
}

SHIFT_MODE_OPTIONS = {
    "Off": "none",
    "Overall (default)": "overall",
    "By shot type (exploratory)": "by_shot_type",
}


def format_state_headline(state: GameState) -> str:
    """Format the live game state as a concise page headline."""
    if state.margin < 0:
        margin_text = f"Down {abs(state.margin)}"
    elif state.margin == 0:
        margin_text = "Tied"
    else:
        margin_text = f"Up {state.margin}"

    if state.period == 4:
        period_text = "Period 4"
    elif state.period == 5:
        period_text = "Overtime"
    else:
        period_text = f"Period {state.period}"

    return f"{margin_text} · {int(round(state.time_left_sec))} s left · {period_text}"


def format_spot_headline(distance_ft: float, shot_value: int) -> str:
    """Format inspected spot distance and shot value in plain visitor language."""
    shot_kind = "three-pointer" if int(shot_value) == 3 else "two-pointer"
    return f"{int(round(distance_ft))} ft from the basket · {shot_kind}"


def format_spot_position(x_ft: float, y_ft: float) -> str:
    """Format court coordinates relative to the hoop centre in plain visitor language."""
    x_rounded = round(float(x_ft), 1)
    y_rounded = round(float(y_ft), 1)
    if abs(x_rounded) < 0.05:
        x_text = "in line with the basket"
    elif x_rounded < 0:
        x_text = f"{abs(x_rounded):.1f} ft left of the basket"
    else:
        x_text = f"{x_rounded:.1f} ft right of the basket"

    if abs(y_rounded) < 0.05:
        y_text = "in line with the hoop"
    elif y_rounded < 0:
        y_text = f"{abs(y_rounded):.1f} ft behind it"
    else:
        y_text = f"{y_rounded:.1f} ft in front of it"

    return f"{x_text} · {y_text}"


def parse_map_selection(selection_state: Any) -> tuple[float, float] | None:
    """Convert an Altair/Streamlit selection payload into clamped coordinates in tenths of a foot."""
    if selection_state is None:
        return None

    selection_dict: Any = None
    if isinstance(selection_state, Mapping):
        selection_dict = selection_state.get("selection", selection_state)
    elif hasattr(selection_state, "selection"):
        selection_dict = getattr(selection_state, "selection")

    if not isinstance(selection_dict, Mapping):
        return None

    points = selection_dict.get("spot")
    if not isinstance(points, (list, tuple)) or len(points) == 0:
        return None

    last = points[-1]
    if not isinstance(last, Mapping):
        return None
    if "x_ft" not in last or "y_ft" not in last:
        return None

    try:
        x_tenths = round(float(last["x_ft"]) * 10.0, 1)
        y_tenths = round(float(last["y_ft"]) * 10.0, 1)
    except (TypeError, ValueError):
        return None

    x_clamped = min(max(x_tenths, HEATMAP_X_MIN), HEATMAP_X_MAX)
    y_clamped = min(max(y_tenths, HEATMAP_Y_MIN), HEATMAP_Y_MAX)
    return float(x_clamped), float(y_clamped)


def build_preset_catalog(presets: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    """Return ordered preset labels and label -> preset_key lookup."""
    ordered_keys = [k for k in PRESET_ORDER if k in presets]
    extra_keys = sorted(k for k in presets if k not in ordered_keys)
    labels: list[str] = []
    label_to_key: dict[str, str] = {}
    for key in ordered_keys + extra_keys:
        label = str(presets[key].get("label", key))
        labels.append(label)
        label_to_key[label] = key
    return labels, label_to_key


def build_shooter_options(player_table: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return alphabetically sorted player names with 'No shooter' first (no offsets shown)."""
    pairs = sorted(
        (
            (str(row.PLAYER_NAME), str(int(row.PLAYER_ID)))
            for row in player_table[["PLAYER_ID", "PLAYER_NAME"]].itertuples(index=False)
        ),
        key=lambda item: (item[0].casefold(), item[0]),
    )
    labels = [NO_SHOOTER_LABEL] + [name for name, _ in pairs]
    name_to_id = {name: pid for name, pid in pairs}
    return labels, name_to_id


def is_rare_shot_type(
    court_defaults: dict[str, Any],
    distance_ft: float,
    shot_type: str,
) -> bool:
    """Return True when a specific ACTION_GROUP has weight < 10/1,000 at distance_ft."""
    if shot_type == TYPICAL_MIX_KEY:
        return False
    weights = get_distance_weights(court_defaults, int(distance_ft))
    return int(weights.get(shot_type, 0)) < MIN_GROUP_WEIGHT_PER_1000


def format_exploratory_shift_note(clutch_shift_info: dict[str, Any]) -> str:
    """Plain-language caution for the exploratory 2PT vs. 3PT clutch shift cut."""
    diff_info = clutch_shift_info["exploratory"]["model_b_3pt_minus_2pt"]
    return (
        "Exploratory cut: 3PT minus 2PT shift is "
        f"{diff_info['diff_logit']:.3f} logit (95% bootstrap interval "
        f"{diff_info['ci_lower_logit']:.3f} to {diff_info['ci_upper_logit']:.3f}; "
        "close to zero)."
    )


def format_map_limitation_note(court_defaults: dict[str, Any]) -> str:
    """Format the rim and within-area bunching limitation notes from court_defaults.json."""
    pc = court_defaults["checks"]["cell_level_typical_mix_vs_actual"]
    cd = court_defaults["checks"]["cell_centre_diagnostic"]
    mc = pc["max_diff_cell"]
    return (
        f"Within about 4 ft of the rim, the typical mix can understate real shots because "
        f"dunk share varies by angle while the mix depends only on distance (largest 2 ft "
        f"zone gap: {mc['n_shots']} shots, {mc['dunk_share']:.1%} dunks, mix "
        f"{mc['mean_p_a_typical_mix']:.1%} vs. actual {mc['mean_p_a_actual']:.1%}; "
        f"weighted mean gap across {pc['n_cells_checked']} zones is "
        f"{pc['weighted_mean_abs_diff']:.4f}). Also, a pointwise map cannot show how "
        f"shots bunch inside a 2 ft area (evaluating real shots against area centres "
        f"gives weighted mean gap {cd['weighted_mean_abs_diff']:.4f}, up to "
        f"{cd['max_abs_diff']:.3f} across {cd['n_cells_ge_threshold']} rim areas)."
    )


def format_flag_messages(meta: dict[str, Any]) -> list[tuple[str, str]]:
    """Map decision-layer flags to plain-language (severity, message) pairs."""
    messages: list[tuple[str, str]] = []
    flags = list(meta.get("flags", []))
    gate_sec = float(meta.get("time_gate_sec", 24.0))

    for flag in flags:
        if flag == "decision_gate_closed":
            messages.append(
                (
                    "info",
                    f"Gate closed: more than {gate_sec:g} s left, so this is not treated "
                    "as the last shot. Only make probability and xP are shown.",
                )
            )
        elif flag == "no_option_changes_end_state":
            messages.append(
                (
                    "warning",
                    "No option changes the end state: every shot has the same outcome "
                    "score, so options are ordered by xP.",
                )
            )
        elif flag == "shift_not_applied_outside_clutch":
            messages.append(
                (
                    "info",
                    "Outside the clutch window (period >= 4, <= 300 s left, margin "
                    "within +/- 5): no clutch shift is applied.",
                )
            )
        elif flag == "unknown_player_offset_zero":
            messages.append(
                (
                    "info",
                    "Selected shooter has no regular-season offset; using league "
                    "baseline (offset 0).",
                )
            )
    return messages


def _sorted_result_table(result: DecisionResult) -> pd.DataFrame:
    df = result.table.copy()
    if result.meta["gate_open"]:
        return df.sort_values(["rank_v", "label"], ascending=[True, True]).reset_index(drop=True)
    return df.sort_values(["xp", "label"], ascending=[False, True]).reset_index(drop=True)


def format_decision_table(result: DecisionResult) -> pd.DataFrame:
    """Format the compact decision-layer table (5 columns when gate is open)."""
    sorted_df = _sorted_result_table(result)
    gate_open = bool(result.meta["gate_open"])

    if not gate_open:
        return pd.DataFrame(
            {
                "Option": sorted_df["label"],
                "Make prob.": sorted_df["p_make"].astype(float) * 100.0,
                "xP": sorted_df["xp"].astype(float),
            }
        )

    return pd.DataFrame(
        {
            "Option": sorted_df["label"],
            "Make prob.": sorted_df["p_make"].astype(float) * 100.0,
            "xP": sorted_df["xp"].astype(float),
            "Outcome score": sorted_df["v"].astype(float),
            "Rank": [int(v) for v in sorted_df["rank_v"]],
        }
    )


def format_sensitivity_table(result: DecisionResult) -> pd.DataFrame:
    """Format clutch-shift sensitivity columns (omits outcome score when gate closed)."""
    sorted_df = _sorted_result_table(result)
    out = pd.DataFrame(
        {
            "Option": sorted_df["label"],
            "Make (no shift)": sorted_df["p_make_no_shift"].astype(float) * 100.0,
            "Make (low)": sorted_df["p_make_shift_low"].astype(float) * 100.0,
            "Make (point)": sorted_df["p_make"].astype(float) * 100.0,
            "Make (high)": sorted_df["p_make_shift_high"].astype(float) * 100.0,
        }
    )
    if result.meta["gate_open"]:
        out["Score (no shift)"] = sorted_df["v_no_shift"].astype(float)
        out["Score (low)"] = sorted_df["v_shift_low"].astype(float)
        out["Score (point)"] = sorted_df["v"].astype(float)
        out["Score (high)"] = sorted_df["v_shift_high"].astype(float)
    return out


def classify_summary_variant(result: DecisionResult) -> str:
    """Return which summary-sentence variant applies to ``result``."""
    meta = result.meta
    if not meta["gate_open"]:
        return "closed_gate"
    if meta.get("no_option_changes_end_state"):
        return "no_option_changes_end_state"

    top = _sorted_result_table(result).iloc[0]
    p_ahead = float(top["p_ahead"])
    p_tied = float(top["p_tied"])
    if p_ahead > 0.0:
        return "can_lead"
    if p_tied > 0.0:
        return "can_tie"
    return "neither"


def format_decision_answer_parts(result: DecisionResult) -> tuple[str, str | None, str | None]:
    """Return ``(main_sentence, next_option_line, stability_line)`` from table and meta."""
    sorted_df = _sorted_result_table(result)
    meta = result.meta
    top = sorted_df.iloc[0]
    top_label = str(top["label"])
    variant = classify_summary_variant(result)

    has_second = len(sorted_df) >= 2
    second = sorted_df.iloc[1] if has_second else None

    if variant == "closed_gate":
        t_left = float(meta["state"]["time_left_sec"])
        top_xp = float(top["xp"])
        main = (
            f"With {t_left:g} s left, outcome scores are hidden. "
            f"{top_label} leads in xP at {top_xp:.2f} ({float(top['p_make']):.1%} make)."
        )
        next_line = (
            f"Next: {second['label']}, {float(second['xp']):.2f} xP against {top_xp:.2f}."
            if second is not None
            else None
        )
        return main, next_line, None

    stability_line = (
        "The best option stays the same with or without the clutch adjustment."
        if meta["ranking_robust"]
        else "The best option changes with the clutch adjustment."
    )

    if variant == "no_option_changes_end_state":
        top_xp = float(top["xp"])
        main = (
            f"No selected shot can tie or take the lead (outcome score "
            f"{float(top['v']):.2f} for all). {top_label} leads in xP at "
            f"{top_xp:.2f} ({float(top['p_make']):.1%} make)."
        )
        next_line = (
            f"Next: {second['label']}, {float(second['xp']):.2f} xP against {top_xp:.2f}."
            if second is not None
            else None
        )
        return main, next_line, stability_line

    top_v = float(top["v"])
    if variant == "can_lead":
        main = (
            f"{top_label} is the best option. It puts the offense ahead "
            f"{float(top['p_ahead']):.1%} of the time (outcome score {top_v:.2f})."
        )
    elif variant == "can_tie":
        main = (
            f"{top_label} is the best option. It ties the game "
            f"{float(top['p_tied']):.1%} of the time (outcome score {top_v:.2f})."
        )
    else:
        main = (
            f"{top_label} is the best option. It leaves the offense behind "
            f"{float(top['p_behind']):.1%} of the time (outcome score {top_v:.2f})."
        )

    next_line = (
        f"Next: {second['label']}, outcome score {float(second['v']):.2f} against {top_v:.2f}."
        if second is not None
        else None
    )
    return main, next_line, stability_line


def format_decision_summary(result: DecisionResult) -> str:
    """Return the combined fixed-template answer block text as one string."""
    parts = format_decision_answer_parts(result)
    return " ".join(part for part in parts if part)


def build_altair_court_chart(
    heatmap: HeatmapResult,
    metric: Literal["p_make", "xp", "v"],
    selected_xy_tenths: tuple[float, float] | None = None,
    width: int = 550,
    height: int = 517,
) -> alt.LayerChart:
    """Build an interactive layered Altair half-court chart with equal x/y scales."""
    df_spots = build_heatmap_dataframe(heatmap, metric)
    if not df_spots.empty:
        df_spots["position"] = [
            format_spot_position(xf, yf)
            for xf, yf in zip(df_spots["x_ft"], df_spots["y_ft"])
        ]
    else:
        df_spots["position"] = pd.Series(dtype=str)

    court_df = get_court_polylines(units="feet")
    solid_lines = court_df[~court_df["dashed"]].copy()
    dashed_lines = court_df[court_df["dashed"]].copy()

    if metric == "p_make":
        metric_title = "Make probability"
        legend_fmt = ".1%"
        default_min, default_max = 0.15, 0.85
    elif metric == "xp":
        metric_title = "Expected points (xP)"
        legend_fmt = ".2f"
        default_min, default_max = 0.50, 1.75
    else:
        metric_title = "Outcome score"
        legend_fmt = ".2f"
        default_min, default_max = 0.00, 1.00

    if not df_spots.empty:
        vmin = min(default_min, float(df_spots["value"].min()))
        vmax = max(default_max, float(df_spots["value"].max()))
        if metric == "v":
            vmin = max(0.0, float(df_spots["value"].min()))
            vmax = min(1.0, max(float(df_spots["value"].max()), vmin + 1e-3))
    else:
        vmin, vmax = default_min, default_max

    x_scale = alt.Scale(domain=[-25.0, 25.0], nice=False, zero=False)
    y_scale = alt.Scale(domain=[-4.75, 42.25], nice=False, zero=False)

    rect_layer = (
        alt.Chart(df_spots)
        .mark_rect(clip=True, strokeWidth=0)
        .encode(
            x=alt.X("x0:Q", scale=x_scale, axis=None),
            x2="x1:Q",
            y=alt.Y("y0:Q", scale=y_scale, axis=None),
            y2="y1:Q",
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="viridis", domain=[vmin, vmax]),
                legend=alt.Legend(
                    title=metric_title,
                    orient="bottom",
                    direction="horizontal",
                    gradientLength=200,
                    format=legend_fmt,
                ),
            ),
        )
    )

    solid_layer = (
        alt.Chart(solid_lines)
        .mark_line(color="#cbd5e1", strokeWidth=1.5, tooltip=None)
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            detail="part:N",
            order="order:Q",
        )
    )
    dashed_layer = (
        alt.Chart(dashed_lines)
        .mark_line(color="#cbd5e1", strokeWidth=1.5, strokeDash=[5, 4], tooltip=None)
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            detail="part:N",
            order="order:Q",
        )
    )

    layers: list[alt.Chart] = [rect_layer, solid_layer, dashed_layer]

    if selected_xy_tenths is not None:
        ix = int(np.argmin(np.abs(heatmap.x_centers - float(selected_xy_tenths[0]))))
        iy = int(np.argmin(np.abs(heatmap.y_centers - float(selected_xy_tenths[1]))))
        sx = float(heatmap.x_centers[ix]) / 10.0
        sy = float(heatmap.y_centers[iy]) / 10.0
        sel_df = pd.DataFrame(
            [{"x0": sx - 0.5, "x1": sx + 0.5, "y0": sy - 0.5, "y1": sy + 0.5}]
        )
        layers.append(
            alt.Chart(sel_df)
            .mark_rect(fill=None, stroke="#f8fafc", strokeWidth=3.5, tooltip=None)
            .encode(
                x=alt.X("x0:Q", scale=x_scale),
                x2="x1:Q",
                y=alt.Y("y0:Q", scale=y_scale),
                y2="y1:Q",
            )
        )
        layers.append(
            alt.Chart(sel_df)
            .mark_rect(fill=None, stroke="#cd3a29", strokeWidth=2.0, tooltip=None)
            .encode(
                x=alt.X("x0:Q", scale=x_scale),
                x2="x1:Q",
                y=alt.Y("y0:Q", scale=y_scale),
                y2="y1:Q",
            )
        )

    spot_sel = alt.selection_point(
        name="spot",
        fields=["x_ft", "y_ft"],
        on="click",
        clear=False,
        toggle=False,
    )
    hit_layer = (
        alt.Chart(df_spots)
        .mark_rect(opacity=0.001, clip=True, strokeWidth=0)
        .encode(
            x=alt.X("x0:Q", scale=x_scale),
            x2="x1:Q",
            y=alt.Y("y0:Q", scale=y_scale),
            y2="y1:Q",
            tooltip=[
                alt.Tooltip("position:N", title="Spot"),
                alt.Tooltip("distance_ft:Q", title="From basket (ft)", format=".0f"),
                alt.Tooltip("shot_kind:N", title="Shot"),
                alt.Tooltip("value_label:N", title=metric_title),
            ],
        )
        .add_params(spot_sel)
    )
    layers.append(hit_layer)

    return (
        alt.layer(*layers)
        .properties(
            width=width,
            height=height,
            autosize=alt.AutoSizeParams(type="pad", contains="padding"),
        )
        .configure(background="#090f1a", padding={"left": 4, "right": 4, "top": 6, "bottom": 6})
        .configure_view(strokeWidth=0, fill="#111928")
        .configure_legend(labelColor="#f1f5f9", titleColor="#f1f5f9", titleFontSize=12, labelFontSize=11)
    )


def render_end_state_bar(
    result: DecisionResult,
    bg_color: str = "#090f1a",
    text_color: str = "#f1f5f9",
) -> matplotlib.figure.Figure:
    """Plot horizontal stacked bars of Ahead, Tied, and Behind percentages with labels."""
    sorted_df = _sorted_result_table(result).iloc[::-1].reset_index(drop=True)
    labels = sorted_df["label"].tolist()
    ahead = sorted_df["p_ahead"].to_numpy(dtype=float) * 100.0
    tied = sorted_df["p_tied"].to_numpy(dtype=float) * 100.0
    behind = sorted_df["p_behind"].to_numpy(dtype=float) * 100.0

    height = max(2.2, 0.56 * len(labels) + 0.9)
    fig, ax = plt.subplots(figsize=(5.6, height), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    y = list(range(len(labels)))

    c_ahead = "#38bdf8"
    c_tied = "#fbbf24"
    c_behind = "#64748b"

    ax.barh(y, ahead, height=0.58, color=c_ahead, label="Ahead")
    ax.barh(y, tied, left=ahead, height=0.58, color=c_tied, label="Tied")
    ax.barh(y, behind, left=ahead + tied, height=0.58, color=c_behind, label="Behind")

    for row_idx in range(len(labels)):
        segments = [
            (0.0, ahead[row_idx], "#090f1a"),
            (ahead[row_idx], tied[row_idx], "#090f1a"),
            (ahead[row_idx] + tied[row_idx], behind[row_idx], "#f1f5f9"),
        ]
        for start, width, lbl_color in segments:
            if width >= 14.0:
                ax.text(
                    start + 0.5 * width,
                    row_idx,
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    color=lbl_color,
                    fontsize=8.5,
                    fontweight="bold",
                )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=text_color, fontsize=9.5)
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=text_color, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=text_color)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    for txt in legend.get_texts():
        txt.set_color(text_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#26344d")
    ax.spines["bottom"].set_color("#26344d")
    fig.tight_layout()
    return fig
