"""Pure formatting, wording, and template helpers for the Streamlit simulator app.

No Streamlit runtime dependency; all functions are deterministic and unit-testable
without ``data/`` or ``models/``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision_layer import DecisionResult  # noqa: E402
from src.heatmap import (  # noqa: E402
    MIN_GROUP_WEIGHT_PER_1000,
    TYPICAL_MIX_KEY,
    get_distance_weights,
)

REPO_URL = "https://github.com/jlage24/nba-clutch-engine"
APP_DISCLAIMER = (
    "Descriptive model from one season of public data. No defender or shot-clock data. "
    "Field goals only. Not coaching advice."
)
NO_SHOOTER_LABEL = "No shooter"
SHOOTER_NOTE = (
    "Shooter adjustments are shrunk single-season estimates from normal play "
    "(players with few shots are pulled toward zero); no player-specific clutch effect "
    "is modelled, and this app is not a ranking of players."
)
CUSTOM_SHOT_RARE_WARNING = (
    "This shot type is very rare at this distance; the model is extrapolating."
)
MAP_TYPICAL_MIX_CAPTION = (
    "Typical mix is the average over the shot types seen at that distance; real shots at "
    "the same spot differ. Small steps between neighbouring cells come from the model "
    "(distance is measured in whole feet and the model splits on thresholds), not from the data."
)
V_DEFINITION_CAPTION = (
    'V counts "ahead" as 1, "tied" as the overtime value, and "behind" as 0.'
)
SENSITIVITY_CAPTION = (
    "Sensitivity to the clutch shift only — not a confidence interval for make probability."
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

SHIFT_MODE_OPTIONS = {
    "Off": "none",
    "Overall (default)": "overall",
    "By shot type (exploratory)": "by_shot_type",
}


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
        "Exploratory, not pre-specified: in this cut the 3-point clutch shift is more "
        f"negative than the 2-point shift (Model B 3PT minus 2PT = "
        f"{diff_info['diff_logit']:.3f} logit, 95% bootstrap interval "
        f"{diff_info['ci_lower_logit']:.3f} to {diff_info['ci_upper_logit']:.3f}; "
        "one of several exploratory cuts; interval close to zero)."
    )


def format_map_limitation_note(court_defaults: dict[str, Any]) -> str:
    """Format the rim and within-cell bunching limitation notes from court_defaults.json."""
    pc = court_defaults["checks"]["cell_level_typical_mix_vs_actual"]
    cd = court_defaults["checks"]["cell_centre_diagnostic"]
    mc = pc["max_diff_cell"]
    return (
        "Limitation notes: Within about 4 ft of the rim, the typical mix can understate "
        "the average of real shots because the share of dunks depends on location while "
        f"the mix depends only on distance (largest non-putback 2 ft cell difference: "
        f"n = {mc['n_shots']} shots, {mc['dunk_share']:.1%} dunks, typical-mix "
        f"{mc['mean_p_a_typical_mix']:.3f} vs. actual {mc['mean_p_a_actual']:.3f}; "
        f"weighted mean difference across {pc['n_cells_checked']} cells is "
        f"{pc['weighted_mean_abs_diff']:.4f}). Also, a pointwise map cannot show how "
        "shots bunch inside a cell (comparing real shots against a 2 ft cell centre gives "
        f"a weighted mean difference of {cd['weighted_mean_abs_diff']:.4f}, up to "
        f"{cd['max_abs_diff']:.3f} across {cd['n_cells_ge_threshold']} rim cells >= 0.03)."
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
                    f"Gate closed: more than {gate_sec:g} s left, so the shot is not "
                    "treated as the last chance; only make probability and xP are shown.",
                )
            )
        elif flag == "no_option_changes_end_state":
            messages.append(
                (
                    "warning",
                    "No option changes the end state: every selected field goal leaves "
                    "the same end-state value V, so options are ranked by xP.",
                )
            )
        elif flag == "shift_not_applied_outside_clutch":
            messages.append(
                (
                    "info",
                    "Outside the clutch window (period >= 4, <= 300 s left, margin within "
                    "+/- 5): no clutch shift is applied.",
                )
            )
        elif flag == "unknown_player_offset_zero":
            messages.append(
                (
                    "info",
                    "Selected shooter has no regular-season baseline offset in the export; "
                    "using league baseline Model A (offset 0).",
                )
            )
    return messages


def _sorted_result_table(result: DecisionResult) -> pd.DataFrame:
    df = result.table.copy()
    if result.meta["gate_open"]:
        return df.sort_values(["rank_v", "label"], ascending=[True, True]).reset_index(drop=True)
    return df.sort_values(["xp", "label"], ascending=[False, True]).reset_index(drop=True)


def format_decision_table(result: DecisionResult) -> pd.DataFrame:
    """Format the primary decision-layer table for display."""
    sorted_df = _sorted_result_table(result)
    gate_open = bool(result.meta["gate_open"])

    if not gate_open:
        return pd.DataFrame(
            {
                "Option": sorted_df["label"],
                "Make prob.": [f"{v:.1%}" for v in sorted_df["p_make"]],
                "xP": [f"{v:.2f}" for v in sorted_df["xp"]],
            }
        )

    return pd.DataFrame(
        {
            "Option": sorted_df["label"],
            "Make prob.": [f"{v:.1%}" for v in sorted_df["p_make"]],
            "xP": [f"{v:.2f}" for v in sorted_df["xp"]],
            "Ahead": [f"{v:.1%}" for v in sorted_df["p_ahead"]],
            "Tied": [f"{v:.1%}" for v in sorted_df["p_tied"]],
            "Behind": [f"{v:.1%}" for v in sorted_df["p_behind"]],
            "V": [f"{v:.3f}" for v in sorted_df["v"]],
            "Rank": [int(v) for v in sorted_df["rank_v"]],
        }
    )


def format_sensitivity_table(result: DecisionResult) -> pd.DataFrame:
    """Format clutch-shift sensitivity columns (omits V when gate is closed)."""
    sorted_df = _sorted_result_table(result)
    out = pd.DataFrame(
        {
            "Option": sorted_df["label"],
            "Make (no shift)": [f"{v:.1%}" for v in sorted_df["p_make_no_shift"]],
            "Make (low)": [f"{v:.1%}" for v in sorted_df["p_make_shift_low"]],
            "Make (point)": [f"{v:.1%}" for v in sorted_df["p_make"]],
            "Make (high)": [f"{v:.1%}" for v in sorted_df["p_make_shift_high"]],
        }
    )
    if result.meta["gate_open"]:
        out["V (no shift)"] = [f"{v:.3f}" for v in sorted_df["v_no_shift"]]
        out["V (low)"] = [f"{v:.3f}" for v in sorted_df["v_shift_low"]]
        out["V (point)"] = [f"{v:.3f}" for v in sorted_df["v"]]
        out["V (high)"] = [f"{v:.3f}" for v in sorted_df["v_shift_high"]]
    return out


def format_decision_summary(result: DecisionResult) -> str:
    """Generate the fixed-template one-paragraph summary from table and meta."""
    sorted_df = _sorted_result_table(result)
    meta = result.meta
    top = sorted_df.iloc[0]
    top_label = str(top["label"])

    if not meta["gate_open"]:
        t_left = float(meta["state"]["time_left_sec"])
        gate = float(meta["time_gate_sec"])
        return (
            f"With {t_left:g} s left (above the {gate:g} s decision gate), this shot is "
            "not treated as the last chance, so end-state outcomes (ahead, tied or behind "
            "after the shot, if nothing else happens) and V are omitted. By expected "
            f"points, {top_label} leads at {float(top['xp']):.2f} xP "
            f"({float(top['p_make']):.1%} make probability)."
        )

    robust_text = "top option unchanged" if meta["ranking_robust"] else "top option changes"
    gap = meta.get("v_gap_first_second")
    if len(sorted_df) >= 2 and gap is not None:
        second_label = str(sorted_df.iloc[1]["label"])
        gap_clause = f"a gap of {float(gap):.3f} over {second_label}"
    else:
        gap_clause = "no second option selected for comparison"

    return (
        f"{top_label} ranks first with end-state value V = {float(top['v']):.3f} "
        f"({float(top['p_ahead']):.1%} ahead, {float(top['p_tied']):.1%} tied, "
        f"{float(top['p_behind']):.1%} behind after the shot, if nothing else happens), "
        f"with {gap_clause}. Under no shift and the lower and upper ends of the clutch "
        f"shift (sensitivity to the clutch shift only, not a confidence interval for make "
        f"probability): {robust_text}."
    )


def render_end_state_bar(result: DecisionResult) -> matplotlib.figure.Figure:
    """Plot horizontal stacked bars of Ahead, Tied, and Behind percentages."""
    sorted_df = _sorted_result_table(result).iloc[::-1].reset_index(drop=True)
    labels = sorted_df["label"].tolist()
    ahead = sorted_df["p_ahead"].to_numpy(dtype=float) * 100.0
    tied = sorted_df["p_tied"].to_numpy(dtype=float) * 100.0
    behind = sorted_df["p_behind"].to_numpy(dtype=float) * 100.0

    height = max(2.2, 0.58 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = range(len(labels))

    ax.barh(y, ahead, color="#0072B2", label="Ahead")
    ax.barh(y, tied, left=ahead, color="#E69F00", label="Tied")
    ax.barh(y, behind, left=ahead + tied, color="#CC79A7", label="Behind")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Share after the shot, if nothing else happens (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
