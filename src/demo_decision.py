"""Phase 4b — Step 4: Decision-layer demo across six game-state scenarios.

Runs six scenarios with the real exported model and real-shot zone presets,
computes the Scenario 2 break-even ``tie_value`` between ``corner_three`` and
``restricted_area``, and reruns Scenarios 1 and 2 with ``shift_mode="by_shot_type"``
and two anonymized shooters picked by offset bucket among players with at least
1,000 regular-play shots (``"median offset"`` and ``"top-decile offset"``).

No ``PLAYER_ID``, player name, exact shot count, or exact offset is written to
``docs/decision_demo.json`` or printed.

Run from the repo root::

    python src/demo_decision.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision_layer import DecisionResult, evaluate_options, load_zone_presets  # noqa: E402
from src.simulator_model import DEFAULT_EXPORT_DIR, ExportedModel, load_export  # noqa: E402

DOCS_DIR = ROOT / "docs"
OUT_JSON = DOCS_DIR / "decision_demo.json"
OUT_DOC = DOCS_DIR / "decision_layer.md"

MIN_REGULAR_SHOTS = 1000
DEMO_OPTIONS = [
    "above_the_break_3",
    "corner_three",
    "long_mid_range",
    "restricted_area",
]

PRIMARY_SCENARIOS = [
    ("scenario_1_down_3_8s", "Scenario 1: Down 3, 8s left (overall, no shooter)", -3, 8.0),
    ("scenario_2_down_2_8s", "Scenario 2: Down 2, 8s left (overall, no shooter)", -2, 8.0),
    ("scenario_3_down_1_6s", "Scenario 3: Down 1, 6s left (overall, no shooter)", -1, 6.0),
    ("scenario_4_tied_5s", "Scenario 4: Tied, 5s left (overall, no shooter)", 0, 5.0),
    ("scenario_5_down_3_40s", "Scenario 5: Down 3, 40s left (gate closed)", -3, 40.0),
    ("scenario_6_down_5_10s", "Scenario 6: Down 5, 10s left (no end-state change)", -5, 10.0),
]


def _pick_bucket_shooters(model: ExportedModel) -> tuple[list[tuple[str, str, int]], int]:
    """Select median-offset and top-decile-offset shooters among >=1,000-shot players."""
    eligible = (
        model.player_table[model.player_table["n_regular_shots"] >= MIN_REGULAR_SHOTS]
        .sort_values(["offset", "PLAYER_ID"], ascending=[True, True])
        .reset_index(drop=True)
    )
    n_qual = len(eligible)
    if n_qual < 10:
        raise RuntimeError(
            f"Only {n_qual} players have >= {MIN_REGULAR_SHOTS} regular-play shots (< 10)."
        )

    idx_median = int(0.5 * (n_qual - 1))
    idx_top_decile = int(0.9 * n_qual)
    pid_median = int(eligible.loc[idx_median, "PLAYER_ID"])
    pid_top = int(eligible.loc[idx_top_decile, "PLAYER_ID"])

    return [
        ("median_offset", "median offset", pid_median),
        ("top_decile_offset", "top-decile offset", pid_top),
    ], n_qual


def _clean_record(raw: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in raw.items():
        if v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            cleaned[k] = None
        elif hasattr(v, "item"):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def _serialize_result(
    key: str,
    title: str,
    res: DecisionResult,
    shooter_bucket_label: Optional[str] = None,
) -> dict[str, Any]:
    meta = dict(res.meta)
    if shooter_bucket_label is not None:
        meta["player_id"] = shooter_bucket_label
        meta["effective_player_id"] = shooter_bucket_label
        meta["player_offset"] = None
        meta["shooter_bucket"] = shooter_bucket_label
    else:
        meta["shooter_bucket"] = None

    rows = [_clean_record(r) for r in res.table.to_dict(orient="records")]
    return {
        "key": key,
        "title": title,
        "meta": meta,
        "rows": rows,
    }


def _print_scenario(entry: dict[str, Any]) -> None:
    meta = entry["meta"]
    print(f"\n=== {entry['title']} ===")
    print(
        f"state={meta['state']} | shift_mode={meta['shift_mode']} | "
        f"shooter={meta['shooter_bucket'] or 'none'} | gate_open={meta['gate_open']} | "
        f"v_gap_1_2={meta['v_gap_first_second']} | ranking_robust={meta['ranking_robust']} | "
        f"flags={meta['flags']}"
    )
    df = pd.DataFrame(entry["rows"])
    cols = [
        "label",
        "n_rows",
        "shot_value",
        "p_make",
        "xp",
        "p_ahead",
        "p_tied",
        "p_behind",
        "v",
        "v_no_shift",
        "v_shift_low",
        "v_shift_high",
        "rank_v",
        "rank_xp",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def run_demo(
    model: Optional[ExportedModel] = None,
    presets: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    loaded_model = model if model is not None else load_export(DEFAULT_EXPORT_DIR)
    loaded_presets = presets if presets is not None else load_zone_presets()

    primary_entries: list[dict[str, Any]] = []
    for key, title, margin, time_left in PRIMARY_SCENARIOS:
        res = evaluate_options(
            {"time_left_sec": time_left, "margin": margin, "period": 4},
            DEMO_OPTIONS,
            player_id=None,
            shift_mode="overall",
            model=loaded_model,
            presets=loaded_presets,
        )
        primary_entries.append(_serialize_result(key, title, res))

    # Break-even tie_value between Corner Three and Restricted Area in Scenario 2 (down 2, 8s)
    scen2_rows = {r["label"]: r for r in primary_entries[1]["rows"]}
    p_corner_3 = float(scen2_rows["Corner Three"]["p_make"])
    p_restricted = float(scen2_rows["Restricted Area"]["p_make"])
    breakeven_tv = p_corner_3 / p_restricted

    shooters, n_qualifying = _pick_bucket_shooters(loaded_model)
    sensitivity_entries: list[dict[str, Any]] = []
    for scen_num, margin in ((1, -3), (2, -2)):
        res_type = evaluate_options(
            {"time_left_sec": 8.0, "margin": margin, "period": 4},
            DEMO_OPTIONS,
            player_id=None,
            shift_mode="by_shot_type",
            model=loaded_model,
            presets=loaded_presets,
        )
        sensitivity_entries.append(
            _serialize_result(
                f"scenario_{scen_num}_by_shot_type_no_shooter",
                f"Scenario {scen_num} rerun (by_shot_type, no shooter): margin={margin}, 8s left",
                res_type,
                shooter_bucket_label=None,
            )
        )
        for bucket_key, bucket_label, pid in shooters:
            for s_mode in ("overall", "by_shot_type"):
                res = evaluate_options(
                    {"time_left_sec": 8.0, "margin": margin, "period": 4},
                    DEMO_OPTIONS,
                    player_id=pid,
                    shift_mode=s_mode,  # type: ignore[arg-type]
                    model=loaded_model,
                    presets=loaded_presets,
                )
                key = f"scenario_{scen_num}_{bucket_key}_{s_mode}"
                title = (
                    f"Scenario {scen_num} rerun ({bucket_label}, {s_mode}): "
                    f"margin={margin}, 8s left"
                )
                sensitivity_entries.append(
                    _serialize_result(key, title, res, shooter_bucket_label=bucket_label)
                )

    payload = {
        "meta": {
            "options": DEMO_OPTIONS,
            "shooter_selection": {
                "min_regular_shots": MIN_REGULAR_SHOTS,
                "n_qualifying_players": n_qualifying,
                "buckets": ["median offset", "top-decile offset"],
                "rule": (
                    "Among players in export/player_offsets.csv with at least 1,000 regular-play shots, "
                    "sort by offset ascending (tie-broken by PLAYER_ID) and select index int(0.5 * (N - 1)) "
                    "for 'median offset' and index int(0.9 * N) for 'top-decile offset'. "
                    "No PLAYER_ID, name, exact shot count, or exact offset is stored."
                ),
            },
            "scenario_2_breakeven_tie_value": {
                "option_a": "Corner Three",
                "option_b": "Restricted Area",
                "p_corner_three": p_corner_3,
                "p_restricted_area": p_restricted,
                "breakeven_tie_value": breakeven_tv,
                "formula": "V(Corner Three) = p_corner_three; V(Restricted Area) = tie_value * p_restricted_area; equal at tie_value = p_corner_three / p_restricted_area.",
            },
        },
        "primary_scenarios": primary_entries,
        "sensitivity_reruns": sensitivity_entries,
    }
    return payload


def _md_scenario_table(entry: dict[str, Any]) -> list[str]:
    meta = entry["meta"]
    gap = meta["v_gap_first_second"]
    gap_str = f"{gap:.6f}" if gap is not None else "None"
    lines = [
        f"#### {entry['title']}",
        "",
        (
            f"- **State:** `time_left_sec={meta['state']['time_left_sec']:.1f}`, "
            f"`margin={meta['state']['margin']}`, `period={meta['state']['period']}` | "
            f"**Shift mode:** `{meta['shift_mode']}` | "
            f"**Shooter:** `{meta['shooter_bucket'] or 'none'}` | "
            f"**Gate open:** `{meta['gate_open']}` | "
            f"**`v_gap_first_second`:** `{gap_str}` | "
            f"**`ranking_robust`:** `{meta['ranking_robust']}` | "
            f"**Flags:** `{meta['flags']}`"
        ),
        "",
        "| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in entry["rows"]:
        def _fmt(val: Any) -> str:
            return "—" if val is None else f"{val:.6f}"

        def _fmt_int(val: Any) -> str:
            return "—" if val is None else str(val)

        lines.append(
            f"| {r['label']} | {r['n_rows']} | {r['shot_value']} | {r['p_make']:.6f} | {r['xp']:.6f} | "
            f"{_fmt(r['p_ahead'])} | {_fmt(r['p_tied'])} | {_fmt(r['p_behind'])} | "
            f"{_fmt(r['v'])} | {_fmt(r['v_no_shift'])} | {_fmt(r['v_shift_low'])} | {_fmt(r['v_shift_high'])} | "
            f"{_fmt_int(r['rank_v'])} | {_fmt_int(r['rank_xp'])} |"
        )
    lines.append("")
    return lines


def build_decision_layer_markdown(
    demo_payload: dict[str, Any],
    presets_payload: dict[str, Any],
    shift_payload: dict[str, Any],
) -> str:
    """Generate docs/decision_layer.md programmatically from exported JSON artifacts."""
    ma = shift_payload["model_a"]
    mb = shift_payload["model_b"]
    exp = shift_payload["exploratory"]
    diff_exp = exp["model_b_3pt_minus_2pt"]
    be = demo_payload["meta"]["scenario_2_breakeven_tie_value"]
    ss = demo_payload["meta"]["shooter_selection"]

    # Extract Scenario 2 rank 2/3 swap numbers programmatically from demo_payload
    s2_overall = {r["label"]: r for r in demo_payload["primary_scenarios"][1]["rows"]}
    reruns = {e["key"]: e for e in demo_payload["sensitivity_reruns"]}
    s2_by_type = {r["label"]: r for r in reruns["scenario_2_by_shot_type_no_shooter"]["rows"]}
    s2_med_over = {r["label"]: r for r in reruns["scenario_2_median_offset_overall"]["rows"]}
    s2_med_type = {r["label"]: r for r in reruns["scenario_2_median_offset_by_shot_type"]["rows"]}

    lines: list[str] = [
        "# Clutch Decision Layer (Phase 4b, Field Goals Only)",
        "",
        "This document describes the pure-Python clutch decision layer in `src/decision_layer.py`, the court geometry helper in `src/shot_geometry.py`, and the real-shot zone presets in `export/zone_presets.json`. All modules run strictly on `export/` without `data/` or `models/`.",
        "",
        "**Scope of v1: field goals only.** No free throws, fouls on the shot, offensive rebounds, turnovers, or opponent possession after the shot are modelled.",
        "",
        "## 1. Public API and Inputs",
        "",
        "Entry point: `evaluate_options(state, options, player_id=None, shift_mode=\"overall\", time_gate_sec=24.0, tie_value=0.5, model=None, presets=None)` returning `DecisionResult(table, meta)`.",
        "",
        "- **`state`** (`GameState` or `dict`):",
        "  - `time_left_sec` (`float`, validated in `[0, 3000]`): seconds remaining in the period.",
        "  - `margin` (`int`, bool/non-integer rejected): offense minus defense **before** the shot.",
        "  - `period` (`int >= 4`, default `4`).",
        "- **`options`** (non-empty sequence): preset keys from `export/zone_presets.json` and/or custom shot dicts containing `label` and the seven baseline features (`LOC_X`, `LOC_Y`, `SHOT_DISTANCE_FT`, `SHOT_ANGLE`, `SHOT_VALUE`, `ACTION_GROUP`, `IS_PUTBACK`). Unknown presets or unknown `ACTION_GROUP` values raise `ValueError`.",
        "- **`player_id`** (optional `int | str | None`): chosen once for all options. Unknown or absent players receive offset `0.0` (falling back to Model A) and set the flag `unknown_player_offset_zero`.",
        "- **`shift_mode`**: `\"overall\"` (default), `\"by_shot_type\"` (exploratory), or `\"none\"`.",
        "- **`time_gate_sec`** (`float >= 0`, default `24.0`): decision gate threshold.",
        "- **`tie_value`** (`float` in `[0, 1]`, default `0.5`): weight assigned to a tied post-shot score in `V`.",
        "",
        "## 2. Geometry and Zone Presets",
        "",
        "### Court Coordinate Geometry (`src/shot_geometry.py`)",
        "",
        "Verified across all `219,160` shots in `data/processed/all_shots.parquet`:",
        "- `SHOT_DISTANCE_FT = floor(hypot(LOC_X, LOC_Y) / 10)` (truncation, matching `100.0%` of stored shots; non-putback shots below `1.0` ft are clamped to `1.0` ft to match `src/simulator_model.py`).",
        "- `SHOT_ANGLE = degrees(arctan2(LOC_X, LOC_Y))` (`100.0%` exact match).",
        "- `SHOT_VALUE` assigns `3` when `|LOC_X| >= 220` in the corners (`y <= sqrt(237.5^2 - 220^2)`) or `hypot(LOC_X, LOC_Y) >= 237.5` on the arc, else `2` (`99.9909%` exact agreement; `20` mismatches on the three-point line).",
        "",
        "### Real-Shot Zone Presets (`export/zone_presets.json`)",
        "",
        "A single representative coordinate per zone is fragile because Model A uses signed `LOC_X` and `SHOT_ANGLE`, so make probability at a single point and its mirror image can differ by up to `0.042` (restricted area) due to local tree splits. **Custom single-point options inherit the model's local structure, including left-right differences in Model A; zone presets average over fixed samples of real regular-play shots to avoid this.**",
        "",
        f"Presets are drawn from `REGULAR` shots only (`{presets_payload['meta']['n_regular_shots_total']:,}` shots, `seed={presets_payload['meta']['seed']}`, no `GAME_ID`, `PLAYER_ID`, or `SHOT_MADE_FLAG`): `300` shots per zone by default, redrawn with `1,000` rows if `|mean_p_A_sample - mean_p_A_zone| > {presets_payload['meta']['target_abs_p_a_diff']}`. In the decision layer, a preset's probability is the mean over its `n_rows` of the per-row final probabilities (player offset and clutch shift applied per row on the logit scale before averaging).",
        "",
        "| Preset | `n_zone` | `n_sample` | Mean `p_A` (zone) | SD `p_A` (zone) | SE sample mean | Mean `p_A` (sample) | Abs diff | `abs_diff / SE` | L/R/C share (zone) | L/R/C share (sample) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]

    for pname, entry in presets_payload["presets"].items():
        sz = entry["side_shares_zone"]
        ss_sh = entry["side_shares_sample"]
        lines.append(
            f"| `{pname}` | {entry['n_zone_shots']:,} | {entry['n_sample']:,} | "
            f"{entry['mean_p_a_zone']:.6f} | {entry['sd_p_a_zone']:.6f} | {entry['se_sample_mean']:.6f} | "
            f"{entry['mean_p_a_sample']:.6f} | {entry['abs_mean_p_a_diff']:.6f} | {entry['abs_diff_over_se']:.4f} | "
            f"{sz['left']:.3f}/{sz['right']:.3f}/{sz['center']:.3f} | "
            f"{ss_sh['left']:.3f}/{ss_sh['right']:.3f}/{ss_sh['center']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Decision Formulas, Clutch Shift, and Gate",
            "",
            "### Baseline and Clutch Shift",
            "",
            "The baseline model was fitted on normal-play (`REGULAR`) shots only; clutch outcomes were used only to estimate the descriptive **clutch shift** (or **clutch gap**), a descriptive quantity bundling defense, fatigue, shot selection within a shot type, and shooter composition:",
            "",
            "```text",
            "p_final = sigmoid(logit(clip(p_base, 1e-6, 1 - 1e-6)) + shift)",
            "```",
            "",
            "The clutch shift is applied **only** in a clutch state (`period >= 4`, `time_left_sec <= 300`, `abs(margin) <= 5`). Outside a clutch state, no shift is applied and `shift_not_applied_outside_clutch` is set.",
            "",
            "- **`\"overall\"` (default):** Model A `{:.6f}` (`95% CI [{:.6f}, {:.6f}]`) when `player_id=None`; Model B `{:.6f}` (`95% CI [{:.6f}, {:.6f}]`) when a known shooter is chosen.".format(
                ma["shift_logit"],
                ma["ci_lower_logit"],
                ma["ci_upper_logit"],
                mb["shift_logit"],
                mb["ci_lower_logit"],
                mb["ci_upper_logit"],
            ),
            "- **`\"by_shot_type\"` (exploratory, not pre-specified):** uses 2PT vs 3PT shifts (Model A: 2PT `{:.6f}`, 3PT `{:.6f}`; Model B: 2PT `{:.6f}`, 3PT `{:.6f}`). In this cut the 3PT shift is more negative than the 2PT shift (Model B 3PT minus 2PT = `{:.3f}` logit, 95% CI `{:.3f}` to `{:.3f}`; one of several exploratory cuts, interval close to zero).".format(
                exp["model_a_2pt"]["shift_logit"],
                exp["model_a_3pt"]["shift_logit"],
                exp["model_b_2pt"]["shift_logit"],
                exp["model_b_3pt"]["shift_logit"],
                diff_exp["diff_logit"],
                diff_exp["ci_lower_logit"],
                diff_exp["ci_upper_logit"],
            ),
            "- **`\"none\"`:** no clutch shift applied (`shift = 0`).",
            "",
            "### Outcomes After the Shot and End-State Value `V`",
            "",
            "With make probability `p`, shot value `v` (`2` or `3`), and pre-shot margin `m` (offense minus defense), the probabilities of ending **ahead, tied or behind after the shot, if nothing else happens** are:",
            "",
            "```text",
            "P_ahead  = p * [m + v > 0] + (1 - p) * [m > 0]",
            "P_tied   = p * [m + v == 0] + (1 - p) * [m == 0]",
            "P_behind = 1 - P_ahead - P_tied",
            "V        = P_ahead + tie_value * P_tied",
            "xP       = p * v",
            "```",
            "",
            "- Default `tie_value = 0.5` treats overtime as a coin flip; this is a pre-specified modelling choice, not an estimate.",
            "- Options are ranked by `V`, then `xP`, then `label`. When all options have identical `V` (differences `< 1e-9`), `no_option_changes_end_state` is set and options are ranked by `xP` only.",
            "- `meta[\"v_gap_first_second\"]` records `V(rank 1) - V(rank 2)` when the gate is open and at least two options are provided (otherwise `None`).",
            "",
            "### Time Gate and Sensitivity to the Clutch Shift",
            "",
            "- **Time gate (`time_gate_sec = 24.0`):** End-state columns (`p_ahead`, `p_tied`, `p_behind`, `v`, `rank_v`, `rank_xp`) are computed only when `time_left_sec <= time_gate_sec`. Above the gate, the \"nothing else happens\" assumption is not credible, so decision columns are empty (`NaN` / `<NA>`) and `decision_gate_closed` is set. The 24-second threshold (about one shot clock) is a modelling choice, not estimated from data.",
            "- **Sensitivity to the clutch shift:** Each option is also evaluated under (a) no shift, (b) the lower 95% shift bound, and (c) the upper 95% shift bound, recomputing `V` for each. This is a **sensitivity to the clutch shift only**, not a confidence interval for make probability (it excludes baseline model error, player-offset error, and unmodelled factors).",
            "- **`ranking_robust`:** `ranking_robust` concerns the **top-ranked option only** (True iff rank 1 by `V` is the same under no shift, lower bound, point shift, and upper bound). Ranks 2 and 3 can still swap between shift modes or shift bounds.",
            "",
            "## 4. Demo Scenarios (`docs/decision_demo.json`)",
            "",
            "### Primary Scenarios (`shift_mode=\"overall\"`, no shooter)",
            "",
        ]
    )

    for entry in demo_payload["primary_scenarios"]:
        lines.extend(_md_scenario_table(entry))

    lines.extend(
        [
            "### Scenario 2 Break-Even `tie_value` Sensitivity",
            "",
            (
                f"In Scenario 2 (down 2, 8 seconds left, `overall` shift, no shooter), "
                f"`Corner Three` gives `P_ahead = {be['p_corner_three']:.6f}` and `P_tied = 0`, "
                f"so `V(Corner Three) = {be['p_corner_three']:.6f}` regardless of `tie_value`. "
                f"`Restricted Area` gives `P_ahead = 0` and `P_tied = {be['p_restricted_area']:.6f}`, "
                f"so `V(Restricted Area) = tie_value * {be['p_restricted_area']:.6f}`. "
                f"Their `V` values are equal at `tie_value = {be['p_corner_three']:.6f} / {be['p_restricted_area']:.6f} = {be['breakeven_tie_value']:.6f}`. "
                f"The ranking of these two options depends on this assumption: `Corner Three` leads for `tie_value < {be['breakeven_tie_value']:.6f}` (including default `0.5`), whereas `Restricted Area` leads for `tie_value > {be['breakeven_tie_value']:.6f}`."
            ),
            "",
            "### Scenarios 1 and 2 Sensitivity Reruns (`by_shot_type` & Offset-Bucket Shooters)",
            "",
            (
                f"**Shooter selection rule:** {ss['rule']} "
                f"({ss['n_qualifying_players']} qualifying players with `>= {ss['min_regular_shots']:,}` regular-play shots; "
                f"bucket labels only: `\"median offset\"` and `\"top-decile offset\"`)."
            ),
            "",
            (
                f"**Rank 2 / Rank 3 swap in Scenario 2 (`overall` vs `by_shot_type`):** "
                f"`ranking_robust` concerns the top-ranked option only (`Corner Three` remains rank 1 in all reruns, so `ranking_robust = True`). "
                f"However, ranks 2 and 3 swap between shift modes when no shooter is selected: under `overall`, `Above-the-Break Three` is rank 2 (`V = {s2_overall['Above-the-Break Three']['v']:.6f}`) ahead of `Restricted Area` (`V = {s2_overall['Restricted Area']['v']:.6f}`), whereas under exploratory `by_shot_type`, `Restricted Area` moves to rank 2 (`V = {s2_by_type['Restricted Area']['v']:.6f}`) ahead of `Above-the-Break Three` (`V = {s2_by_type['Above-the-Break Three']['v']:.6f}`). "
                f"The same rank 2/3 swap appears for the `\"median offset\"` shooter (`overall`: `Above-the-Break Three` `V = {s2_med_over['Above-the-Break Three']['v']:.6f}` vs `Restricted Area` `V = {s2_med_over['Restricted Area']['v']:.6f}`; `by_shot_type`: `Restricted Area` `V = {s2_med_type['Restricted Area']['v']:.6f}` vs `Above-the-Break Three` `V = {s2_med_type['Above-the-Break Three']['v']:.6f}`)."
            ),
            "",
        ]
    )

    for entry in demo_payload["sensitivity_reruns"]:
        lines.extend(_md_scenario_table(entry))

    lines.extend(
        [
            "## 5. Limitations",
            "",
            "- **Field goals only:** no free throws, fouls on the shot, offensive rebounds, or turnovers are modelled.",
            "- **No opponent possession after the shot:** end-state probabilities describe being ahead, tied, or behind immediately after the shot if nothing else happens.",
            "- **No defender or shot-clock data:** closest-defender distance, contest level, and shot-clock time are absent from the data.",
            "- **Single season:** data covers only the `2025-26` NBA regular season.",
            "- **No time/margin effect on baseline make probability:** Phase 3 found no detectable effect of time remaining or score margin on make probability.",
            "- **Clock feasibility:** very short times remaining (a few seconds) may not allow some shot types in practice.",
            "- **Clutch shift scope:** the clutch shift is estimated on close late-game shots only (`period >= 4`, `time_left_sec <= 300`, `abs(margin) <= 5`) and is a descriptive gap rather than a causal effect.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload = run_demo()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    presets_payload = json.loads((DEFAULT_EXPORT_DIR / "zone_presets.json").read_text(encoding="utf-8"))
    shift_payload = json.loads((DEFAULT_EXPORT_DIR / "clutch_shift.json").read_text(encoding="utf-8"))
    OUT_DOC.write_text(
        build_decision_layer_markdown(payload, presets_payload, shift_payload),
        encoding="utf-8",
    )

    be = payload["meta"]["scenario_2_breakeven_tie_value"]
    print(f"Wrote {OUT_JSON} and {OUT_DOC}")
    print(
        f"Scenario 2 break-even tie_value (Corner Three vs Restricted Area): "
        f"{be['p_corner_three']:.6f} / {be['p_restricted_area']:.6f} = {be['breakeven_tie_value']:.6f}"
    )
    for entry in payload["primary_scenarios"]:
        _print_scenario(entry)
    for entry in payload["sensitivity_reruns"]:
        _print_scenario(entry)


if __name__ == "__main__":
    main()
