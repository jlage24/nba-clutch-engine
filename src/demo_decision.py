"""Phase 4b — Step 4: Decision-layer demo across six game-state scenarios.

Runs six scenarios with the real exported model and real-shot zone presets,
plus follow-up runs on scenarios 1 and 2 with ``shift_mode="by_shot_type"``
and one high-volume anonymized shooter (never named or ranked). Writes
``docs/decision_demo.json`` and prints the evaluation tables.

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


def _pick_anonymized_high_volume_shooter(model: ExportedModel) -> tuple[int, dict[str, Any]]:
    """Select one high-volume shooter from player_offsets.csv without exposing name or rank."""
    eligible = model.player_table[model.player_table["n_regular_shots"] >= 1000].sort_values(
        "PLAYER_ID"
    )
    row = eligible.iloc[0]
    pid = int(row["PLAYER_ID"])
    info = {
        "shooter_label": "anonymized_high_volume_shooter",
        "n_regular_shots": int(row["n_regular_shots"]),
        "player_offset": float(row["offset"]),
    }
    return pid, info


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
    anonymize_player: bool = False,
) -> dict[str, Any]:
    meta = dict(res.meta)
    if anonymize_player and meta["player_id"] is not None:
        meta["player_id"] = "anonymized_high_volume_shooter"
        meta["effective_player_id"] = "anonymized_high_volume_shooter"

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
        f"gate_open={meta['gate_open']} | ranking_robust={meta['ranking_robust']} | "
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

    chosen_pid, shooter_summary = _pick_anonymized_high_volume_shooter(loaded_model)
    sensitivity_entries: list[dict[str, Any]] = []
    for scen_num, margin in ((1, -3), (2, -2)):
        for mode_label, pid, s_mode, anon in (
            ("by_shot_type_no_shooter", None, "by_shot_type", False),
            ("chosen_shooter_overall", chosen_pid, "overall", True),
            ("chosen_shooter_by_shot_type", chosen_pid, "by_shot_type", True),
        ):
            res = evaluate_options(
                {"time_left_sec": 8.0, "margin": margin, "period": 4},
                DEMO_OPTIONS,
                player_id=pid,
                shift_mode=s_mode,  # type: ignore[arg-type]
                model=loaded_model,
                presets=loaded_presets,
            )
            key = f"scenario_{scen_num}_{mode_label}"
            title = f"Scenario {scen_num} rerun ({mode_label}): margin={margin}, 8s left"
            sensitivity_entries.append(_serialize_result(key, title, res, anonymize_player=anon))

    payload = {
        "meta": {
            "options": DEMO_OPTIONS,
            "anonymized_shooter": shooter_summary,
        },
        "primary_scenarios": primary_entries,
        "sensitivity_reruns": sensitivity_entries,
    }
    return payload


def main() -> None:
    payload = run_demo()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Anonymized high-volume shooter: {payload['meta']['anonymized_shooter']}")
    for entry in payload["primary_scenarios"]:
        _print_scenario(entry)
    for entry in payload["sensitivity_reruns"]:
        _print_scenario(entry)


if __name__ == "__main__":
    main()
