"""Phase 4b — Step 3: Pure-Python clutch decision layer (field goals only).

Sits on top of ``src/simulator_model.py`` and ``export/`` (no dependency on
``data/`` or ``models/``). Given a game state and candidate shot options
(zone presets from ``export/zone_presets.json`` or custom shot dicts), computes:

- Make probability ``p_make`` and expected points ``xp``.
- Sensitivity to the descriptive clutch shift (no shift, 95% CI low, 95% CI high).
- End-state outcome probabilities (``p_ahead``, ``p_tied``, ``p_behind``) and
  end-state value ``V = P_ahead + 0.5 * P_tied`` when ``time_left_sec <= time_gate_sec``.
- Rankings by ``V`` and by ``xp``, plus ``ranking_robust`` and diagnostic flags.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator_model import (  # noqa: E402
    DEFAULT_EXPORT_DIR,
    ExportedModel,
    load_export,
    predict_make_probability,
)

DEFAULT_PRESETS_PATH = DEFAULT_EXPORT_DIR / "zone_presets.json"
MAX_TIME_LEFT_SEC = 3000.0
CLUTCH_TIME_SEC = 300.0
CLUTCH_MAX_ABS_MARGIN = 5
END_STATE_TIE_TOL = 1e-9
VALID_SHIFT_MODES = ("overall", "by_shot_type", "none")

RESULT_COLUMNS = [
    "label",
    "n_rows",
    "shot_value",
    "p_make",
    "xp",
    "p_make_no_shift",
    "p_make_shift_low",
    "p_make_shift_high",
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


@dataclass(frozen=True)
class GameState:
    """Validated pre-shot game state."""

    time_left_sec: float
    margin: int
    period: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.time_left_sec, bool) or not isinstance(
            self.time_left_sec, (int, float, np.number)
        ):
            raise ValueError(f"time_left_sec must be numeric, got {self.time_left_sec!r}.")
        t = float(self.time_left_sec)
        if not math.isfinite(t) or t < 0.0 or t > MAX_TIME_LEFT_SEC:
            raise ValueError(
                f"time_left_sec must be between 0 and {int(MAX_TIME_LEFT_SEC)}, got {self.time_left_sec!r}."
            )
        object.__setattr__(self, "time_left_sec", t)

        if isinstance(self.margin, bool) or not isinstance(self.margin, (int, np.integer)):
            raise ValueError(f"margin must be an integer, got {self.margin!r}.")
        object.__setattr__(self, "margin", int(self.margin))

        if isinstance(self.period, bool) or not isinstance(self.period, (int, np.integer)):
            raise ValueError(f"period must be an integer, got {self.period!r}.")
        p = int(self.period)
        if p < 4:
            raise ValueError(f"period must be >= 4, got {self.period!r}.")
        object.__setattr__(self, "period", p)

    @property
    def is_clutch(self) -> bool:
        """Whether state falls inside the pre-specified clutch window."""
        return (
            self.period >= 4
            and self.time_left_sec <= CLUTCH_TIME_SEC
            and abs(self.margin) <= CLUTCH_MAX_ABS_MARGIN
        )


@dataclass(frozen=True)
class DecisionResult:
    """Decision-layer evaluation table and accompanying metadata."""

    table: pd.DataFrame
    meta: dict[str, Any]


def load_zone_presets(
    path: Union[str, Path] = DEFAULT_PRESETS_PATH,
) -> dict[str, dict[str, Any]]:
    """Load exported real-shot zone presets from ``export/zone_presets.json``."""
    presets_path = Path(path)
    if not presets_path.exists():
        raise FileNotFoundError(f"Zone presets file not found: {presets_path}")
    payload = json.loads(presets_path.read_text(encoding="utf-8"))
    return dict(payload["presets"])


def _coerce_state(state: Union[GameState, dict[str, Any]]) -> GameState:
    if isinstance(state, GameState):
        return state
    if isinstance(state, dict):
        allowed_keys = {"time_left_sec", "margin", "period"}
        unknown = set(state.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown state keys: {sorted(unknown)}")
        if "time_left_sec" not in state or "margin" not in state:
            raise ValueError("State dict must include 'time_left_sec' and 'margin'.")
        return GameState(
            time_left_sec=state["time_left_sec"],
            margin=state["margin"],
            period=state.get("period", 4),
        )
    raise ValueError(f"state must be a GameState or dict, got {type(state).__name__}.")


def _resolve_options(
    options: Sequence[Union[str, dict[str, Any]]],
    presets: dict[str, dict[str, Any]],
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    if not options:
        raise ValueError("options must contain at least one shot option.")

    resolved: list[tuple[str, int, list[dict[str, Any]]]] = []
    for opt in options:
        if isinstance(opt, str):
            if opt not in presets:
                available = ", ".join(sorted(presets.keys()))
                raise ValueError(f"Unknown preset {opt!r}. Available presets: {available}.")
            entry = presets[opt]
            shots = list(entry["shots"])
            if not shots:
                raise ValueError(f"Preset {opt!r} contains no shots.")
            shot_val = int(entry["shot_value"])
            label = str(entry.get("label", opt))
            resolved.append((label, shot_val, shots))
        elif isinstance(opt, dict):
            if "label" not in opt or not str(opt["label"]).strip():
                raise ValueError("Custom option dict must include a non-empty 'label'.")
            shot_dict = {k: v for k, v in opt.items() if k != "label"}
            if "SHOT_VALUE" not in shot_dict:
                raise ValueError(
                    f"Custom option {opt['label']!r} must include 'SHOT_VALUE' (2 or 3)."
                )
            shot_val = int(shot_dict["SHOT_VALUE"])
            if shot_val not in (2, 3) or isinstance(shot_dict["SHOT_VALUE"], bool):
                raise ValueError(
                    f"SHOT_VALUE must be 2 or 3 for option {opt['label']!r}, got {shot_dict['SHOT_VALUE']!r}."
                )
            if "IS_PUTBACK" in shot_dict and (
                isinstance(shot_dict["IS_PUTBACK"], bool)
                or int(shot_dict["IS_PUTBACK"]) not in (0, 1)
            ):
                raise ValueError(
                    f"IS_PUTBACK must be 0 or 1 for option {opt['label']!r}."
                )
            resolved.append((str(opt["label"]), shot_val, [shot_dict]))
        else:
            raise ValueError(
                f"Each option must be a preset name (str) or dict, got {type(opt).__name__}."
            )
    return resolved


def _end_state_probs(p: float, shot_value: int, margin: int) -> tuple[float, float, float, float]:
    """Compute (P_ahead, P_tied, P_behind, V) assuming nothing else happens after the shot."""
    make_margin = margin + shot_value
    p_ahead = p * float(make_margin > 0) + (1.0 - p) * float(margin > 0)
    p_tied = p * float(make_margin == 0) + (1.0 - p) * float(margin == 0)
    p_behind = 1.0 - p_ahead - p_tied
    v = p_ahead + 0.5 * p_tied
    return p_ahead, p_tied, p_behind, v


def _rank_indices(
    labels: list[str],
    v_vals: list[float],
    xp_vals: list[float],
) -> tuple[list[int], bool, int]:
    """Return 1-based ranks, no_option_changes_end_state flag, and index of top option."""
    n = len(labels)
    no_change = (max(v_vals) - min(v_vals)) < END_STATE_TIE_TOL
    if no_change:
        order = sorted(range(n), key=lambda i: (-xp_vals[i], labels[i]))
    else:
        order = sorted(range(n), key=lambda i: (-v_vals[i], -xp_vals[i], labels[i]))

    ranks = [0] * n
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks, no_change, order[0]


def _describe_shifts_used(
    model: ExportedModel,
    effective_player_id: Optional[str],
    effective_shift_mode: str,
) -> dict[str, Any]:
    if effective_shift_mode == "none":
        return {"mode": "none", "model": None, "shift_logit": 0.0}

    prefix = "model_b" if effective_player_id is not None else "model_a"
    if effective_shift_mode == "overall":
        entry = model.clutch_shift_info[prefix]
        return {
            "mode": "overall",
            "model": prefix,
            "shift_logit": float(entry["shift_logit"]),
            "ci_lower_logit": float(entry["ci_lower_logit"]),
            "ci_upper_logit": float(entry["ci_upper_logit"]),
        }

    exp = model.clutch_shift_info["exploratory"]
    return {
        "mode": "by_shot_type",
        "model": prefix,
        "2pt": {
            "shift_logit": float(exp[f"{prefix}_2pt"]["shift_logit"]),
            "ci_lower_logit": float(exp[f"{prefix}_2pt"]["ci_lower_logit"]),
            "ci_upper_logit": float(exp[f"{prefix}_2pt"]["ci_upper_logit"]),
        },
        "3pt": {
            "shift_logit": float(exp[f"{prefix}_3pt"]["shift_logit"]),
            "ci_lower_logit": float(exp[f"{prefix}_3pt"]["ci_lower_logit"]),
            "ci_upper_logit": float(exp[f"{prefix}_3pt"]["ci_upper_logit"]),
        },
    }


def _build_assumptions(shift_mode: str, time_gate_sec: float) -> list[str]:
    assumptions = [
        "Field goals only: no free throws, fouls on the shot, offensive rebounds, or turnovers are modelled.",
        "No opponent possession after the shot: P_ahead, P_tied, and P_behind apply after the shot if nothing else happens.",
        "End-state value V = P_ahead + 0.5 * P_tied treats overtime as a coin flip (a pre-specified modelling choice, not an estimate).",
        f"Time gate of {time_gate_sec:g} seconds (default 24.0 s, about one shot clock) is a modelling choice, not estimated from data.",
        "Baseline Model A was fitted on normal-play (REGULAR) shots only; clutch outcomes were used only to estimate the descriptive clutch shift (a clutch gap bundling defense, fatigue, shot selection within shot type, and shooter composition).",
        "Sensitivity bounds (low/high) reflect sensitivity to the clutch shift only and are not a confidence interval for make probability.",
        "Single-season data (2025-26 regular season), no defender or shot-clock tracking, baseline make probability does not depend on time or margin (Phase 3 found no detectable effect), and very short clock times may preclude some shot types.",
        "Zone presets average predictions over fixed samples of real regular-play shots; custom single-point options inherit local Model A structure including left-right coordinate differences.",
    ]
    if shift_mode == "by_shot_type":
        assumptions.append(
            "Per-shot-type shift is exploratory, not pre-specified: in this cut 3PT minus 2PT = -0.117 logit (95% CI -0.217 to -0.017; interval close to zero)."
        )
    return assumptions


def evaluate_options(
    state: Union[GameState, dict[str, Any]],
    options: Sequence[Union[str, dict[str, Any]]],
    player_id: Union[int, str, None] = None,
    shift_mode: Literal["overall", "by_shot_type", "none"] = "overall",
    time_gate_sec: float = 24.0,
    model: Optional[ExportedModel] = None,
    presets: Optional[dict[str, dict[str, Any]]] = None,
) -> DecisionResult:
    """Evaluate candidate field-goal options in a given pre-shot game state."""
    game_state = _coerce_state(state)

    if shift_mode not in VALID_SHIFT_MODES:
        raise ValueError(
            f"Unknown shift_mode {shift_mode!r}. Expected one of {VALID_SHIFT_MODES}."
        )
    if isinstance(time_gate_sec, bool) or not isinstance(
        time_gate_sec, (int, float, np.number)
    ):
        raise ValueError(f"time_gate_sec must be numeric, got {time_gate_sec!r}.")
    gate_sec = float(time_gate_sec)
    if not math.isfinite(gate_sec) or gate_sec < 0.0:
        raise ValueError(f"time_gate_sec must be >= 0, got {time_gate_sec!r}.")

    loaded_model = model if model is not None else load_export(DEFAULT_EXPORT_DIR)
    loaded_presets = presets if presets is not None else load_zone_presets(DEFAULT_PRESETS_PATH)
    resolved_options = _resolve_options(options, loaded_presets)

    flags: list[str] = []

    # Resolve shooter offset: unknown player falls back to Model A (offset 0) with flag
    effective_player_id: Optional[str]
    if player_id is None:
        effective_player_id = None
        player_offset = 0.0
    else:
        pid_str = str(player_id)
        if pid_str in loaded_model.player_offsets:
            effective_player_id = pid_str
            player_offset = float(loaded_model.player_offsets[pid_str])
        else:
            effective_player_id = None
            player_offset = 0.0
            flags.append("unknown_player_offset_zero")

    is_clutch = game_state.is_clutch
    if not is_clutch:
        flags.append("shift_not_applied_outside_clutch")
        effective_shift_mode = "none"
    else:
        effective_shift_mode = shift_mode

    shift_applied = effective_shift_mode != "none"

    # Concatenate all rows across options to evaluate cleanly in four vectorized calls
    all_rows: list[dict[str, Any]] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for _, _, shot_rows in resolved_options:
        n_r = len(shot_rows)
        all_rows.extend(shot_rows)
        slices.append((cursor, cursor + n_r))
        cursor += n_r

    p_no_shift_all = predict_make_probability(
        loaded_model,
        all_rows,
        player_id=effective_player_id,
        clutch_shift=False,
    )
    if shift_applied:
        p_point_all = predict_make_probability(
            loaded_model,
            all_rows,
            player_id=effective_player_id,
            clutch_shift=True,
            shift_mode=effective_shift_mode,
            shift_bound="point",
        )
        p_low_all = predict_make_probability(
            loaded_model,
            all_rows,
            player_id=effective_player_id,
            clutch_shift=True,
            shift_mode=effective_shift_mode,
            shift_bound="low",
        )
        p_high_all = predict_make_probability(
            loaded_model,
            all_rows,
            player_id=effective_player_id,
            clutch_shift=True,
            shift_mode=effective_shift_mode,
            shift_bound="high",
        )
    else:
        p_point_all = p_no_shift_all
        p_low_all = p_no_shift_all
        p_high_all = p_no_shift_all

    gate_open = game_state.time_left_sec <= gate_sec
    if not gate_open:
        flags.append("decision_gate_closed")

    labels: list[str] = []
    shot_values: list[int] = []
    n_rows_list: list[int] = []
    p_make_list: list[float] = []
    xp_list: list[float] = []
    p_no_shift_list: list[float] = []
    p_low_list: list[float] = []
    p_high_list: list[float] = []

    p_ahead_list: list[float] = []
    p_tied_list: list[float] = []
    p_behind_list: list[float] = []
    v_list: list[float] = []
    v_no_shift_list: list[float] = []
    v_low_list: list[float] = []
    v_high_list: list[float] = []

    for (label, shot_val, _), (start, end) in zip(resolved_options, slices):
        p_make = float(np.mean(p_point_all[start:end]))
        p_no_shift = float(np.mean(p_no_shift_all[start:end]))
        p_low = float(np.mean(p_low_all[start:end]))
        p_high = float(np.mean(p_high_all[start:end]))

        labels.append(label)
        n_rows_list.append(end - start)
        shot_values.append(shot_val)
        p_make_list.append(p_make)
        xp_list.append(p_make * shot_val)
        p_no_shift_list.append(p_no_shift)
        p_low_list.append(p_low)
        p_high_list.append(p_high)

        if gate_open:
            pa, pt, pb, v_pt = _end_state_probs(p_make, shot_val, game_state.margin)
            _, _, _, v_ns = _end_state_probs(p_no_shift, shot_val, game_state.margin)
            _, _, _, v_lo = _end_state_probs(p_low, shot_val, game_state.margin)
            _, _, _, v_hi = _end_state_probs(p_high, shot_val, game_state.margin)
            p_ahead_list.append(pa)
            p_tied_list.append(pt)
            p_behind_list.append(pb)
            v_list.append(v_pt)
            v_no_shift_list.append(v_ns)
            v_low_list.append(v_lo)
            v_high_list.append(v_hi)
        else:
            p_ahead_list.append(float("nan"))
            p_tied_list.append(float("nan"))
            p_behind_list.append(float("nan"))
            v_list.append(float("nan"))
            v_no_shift_list.append(float("nan"))
            v_low_list.append(float("nan"))
            v_high_list.append(float("nan"))

    if gate_open:
        ranks_v, no_end_state_change, top_point = _rank_indices(labels, v_list, xp_list)
        if no_end_state_change:
            flags.append("no_option_changes_end_state")

        ranks_xp = [0] * len(labels)
        for rank, idx in enumerate(
            sorted(range(len(labels)), key=lambda i: (-xp_list[i], labels[i])),
            start=1,
        ):
            ranks_xp[idx] = rank

        _, _, top_ns = _rank_indices(
            labels, v_no_shift_list, [p * v for p, v in zip(p_no_shift_list, shot_values)]
        )
        _, _, top_lo = _rank_indices(
            labels, v_low_list, [p * v for p, v in zip(p_low_list, shot_values)]
        )
        _, _, top_hi = _rank_indices(
            labels, v_high_list, [p * v for p, v in zip(p_high_list, shot_values)]
        )
        ranking_robust: Optional[bool] = len({top_point, top_ns, top_lo, top_hi}) == 1
        no_option_changes: Optional[bool] = no_end_state_change
        rank_v_series = pd.Series(ranks_v, dtype="Int64")
        rank_xp_series = pd.Series(ranks_xp, dtype="Int64")
    else:
        ranking_robust = None
        no_option_changes = None
        rank_v_series = pd.Series([pd.NA] * len(labels), dtype="Int64")
        rank_xp_series = pd.Series([pd.NA] * len(labels), dtype="Int64")

    table = pd.DataFrame(
        {
            "label": labels,
            "n_rows": pd.Series(n_rows_list, dtype="int64"),
            "shot_value": pd.Series(shot_values, dtype="int64"),
            "p_make": p_make_list,
            "xp": xp_list,
            "p_make_no_shift": p_no_shift_list,
            "p_make_shift_low": p_low_list,
            "p_make_shift_high": p_high_list,
            "p_ahead": p_ahead_list,
            "p_tied": p_tied_list,
            "p_behind": p_behind_list,
            "v": v_list,
            "v_no_shift": v_no_shift_list,
            "v_shift_low": v_low_list,
            "v_shift_high": v_high_list,
            "rank_v": rank_v_series,
            "rank_xp": rank_xp_series,
        }
    )[RESULT_COLUMNS]

    meta: dict[str, Any] = {
        "state": {
            "time_left_sec": game_state.time_left_sec,
            "margin": game_state.margin,
            "period": game_state.period,
        },
        "player_id": None if player_id is None else str(player_id),
        "effective_player_id": effective_player_id,
        "player_offset": player_offset,
        "is_clutch_state": is_clutch,
        "shift_mode": shift_mode,
        "shift_applied": shift_applied,
        "shifts_used": _describe_shifts_used(
            loaded_model, effective_player_id, effective_shift_mode
        ),
        "gate_open": gate_open,
        "time_gate_sec": gate_sec,
        "ranking_robust": ranking_robust,
        "no_option_changes_end_state": no_option_changes,
        "flags": flags,
        "assumptions": _build_assumptions(shift_mode, gate_sec),
    }
    return DecisionResult(table=table, meta=meta)
