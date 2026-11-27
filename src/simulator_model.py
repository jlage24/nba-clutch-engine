"""Phase 4a — Step 3: Portable simulator model loader and inference API.

This module intentionally depends ONLY on the exported artifacts folder
(``export/`` by default), Python stdlib, ``numpy``, ``pandas``, and ``xgboost``.
It does NOT import ``src/features.py``, ``src/baseline_model.py``,
``src/train.py``, or anything that reads ``data/`` or ``models/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Sequence, Union

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = ROOT / "export"


@dataclass(frozen=True)
class ExportedModel:
    """In-memory representation of the portable simulator export."""

    model_a: xgb.XGBClassifier
    player_offsets: dict[str, float]
    player_table: pd.DataFrame
    clutch_shift_a: float
    clutch_shift_a_ci: tuple[float, float]
    clutch_shift_b: float
    clutch_shift_b_ci: tuple[float, float]
    clutch_shift_info: dict[str, Any]
    manifest: dict[str, Any]
    features: list[str]
    action_group_categories: list[str]


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid without scipy dependency."""
    x_arr = np.asarray(x, dtype=float)
    return np.where(
        x_arr >= 0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr)),
    )


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds transform."""
    p_arr = np.asarray(p, dtype=float)
    return np.log(p_arr / (1.0 - p_arr))


def _map_action_group(
    action_type: pd.Series,
    categories: list[str],
) -> pd.Series:
    """Collapse raw ACTION_TYPE strings into ACTION_GROUP categorical levels."""
    s = action_type.astype(str).str.lower()
    conditions = [
        s.str.contains("dunk"),
        s.str.contains("layup|finger roll"),
        s.str.contains("hook"),
        s.str.contains("floating|floater"),
        s.str.contains("step back"),
        s.str.contains("pullup|pull-up"),
        s.str.contains("fadeaway|turnaround"),
        s.str.contains(r"jump.*shot|jumper|bank"),
    ]
    choices = [
        "dunk",
        "layup",
        "hook",
        "floater",
        "stepback",
        "pullup",
        "fadeaway",
        "jump_shot",
    ]
    raw = np.select(conditions, choices, default="other")
    dtype = pd.CategoricalDtype(categories=categories)
    return pd.Series(pd.Categorical(raw, dtype=dtype), index=action_type.index)


def load_export(path: Union[str, Path] = DEFAULT_EXPORT_DIR) -> ExportedModel:
    """Load the portable simulator export and validate its schema against manifest.json."""
    export_dir = Path(path)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    manifest_path = export_dir / "manifest.json"
    model_path = export_dir / "model_a.json"
    offsets_path = export_dir / "player_offsets.csv"
    shift_path = export_dir / "clutch_shift.json"

    for required in (manifest_path, model_path, offsets_path, shift_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing export file: {required}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(shift_path) as f:
        shift_info = json.load(f)

    features = list(manifest["features"])
    categories = list(manifest["action_group_categories"])

    model_a = xgb.XGBClassifier()
    model_a.load_model(str(model_path))

    booster_features = model_a.get_booster().feature_names
    if booster_features != features:
        raise ValueError(
            f"Feature order mismatch between model_a.json ({booster_features}) "
            f"and manifest.json ({features})."
        )

    player_table = pd.read_csv(offsets_path)
    player_offsets = {
        str(pid): float(off)
        for pid, off in zip(player_table["PLAYER_ID"], player_table["offset"])
    }

    sa = shift_info["model_a"]
    sb = shift_info["model_b"]

    return ExportedModel(
        model_a=model_a,
        player_offsets=player_offsets,
        player_table=player_table,
        clutch_shift_a=float(sa["shift_logit"]),
        clutch_shift_a_ci=(float(sa["ci_lower_logit"]), float(sa["ci_upper_logit"])),
        clutch_shift_b=float(sb["shift_logit"]),
        clutch_shift_b_ci=(float(sb["ci_lower_logit"]), float(sb["ci_upper_logit"])),
        clutch_shift_info=shift_info,
        manifest=manifest,
        features=features,
        action_group_categories=categories,
    )


def prepare_features(
    shots: Union[pd.DataFrame, Sequence[dict[str, Any]], dict[str, Any]],
    model: ExportedModel,
    clamp_distance: bool = True,
) -> pd.DataFrame:
    """Transform shot inputs into the exact baseline feature DataFrame.

    Matches ``src/features.py:build_features`` on baseline columns and optionally
    applies the ``src/predict.py`` 1 ft minimum distance clamp for non-putback shots.
    Raises ``ValueError`` if an explicit ``ACTION_GROUP`` contains unknown values.
    """
    if isinstance(shots, dict):
        df = pd.DataFrame([shots])
    elif isinstance(shots, pd.DataFrame):
        df = shots.copy()
    else:
        df = pd.DataFrame(list(shots))

    if df.empty:
        out = pd.DataFrame(columns=model.features)
        out["ACTION_GROUP"] = pd.Categorical(
            [], categories=model.action_group_categories
        )
        return out

    # Polar coordinates (tenths of a foot -> feet, degrees)
    if "LOC_X" in df.columns and "LOC_Y" in df.columns:
        x = df["LOC_X"].astype(float)
        y = df["LOC_Y"].astype(float)
        df["LOC_X"] = x
        df["LOC_Y"] = y
        if "SHOT_DISTANCE_FT" not in df.columns:
            df["SHOT_DISTANCE_FT"] = np.hypot(x, y) / 10.0
        if "SHOT_ANGLE" not in df.columns:
            df["SHOT_ANGLE"] = np.degrees(np.arctan2(x, y))
    elif "SHOT_DISTANCE_FT" not in df.columns or "SHOT_ANGLE" not in df.columns:
        raise KeyError(
            "Either 'LOC_X' and 'LOC_Y' or 'SHOT_DISTANCE_FT' and 'SHOT_ANGLE' must be provided."
        )

    # Shot point value (2 or 3)
    if "SHOT_VALUE" not in df.columns:
        if "SHOT_TYPE" in df.columns:
            df["SHOT_VALUE"] = np.where(df["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2)
        else:
            df["SHOT_VALUE"] = 2

    # Putback / tip indicator
    if "IS_PUTBACK" not in df.columns:
        if "ACTION_TYPE" in df.columns:
            df["IS_PUTBACK"] = (
                df["ACTION_TYPE"]
                .astype(str)
                .str.contains("tip|putback", case=False, na=False)
                .astype(int)
            )
        else:
            df["IS_PUTBACK"] = 0
    else:
        df["IS_PUTBACK"] = df["IS_PUTBACK"].astype(int)

    # Categorical ACTION_GROUP with strict validation against manifest categories
    categories = model.action_group_categories
    cat_dtype = pd.CategoricalDtype(categories=categories)
    if "ACTION_GROUP" in df.columns:
        raw_ag = df["ACTION_GROUP"]
        if raw_ag.isna().any():
            raise ValueError(
                f"Unknown ACTION_GROUP value(s): NaN/None found. Expected one of {categories}."
            )
        allowed = set(categories)
        vals_str = raw_ag.astype(str)
        unknown = [v for v in vals_str.unique() if v not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown ACTION_GROUP value(s): {unknown!r}. Expected one of {categories}."
            )
        df["ACTION_GROUP"] = pd.Series(
            pd.Categorical(vals_str, dtype=cat_dtype), index=df.index
        )
    elif "ACTION_TYPE" in df.columns:
        df["ACTION_GROUP"] = _map_action_group(df["ACTION_TYPE"], categories)
    else:
        df["ACTION_GROUP"] = pd.Series(
            pd.Categorical(["other"] * len(df), dtype=cat_dtype), index=df.index
        )

    # Same 1 ft clamp guardrail as src/predict.py for non-putback rim attempts
    if clamp_distance:
        mask_clamp = (df["IS_PUTBACK"] == 0) & (df["SHOT_DISTANCE_FT"] < 1.0)
        if mask_clamp.any():
            df["SHOT_DISTANCE_FT"] = df["SHOT_DISTANCE_FT"].astype(float)
            df.loc[mask_clamp, "SHOT_DISTANCE_FT"] = 1.0

    for col in model.features:
        if col not in df.columns:
            raise KeyError(f"Missing required feature: '{col}'")

    return df[model.features]


def _resolve_player_offsets(
    player_id: Union[int, str, Sequence[Any], pd.Series, np.ndarray],
    n_shots: int,
    offsets_map: dict[str, float],
) -> np.ndarray:
    """Resolve scalar or per-shot player_id into an array of offsets."""
    if isinstance(player_id, (int, np.integer, str)):
        return np.full(n_shots, offsets_map.get(str(player_id), 0.0), dtype=float)

    pid_list = list(player_id)
    if len(pid_list) != n_shots:
        raise ValueError(
            f"Length of player_id ({len(pid_list)}) must match number of shots ({n_shots})."
        )
    resolved = np.zeros(n_shots, dtype=float)
    for i, pid in enumerate(pid_list):
        if pid is not None and not pd.isna(pid):
            resolved[i] = offsets_map.get(str(pid), 0.0)
    return resolved


_SHIFT_BOUND_KEYS = {
    "point": "shift_logit",
    "low": "ci_lower_logit",
    "high": "ci_upper_logit",
}


def _resolve_shift(
    model: ExportedModel,
    shot_values: np.ndarray,
    use_model_b: bool,
    shift_mode: str,
    shift_bound: str,
) -> Union[float, np.ndarray]:
    """Return scalar or per-shot logit shift from exported clutch_shift.json."""
    if shift_mode not in {"overall", "by_shot_type", "none"}:
        raise ValueError(
            f"Unknown shift_mode {shift_mode!r}. Expected 'overall', 'by_shot_type', or 'none'."
        )
    if shift_bound not in _SHIFT_BOUND_KEYS:
        raise ValueError(
            f"Unknown shift_bound {shift_bound!r}. Expected 'point', 'low', or 'high'."
        )
    if shift_mode == "none":
        return 0.0

    bound_key = _SHIFT_BOUND_KEYS[shift_bound]
    prefix = "model_b" if use_model_b else "model_a"
    if shift_mode == "overall":
        return float(model.clutch_shift_info[prefix][bound_key])

    exp = model.clutch_shift_info["exploratory"]
    shift_2pt = float(exp[f"{prefix}_2pt"][bound_key])
    shift_3pt = float(exp[f"{prefix}_3pt"][bound_key])
    return np.where(shot_values == 3, shift_3pt, shift_2pt)


def predict_make_probability(
    model: ExportedModel,
    shots: Union[pd.DataFrame, Sequence[dict[str, Any]], dict[str, Any]],
    player_id: Union[int, str, Sequence[Any], pd.Series, np.ndarray, None] = None,
    clutch_shift: bool = True,
    shift_mode: Literal["overall", "by_shot_type", "none"] = "overall",
    shift_bound: Literal["point", "low", "high"] = "point",
) -> np.ndarray:
    """Predict shot make probabilities for one or more shots.

    Parameters
    ----------
    model:
        Loaded ``ExportedModel`` from ``load_export``.
    shots:
        DataFrame, list of shot dicts, or a single shot dict.
    player_id:
        Optional NBA player ID (scalar or sequence matching ``shots``). When
        ``None``, uses Model A (no shooter offset). When provided, uses Model B
        (unknown/absent players receive offset 0 so ``p_B == p_A``).
    clutch_shift:
        If True (default), adds the logit-scale clutch shift selected by
        ``shift_mode`` and ``shift_bound``; if False, returns the baseline
        ``p_A`` or ``p_B`` probability.
    shift_mode:
        ``"overall"`` (default), ``"by_shot_type"`` (exploratory 2PT/3PT split),
        or ``"none"`` (no shift).
    shift_bound:
        ``"point"`` (default), ``"low"`` (95% CI lower bound), or ``"high"``
        (95% CI upper bound) of the selected clutch shift.
    """
    if shift_mode not in {"overall", "by_shot_type", "none"}:
        raise ValueError(
            f"Unknown shift_mode {shift_mode!r}. Expected 'overall', 'by_shot_type', or 'none'."
        )
    if shift_bound not in _SHIFT_BOUND_KEYS:
        raise ValueError(
            f"Unknown shift_bound {shift_bound!r}. Expected 'point', 'low', or 'high'."
        )

    X = prepare_features(shots, model, clamp_distance=True)
    if X.empty:
        return np.empty(0, dtype=float)

    p_a = model.model_a.predict_proba(X)[:, 1].astype(float)

    pred_cfg = model.manifest["prediction_formula"]
    p_low = float(pred_cfg["p_b_clip_low"])
    p_high = float(pred_cfg["p_b_clip_high"])
    eps = float(pred_cfg["logit_clip_eps"])

    use_model_b = player_id is not None
    if not use_model_b:
        p_base = p_a
    else:
        offsets = _resolve_player_offsets(player_id, len(X), model.player_offsets)
        p_base = np.where(
            offsets == 0.0,
            p_a,
            np.clip(p_a + offsets, p_low, p_high),
        )

    if not clutch_shift or shift_mode == "none":
        return p_base

    shift = _resolve_shift(
        model=model,
        shot_values=X["SHOT_VALUE"].to_numpy(dtype=int),
        use_model_b=use_model_b,
        shift_mode=shift_mode,
        shift_bound=shift_bound,
    )
    if isinstance(shift, float) and shift == 0.0:
        return p_base

    eta = _logit(np.clip(p_base, eps, 1.0 - eps))
    return _expit(eta + shift)
