"""Phase 4b — Step 2: Export representative zone-preset shot samples.

Builds ``export/zone_presets.json`` from ``REGULAR`` (non-clutch, non-blowout)
shots only, using shot-context columns only (never ``SHOT_MADE_FLAG``). Each
preset stores a fixed random sample of real shots from its zone (300 rows by
default, redrawn to 1,000 rows if ``|mean_p_A_sample - mean_p_A_zone| > 0.003``)
containing strictly the seven baseline features from ``all_shots.parquet``.

Run from the repo root::

    python src/export_zone_presets.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shot_geometry import compute_shot_distance_and_angle  # noqa: E402
from src.simulator_model import (  # noqa: E402
    DEFAULT_EXPORT_DIR,
    ExportedModel,
    load_export,
    predict_make_probability,
)

DATA_DIR = ROOT / "data"
ALL_SHOTS_PATH = DATA_DIR / "processed" / "all_shots.parquet"
OUT_PATH = DEFAULT_EXPORT_DIR / "zone_presets.json"

RNG_SEED = 42
INITIAL_SAMPLE_SIZE = 300
REDRAW_SAMPLE_SIZE = 1000
TARGET_ABS_PA_DIFF = 0.003
MAX_ABS_PA_DIFF = 0.005
MAX_FILE_BYTES = 1_000_000

BASELINE_FEATURES = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE_FT",
    "SHOT_ANGLE",
    "SHOT_VALUE",
    "ACTION_GROUP",
    "IS_PUTBACK",
]

PRESET_SPECS: list[tuple[str, str, str]] = [
    (
        "restricted_area",
        "Restricted Area",
        "REGULAR & SHOT_ZONE_BASIC == 'Restricted Area' & IS_PUTBACK == 0 & SHOT_VALUE == 2",
    ),
    (
        "putback",
        "Putback / Tip",
        "REGULAR & IS_PUTBACK == 1 & SHOT_VALUE == 2",
    ),
    (
        "paint",
        "Paint (Non-RA)",
        "REGULAR & SHOT_ZONE_BASIC == 'In The Paint (Non-RA)' & IS_PUTBACK == 0 & SHOT_VALUE == 2",
    ),
    (
        "short_mid_range",
        "Short Mid-Range (8-16 ft)",
        "REGULAR & SHOT_ZONE_BASIC == 'Mid-Range' & SHOT_ZONE_RANGE == '8-16 ft.' & IS_PUTBACK == 0 & SHOT_VALUE == 2",
    ),
    (
        "long_mid_range",
        "Long Mid-Range (16-24 ft)",
        "REGULAR & SHOT_ZONE_BASIC == 'Mid-Range' & SHOT_ZONE_RANGE == '16-24 ft.' & IS_PUTBACK == 0 & SHOT_VALUE == 2",
    ),
    (
        "corner_three",
        "Corner Three",
        "REGULAR & SHOT_ZONE_BASIC in {'Left Corner 3', 'Right Corner 3'} & SHOT_VALUE == 3",
    ),
    (
        "above_the_break_3",
        "Above-the-Break Three",
        "REGULAR & SHOT_ZONE_BASIC == 'Above the Break 3' & SHOT_VALUE == 3",
    ),
]


def load_regular_shots_with_zones() -> pd.DataFrame:
    """Load REGULAR shots joined 1-to-1 with raw zone columns (without SHOT_MADE_FLAG)."""
    if not ALL_SHOTS_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {ALL_SHOTS_PATH}")

    all_cols = ["GAME_ID", "GAME_EVENT_ID", "REGULAR", *BASELINE_FEATURES]
    all_df = pd.read_parquet(ALL_SHOTS_PATH, columns=all_cols)
    all_df["GAME_ID"] = all_df["GAME_ID"].astype(str)

    raw_files = sorted(DATA_DIR.glob("raw_shots_*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw_shots_*.parquet files found under {DATA_DIR}")

    raw_frames = []
    for path in raw_files:
        raw_part = pd.read_parquet(
            path,
            columns=["GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC", "SHOT_ZONE_RANGE"],
        )
        raw_part["GAME_ID"] = raw_part["GAME_ID"].astype(str)
        raw_frames.append(raw_part)
    raw_df = pd.concat(raw_frames, ignore_index=True)

    merged = all_df.merge(raw_df, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
    if merged["SHOT_ZONE_BASIC"].isna().any():
        raise RuntimeError("Missing SHOT_ZONE_BASIC after joining all_shots to raw_shots.")

    reg = (
        merged[merged["REGULAR"]]
        .sort_values(["GAME_ID", "GAME_EVENT_ID"])
        .reset_index(drop=True)
    )
    return reg


def _zone_masks(reg: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "restricted_area": (
            (reg["SHOT_ZONE_BASIC"] == "Restricted Area")
            & (reg["IS_PUTBACK"] == 0)
            & (reg["SHOT_VALUE"] == 2)
        ),
        "putback": (reg["IS_PUTBACK"] == 1) & (reg["SHOT_VALUE"] == 2),
        "paint": (
            (reg["SHOT_ZONE_BASIC"] == "In The Paint (Non-RA)")
            & (reg["IS_PUTBACK"] == 0)
            & (reg["SHOT_VALUE"] == 2)
        ),
        "short_mid_range": (
            (reg["SHOT_ZONE_BASIC"] == "Mid-Range")
            & (reg["SHOT_ZONE_RANGE"] == "8-16 ft.")
            & (reg["IS_PUTBACK"] == 0)
            & (reg["SHOT_VALUE"] == 2)
        ),
        "long_mid_range": (
            (reg["SHOT_ZONE_BASIC"] == "Mid-Range")
            & (reg["SHOT_ZONE_RANGE"] == "16-24 ft.")
            & (reg["IS_PUTBACK"] == 0)
            & (reg["SHOT_VALUE"] == 2)
        ),
        "corner_three": (
            reg["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3"])
            & (reg["SHOT_VALUE"] == 3)
        ),
        "above_the_break_3": (
            (reg["SHOT_ZONE_BASIC"] == "Above the Break 3")
            & (reg["SHOT_VALUE"] == 3)
        ),
    }


def _side_shares(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    if n == 0:
        return {"left": 0.0, "right": 0.0, "center": 0.0}
    return {
        "left": round(float((df["LOC_X"] < 0).sum() / n), 6),
        "right": round(float((df["LOC_X"] > 0).sum() / n), 6),
        "center": round(float((df["LOC_X"] == 0).sum() / n), 6),
    }


def _mirror_diagnostic(
    name: str,
    zone_df: pd.DataFrame,
    model: ExportedModel,
) -> dict[str, Any]:
    """Compute the audit single-point mirror diagnostic (stored as information only)."""
    n_left = int((zone_df["LOC_X"] < 0).sum())
    n_right = int((zone_df["LOC_X"] > 0).sum())
    if name == "putback":
        chosen_side = "center"
        sign = 0.0
    elif n_left >= n_right:
        chosen_side = "left"
        sign = -1.0
    else:
        chosen_side = "right"
        sign = 1.0

    ag = str(zone_df["ACTION_GROUP"].value_counts().index[0])
    sv = int(zone_df["SHOT_VALUE"].mode().iloc[0])
    pb = int(zone_df["IS_PUTBACK"].mode().iloc[0])
    med_euclid = float((np.hypot(zone_df["LOC_X"], zone_df["LOC_Y"]) / 10.0).median())
    med_abs_ang = float(zone_df["SHOT_ANGLE"].abs().median())

    if pb == 1 and med_euclid == 0.0:
        px, py = 0.0, 0.0
    else:
        rad = np.radians(sign * med_abs_ang)
        px = round(float(med_euclid * 10.0 * np.sin(rad)), 1)
        py = round(float(med_euclid * 10.0 * np.cos(rad)), 1)

    dist, ang = compute_shot_distance_and_angle(px, py, is_putback=pb, clamp_distance=True)
    shot_chosen = {
        "LOC_X": px,
        "LOC_Y": py,
        "SHOT_DISTANCE_FT": dist,
        "SHOT_ANGLE": ang,
        "SHOT_VALUE": sv,
        "ACTION_GROUP": ag,
        "IS_PUTBACK": pb,
    }
    shot_mirror = {
        "LOC_X": -px,
        "LOC_Y": py,
        "SHOT_DISTANCE_FT": dist,
        "SHOT_ANGLE": -ang if ang != 0.0 else 0.0,
        "SHOT_VALUE": sv,
        "ACTION_GROUP": ag,
        "IS_PUTBACK": pb,
    }
    p_chosen = float(predict_make_probability(model, shot_chosen, clutch_shift=False)[0])
    p_mirror = float(predict_make_probability(model, shot_mirror, clutch_shift=False)[0])
    return {
        "note": "Audit diagnostic on a single polar-consistent point (information only; presets use real-shot samples).",
        "side_rule": "Side with more shots in the zone (ties to left); center for putback.",
        "chosen_side": chosen_side,
        "loc_x": px,
        "loc_y": py,
        "shot_distance_ft": float(dist),
        "shot_angle": round(float(ang), 6),
        "action_group": ag,
        "p_a_chosen": round(p_chosen, 6),
        "p_a_mirror": round(p_mirror, 6),
        "abs_mirror_diff": round(abs(p_chosen - p_mirror), 6),
    }


def _records_from_sample(sample_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in sample_df[BASELINE_FEATURES].itertuples(index=False):
        records.append(
            {
                "LOC_X": float(row.LOC_X),
                "LOC_Y": float(row.LOC_Y),
                "SHOT_DISTANCE_FT": int(row.SHOT_DISTANCE_FT),
                "SHOT_ANGLE": float(row.SHOT_ANGLE),
                "SHOT_VALUE": int(row.SHOT_VALUE),
                "ACTION_GROUP": str(row.ACTION_GROUP),
                "IS_PUTBACK": int(row.IS_PUTBACK),
            }
        )
    return records


def build_zone_presets(
    model: ExportedModel | None = None,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Build the zone presets dictionary from REGULAR shots and exported Model A."""
    if model is None:
        model = load_export(DEFAULT_EXPORT_DIR)

    reg = load_regular_shots_with_zones()
    masks = _zone_masks(reg)
    rng = np.random.default_rng(seed)

    presets_out: dict[str, Any] = {}
    for name, label, filter_desc in PRESET_SPECS:
        zone_df = reg[masks[name]].reset_index(drop=True)
        n_zone = len(zone_df)
        if n_zone == 0:
            raise RuntimeError(f"Zone '{name}' matched 0 REGULAR shots.")

        p_zone = predict_make_probability(model, zone_df[BASELINE_FEATURES], clutch_shift=False)
        mean_p_zone = float(np.mean(p_zone))
        sd_p_zone = float(np.std(p_zone, ddof=1))

        n_draw = min(INITIAL_SAMPLE_SIZE, n_zone)
        idx = rng.choice(n_zone, size=n_draw, replace=False)
        sample_df = zone_df.iloc[idx].reset_index(drop=True)
        records = _records_from_sample(sample_df)
        p_sample = predict_make_probability(model, records, clutch_shift=False)
        mean_p_sample = float(np.mean(p_sample))
        abs_diff = abs(mean_p_sample - mean_p_zone)
        redrawn = False

        if abs_diff > TARGET_ABS_PA_DIFF and n_zone > INITIAL_SAMPLE_SIZE:
            n_draw = min(REDRAW_SAMPLE_SIZE, n_zone)
            idx = rng.choice(n_zone, size=n_draw, replace=False)
            sample_df = zone_df.iloc[idx].reset_index(drop=True)
            records = _records_from_sample(sample_df)
            p_sample = predict_make_probability(model, records, clutch_shift=False)
            mean_p_sample = float(np.mean(p_sample))
            abs_diff = abs(mean_p_sample - mean_p_zone)
            redrawn = True

        if abs_diff > MAX_ABS_PA_DIFF:
            raise RuntimeError(
                f"Preset '{name}' exceeds {MAX_ABS_PA_DIFF} mean p_A diff "
                f"after {n_draw} rows: {abs_diff:.6f}."
            )

        se_sample_mean = sd_p_zone / float(np.sqrt(n_draw))
        abs_diff_over_se = abs_diff / se_sample_mean if se_sample_mean > 0.0 else 0.0

        presets_out[name] = {
            "label": label,
            "zone_filter": filter_desc,
            "shot_value": int(zone_df["SHOT_VALUE"].mode().iloc[0]),
            "is_putback": int(zone_df["IS_PUTBACK"].mode().iloc[0]),
            "modal_action_group": str(zone_df["ACTION_GROUP"].value_counts().index[0]),
            "n_zone_shots": int(n_zone),
            "n_sample": int(n_draw),
            "redrawn_to_1000": redrawn,
            "mean_p_a_zone": round(mean_p_zone, 6),
            "sd_p_a_zone": round(sd_p_zone, 6),
            "se_sample_mean": round(se_sample_mean, 6),
            "mean_p_a_sample": round(mean_p_sample, 6),
            "abs_mean_p_a_diff": round(abs_diff, 6),
            "abs_diff_over_se": round(abs_diff_over_se, 4),
            "side_shares_zone": _side_shares(zone_df),
            "side_shares_sample": _side_shares(sample_df),
            "mirror_diagnostic": _mirror_diagnostic(name, zone_df, model),
            "shots": records,
        }

    return {
        "meta": {
            "schema_version": "1.0.0",
            "seed": seed,
            "source": "REGULAR shots in data/processed/all_shots.parquet joined to data/raw_shots_*.parquet on (GAME_ID, GAME_EVENT_ID)",
            "selection_rule": (
                f"Draw {INITIAL_SAMPLE_SIZE} real REGULAR shots without replacement using "
                f"numpy.random.default_rng({seed}); if |mean_p_a_sample - mean_p_a_zone| > "
                f"{TARGET_ABS_PA_DIFF}, redraw with {REDRAW_SAMPLE_SIZE} rows."
            ),
            "target_abs_p_a_diff": TARGET_ABS_PA_DIFF,
            "max_abs_p_a_diff": MAX_ABS_PA_DIFF,
            "shot_features": BASELINE_FEATURES,
            "no_outcomes_or_ids": True,
            "n_regular_shots_total": int(len(reg)),
            "n_presets": len(presets_out),
        },
        "presets": presets_out,
    }


def main() -> None:
    payload = build_zone_presets()
    serialized = json.dumps(payload, separators=(",", ":"))
    size_bytes = len(serialized.encode("utf-8"))
    if size_bytes > MAX_FILE_BYTES:
        raise RuntimeError(
            f"Serialized zone_presets.json is {size_bytes:,} bytes (> {MAX_FILE_BYTES:,} bytes)."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(serialized + "\n", encoding="utf-8")

    print(f"Wrote {OUT_PATH} ({size_bytes:,} bytes, {size_bytes / 1024:.1f} KB)")
    print(
        f"{'Preset':<24} {'n_zone':>7} {'n_samp':>7} {'pA_zone':>8} {'pA_samp':>8} "
        f"{'abs_diff':>9} {'L/R/C zone':<21} {'L/R/C samp':<21}"
    )
    print("-" * 112)
    for name, entry in payload["presets"].items():
        sz = entry["side_shares_zone"]
        ss = entry["side_shares_sample"]
        z_str = f"{sz['left']:.3f}/{sz['right']:.3f}/{sz['center']:.3f}"
        s_str = f"{ss['left']:.3f}/{ss['right']:.3f}/{ss['center']:.3f}"
        print(
            f"{name:<24} {entry['n_zone_shots']:>7,} {entry['n_sample']:>7,} "
            f"{entry['mean_p_a_zone']:>8.6f} {entry['mean_p_a_sample']:>8.6f} "
            f"{entry['abs_mean_p_a_diff']:>9.6f} {z_str:<21} {s_str:<21}"
        )


if __name__ == "__main__":
    main()
