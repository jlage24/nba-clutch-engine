"""Phase 4c — Step 2: Export court-level action-group mix defaults and density mask.

Builds ``export/court_defaults.json`` from ``REGULAR`` (non-clutch, non-blowout)
shots in ``data/processed/all_shots.parquet`` using shot-context columns only
(never ``SHOT_MADE_FLAG`` or player/game identifiers in the exported JSON):

1. ``action_group_mix``: for each integer ``SHOT_DISTANCE_FT``, the share of each
   ``ACTION_GROUP`` among ``REGULAR`` non-putback shots (``IS_PUTBACK == 0``),
   stored as integer weights summing to exactly ``1,000`` via largest-remainder
   rounding, alongside the underlying shot counts. Distances with ``>= 100`` such
   shots use their per-foot distribution; all others use the pooled distribution
   of ``REGULAR`` non-putback shots at ``>= 22`` ft.
2. ``density_mask``: a ``25 x 24`` grid of ``2 ft`` (20 tenths) cells covering
   ``LOC_X`` in ``[-250, 250]`` and ``LOC_Y`` in ``[-50, 430]``, storing raw
   ``REGULAR`` shot counts and the boolean mask ``count >= 10``.
3. ``checks``:
   - ``cell_level_typical_mix_vs_actual`` (primary check): for every 2 ft cell
     with ``>= 100`` ``REGULAR`` non-putback shots, evaluates the distance-based
     typical mix at each shot's real coordinates and compares mean Model A ``p_A``
     against the actual ``ACTION_GROUP`` (non-putback ``REGULAR`` shots on both
     sides).
   - ``cell_centre_diagnostic``: diagnostic comparing all ``REGULAR`` shots in
     ``>= 100``-shot cells against typical-mix ``p_A`` at the 2 ft cell centre
     (measures within-cell geometry/bunching as well as shot-type mix).
   - ``modal_comparison``: Step 1.4 distance-modal check retained for comparison.

Run from the repo root::

    python src/export_court_defaults.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shot_geometry import compute_shot_distance_and_angle, compute_shot_value  # noqa: E402
from src.simulator_model import (  # noqa: E402
    DEFAULT_EXPORT_DIR,
    ExportedModel,
    load_export,
    predict_make_probability,
)

DATA_DIR = ROOT / "data"
ALL_SHOTS_PATH = DATA_DIR / "processed" / "all_shots.parquet"
OUT_PATH = DEFAULT_EXPORT_DIR / "court_defaults.json"

RNG_SEED = 42
WEIGHT_SCALE = 1_000
MIN_PER_FOOT_SHOTS = 100
POOLED_MIN_DISTANCE_FT = 22

GRID_X_MIN = -250.0
GRID_X_MAX = 250.0
GRID_Y_MIN = -50.0
GRID_Y_MAX = 430.0
CELL_SIZE_TENTHS = 20.0
MIN_MASK_SHOTS = 10

CELL_CHECK_MIN_SHOTS = 100
CELL_CHECK_DIFF_THRESHOLD = 0.03

MODAL_MIN_SHOTS = 30
MODAL_FALLBACK_GROUP = "jump_shot"
MODAL_SAMPLE_SIZE = 20_000
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


def load_regular_shots() -> pd.DataFrame:
    """Load REGULAR shots with context columns only (never SHOT_MADE_FLAG)."""
    if not ALL_SHOTS_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {ALL_SHOTS_PATH}")

    cols = ["GAME_ID", "GAME_EVENT_ID", "REGULAR", *BASELINE_FEATURES]
    df = pd.read_parquet(ALL_SHOTS_PATH, columns=cols)
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    reg = (
        df[df["REGULAR"]]
        .sort_values(["GAME_ID", "GAME_EVENT_ID"])
        .reset_index(drop=True)
    )
    return reg


def largest_remainder_weights(
    counts: dict[str, int],
    categories: list[str],
    scale: int = WEIGHT_SCALE,
) -> dict[str, int]:
    """Convert non-negative counts into integer weights summing to ``scale``."""
    total = sum(counts[cat] for cat in categories)
    if total <= 0:
        raise ValueError("Cannot compute weights from zero total count.")

    exact = {cat: (counts[cat] / total) * scale for cat in categories}
    weights = {cat: int(math.floor(exact[cat])) for cat in categories}
    remainder = scale - sum(weights.values())

    order = sorted(
        categories,
        key=lambda cat: (
            -(exact[cat] - weights[cat]),
            -counts[cat],
            categories.index(cat),
        ),
    )
    for cat in order[:remainder]:
        weights[cat] += 1
    return weights


def _count_groups(df: pd.DataFrame, categories: list[str]) -> dict[str, int]:
    vc = df["ACTION_GROUP"].astype(str).value_counts()
    return {cat: int(vc.get(cat, 0)) for cat in categories}


def _shares_summary(df: pd.DataFrame, categories: list[str]) -> dict[str, dict[str, Any]]:
    counts = _count_groups(df, categories)
    total = len(df)
    return {
        cat: {
            "count": counts[cat],
            "share": round(counts[cat] / total, 6) if total > 0 else 0.0,
        }
        for cat in categories
    }


def assign_2ft_grid_indices(
    loc_x: np.ndarray,
    loc_y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map (LOC_X, LOC_Y) to 2 ft cell indices and a boolean in-box mask."""
    x = np.asarray(loc_x, dtype=float)
    y = np.asarray(loc_y, dtype=float)
    in_box = (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1
    ix = np.where(x == x_edges[-1], nx - 1, ix)
    iy = np.where(y == y_edges[-1], ny - 1, iy)
    return ix, iy, in_box


def _build_action_group_mix(
    non_pb: pd.DataFrame,
    categories: list[str],
) -> tuple[dict[str, Any], int, int]:
    pool_df = non_pb[non_pb["SHOT_DISTANCE_FT"] >= POOLED_MIN_DISTANCE_FT]
    pool_counts = _count_groups(pool_df, categories)
    pool_weights = largest_remainder_weights(pool_counts, categories, WEIGHT_SCALE)

    max_dist = int(non_pb["SHOT_DISTANCE_FT"].max())
    grouped_counts = (
        non_pb.groupby(["SHOT_DISTANCE_FT", "ACTION_GROUP"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    by_distance: dict[str, dict[str, Any]] = {}
    per_foot_dists: list[int] = []
    for dist in range(0, max_dist + 1):
        if dist in grouped_counts.index:
            row = grouped_counts.loc[dist]
            counts = {cat: int(row.get(cat, 0)) for cat in categories}
        else:
            counts = {cat: 0 for cat in categories}
        n_shots = sum(counts.values())
        if n_shots >= MIN_PER_FOOT_SHOTS:
            source = "per_foot"
            weights = largest_remainder_weights(counts, categories, WEIGHT_SCALE)
            per_foot_dists.append(dist)
        else:
            source = "pooled_ge_22ft"
            weights = dict(pool_weights)

        by_distance[str(dist)] = {
            "distance_ft": dist,
            "n_shots": n_shots,
            "source": source,
            "group_counts": counts,
            "weights_per_1000": weights,
        }

    mix_payload = {
        "categories": categories,
        "pooled_ge_22ft": {
            "n_shots": int(len(pool_df)),
            "group_counts": pool_counts,
            "weights_per_1000": pool_weights,
        },
        "by_distance_ft": by_distance,
    }
    return mix_payload, min(per_foot_dists), max(per_foot_dists)


def _compute_real_coords_cell_check(
    non_pb: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mix_by_dist: dict[str, dict[str, Any]],
    pooled_weights: dict[str, int],
    categories: list[str],
    model: ExportedModel,
) -> dict[str, Any]:
    """Primary check: evaluate distance typical mix at real shot coordinates on non-putbacks."""
    ix, iy, in_box = assign_2ft_grid_indices(
        non_pb["LOC_X"].to_numpy(dtype=float),
        non_pb["LOC_Y"].to_numpy(dtype=float),
        x_edges,
        y_edges,
    )
    npb_box = non_pb[in_box].copy().reset_index(drop=True)
    npb_box["cell_ix"] = ix[in_box]
    npb_box["cell_iy"] = iy[in_box]
    npb_box["_is_dunk"] = (npb_box["ACTION_GROUP"].astype(str) == "dunk").astype(float)
    npb_box["_p_actual"] = predict_make_probability(
        model, npb_box[BASELINE_FEATURES], clutch_shift=False
    )

    p_mix = np.zeros(len(npb_box), dtype=float)
    shot_dists = npb_box["SHOT_DISTANCE_FT"].to_numpy(dtype=int)
    for cat in categories:
        w_arr = np.array(
            [
                mix_by_dist[str(d)]["weights_per_1000"][cat]
                if str(d) in mix_by_dist
                else pooled_weights[cat]
                for d in shot_dists
            ],
            dtype=float,
        ) / float(WEIGHT_SCALE)
        if np.any(w_arr > 0):
            tmp = npb_box[BASELINE_FEATURES].copy()
            tmp["ACTION_GROUP"] = cat
            p_mix += w_arr * predict_make_probability(model, tmp, clutch_shift=False)
    npb_box["_p_mix"] = p_mix

    grouped = (
        npb_box.groupby(["cell_ix", "cell_iy"])
        .agg(
            n_shots=("_p_actual", "size"),
            mean_p_a_actual=("_p_actual", "mean"),
            mean_p_a_typical_mix=("_p_mix", "mean"),
            dunk_share=("_is_dunk", "mean"),
        )
        .reset_index()
    )
    checked = grouped[grouped["n_shots"] >= CELL_CHECK_MIN_SHOTS].copy()

    records: list[dict[str, Any]] = []
    weighted_abs_sum = 0.0
    total_shots = 0
    for row in checked.itertuples(index=False):
        i = int(row.cell_ix)
        j = int(row.cell_iy)
        n_cell = int(row.n_shots)
        p_act = float(row.mean_p_a_actual)
        p_m = float(row.mean_p_a_typical_mix)
        signed_diff = p_m - p_act
        abs_diff = abs(signed_diff)
        weighted_abs_sum += abs_diff * n_cell
        total_shots += n_cell
        records.append(
            {
                "ix": i,
                "iy": j,
                "center_x": float(0.5 * (x_edges[i] + x_edges[i + 1])),
                "center_y": float(0.5 * (y_edges[j] + y_edges[j + 1])),
                "n_shots": n_cell,
                "dunk_share": round(float(row.dunk_share), 6),
                "mean_p_a_actual": round(p_act, 6),
                "mean_p_a_typical_mix": round(p_m, 6),
                "diff": round(signed_diff, 6),
                "abs_diff": round(abs_diff, 6),
            }
        )

    records.sort(key=lambda item: (-item["abs_diff"], -item["n_shots"], item["ix"], item["iy"]))
    cells_ge_thresh = [r for r in records if r["abs_diff"] >= CELL_CHECK_DIFF_THRESHOLD]
    max_cell = records[0]

    return {
        "description": (
            "For every 2 ft cell with >= 100 REGULAR non-putback shots, evaluate the "
            "distance-based typical mix at each shot's real coordinates and compare mean "
            "Model A p_A against the actual ACTION_GROUP (REGULAR, IS_PUTBACK == 0 on both sides)."
        ),
        "subset": "REGULAR & IS_PUTBACK == 0",
        "min_cell_shots": CELL_CHECK_MIN_SHOTS,
        "n_cells_checked": len(records),
        "n_shots_covered": total_shots,
        "share_non_putback_regular_covered": round(total_shots / len(non_pb), 6),
        "weighted_mean_abs_diff": round(weighted_abs_sum / total_shots, 6),
        "max_abs_diff": max_cell["abs_diff"],
        "diff_threshold": CELL_CHECK_DIFF_THRESHOLD,
        "n_cells_ge_threshold": len(cells_ge_thresh),
        "cells_ge_threshold": cells_ge_thresh,
        "max_diff_cell": max_cell,
    }


def _compute_cell_centre_diagnostic(
    reg: pd.DataFrame,
    ix_all: np.ndarray,
    iy_all: np.ndarray,
    in_box: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    counts_2d: np.ndarray,
    mix_by_dist: dict[str, dict[str, Any]],
    pooled_weights: dict[str, int],
    categories: list[str],
    model: ExportedModel,
) -> dict[str, Any]:
    """Diagnostic comparing all REGULAR shots in >=100-shot cells against cell-centre typical mix."""
    reg_box = reg[in_box].copy().reset_index(drop=True)
    reg_box["_ix"] = ix_all[in_box]
    reg_box["_iy"] = iy_all[in_box]
    reg_box["_p_actual"] = predict_make_probability(
        model, reg_box[BASELINE_FEATURES], clutch_shift=False
    )
    actual_means = reg_box.groupby(["_ix", "_iy"])["_p_actual"].mean()

    cells_ge_100 = np.argwhere(counts_2d >= CELL_CHECK_MIN_SHOTS)
    cell_meta: list[tuple[int, int, float, float, int, int]] = []
    eval_rows: list[dict[str, Any]] = []
    eval_cell_idx: list[int] = []
    eval_weights: list[int] = []

    for i, j in cells_ge_100:
        cx = float(0.5 * (x_edges[i] + x_edges[i + 1]))
        cy = float(0.5 * (y_edges[j] + y_edges[j + 1]))
        dist_ft, angle = compute_shot_distance_and_angle(
            cx, cy, is_putback=0, clamp_distance=True
        )
        shot_val = compute_shot_value(cx, cy)
        dist_key = str(int(dist_ft))
        weights = (
            mix_by_dist[dist_key]["weights_per_1000"]
            if dist_key in mix_by_dist
            else pooled_weights
        )
        cidx = len(cell_meta)
        cell_meta.append((int(i), int(j), cx, cy, int(dist_ft), int(counts_2d[i, j])))
        for cat in categories:
            w = int(weights[cat])
            if w > 0:
                eval_cell_idx.append(cidx)
                eval_weights.append(w)
                eval_rows.append(
                    {
                        "LOC_X": cx,
                        "LOC_Y": cy,
                        "SHOT_DISTANCE_FT": float(dist_ft),
                        "SHOT_ANGLE": float(angle),
                        "SHOT_VALUE": int(shot_val),
                        "ACTION_GROUP": cat,
                        "IS_PUTBACK": 0,
                    }
                )

    p_eval = predict_make_probability(model, eval_rows, clutch_shift=False)
    mix_num = np.zeros(len(cell_meta), dtype=float)
    for cidx, w, p_val in zip(eval_cell_idx, eval_weights, p_eval):
        mix_num[cidx] += w * float(p_val)
    p_mix_center = mix_num / float(WEIGHT_SCALE)

    diffs: list[dict[str, Any]] = []
    weighted_abs_sum = 0.0
    total_shots = 0
    max_abs_diff = 0.0
    for cidx, (i, j, cx, cy, dist_ft, n_cell) in enumerate(cell_meta):
        p_act = float(actual_means.loc[(i, j)])
        p_mix = float(p_mix_center[cidx])
        signed_diff = p_mix - p_act
        abs_diff = abs(signed_diff)
        weighted_abs_sum += abs_diff * n_cell
        total_shots += n_cell
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
        if abs_diff >= CELL_CHECK_DIFF_THRESHOLD:
            diffs.append(
                {
                    "ix": i,
                    "iy": j,
                    "center_x": cx,
                    "center_y": cy,
                    "center_distance_ft": dist_ft,
                    "n_shots": n_cell,
                    "mean_p_a_actual": round(p_act, 6),
                    "p_a_typical_mix_center": round(p_mix, 6),
                    "diff": round(signed_diff, 6),
                    "abs_diff": round(abs_diff, 6),
                }
            )
    diffs.sort(key=lambda item: (-item["abs_diff"], -item["n_shots"], item["ix"], item["iy"]))

    return {
        "note": (
            "Diagnostic only: compares all REGULAR shots in >=100-shot 2 ft cells against "
            "typical-mix p_A at the 2 ft cell centre, so it measures geometry inside the cell "
            "(real shots bunch closer to the rim than the cell centre) and putbacks, not just "
            "shot-type mix error."
        ),
        "min_cell_shots": CELL_CHECK_MIN_SHOTS,
        "n_cells_checked": len(cell_meta),
        "n_shots_covered": total_shots,
        "share_regular_shots_covered": round(total_shots / len(reg), 6),
        "weighted_mean_abs_diff": round(weighted_abs_sum / total_shots, 6),
        "max_abs_diff": round(max_abs_diff, 6),
        "diff_threshold": CELL_CHECK_DIFF_THRESHOLD,
        "n_cells_ge_threshold": len(diffs),
        "cells_ge_threshold": diffs,
    }


def _compute_modal_comparison(
    reg: pd.DataFrame,
    non_pb: pd.DataFrame,
    model: ExportedModel,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    vc = (
        non_pb.groupby(["SHOT_DISTANCE_FT", "ACTION_GROUP"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    totals = vc.sum(axis=1)
    modes = vc.idxmax(axis=1).astype(str).to_dict()
    modal_map = {
        str(int(d)): (modes[d] if int(totals[d]) >= MODAL_MIN_SHOTS else MODAL_FALLBACK_GROUP)
        for d in sorted(totals.index)
    }

    reg_default = reg["SHOT_DISTANCE_FT"].map(
        lambda d: modal_map.get(str(int(d)), MODAL_FALLBACK_GROUP)
    )
    non_pb_default = non_pb["SHOT_DISTANCE_FT"].map(
        lambda d: modal_map.get(str(int(d)), MODAL_FALLBACK_GROUP)
    )
    agree_all = float((reg_default == reg["ACTION_GROUP"].astype(str)).mean())
    agree_non_pb = float((non_pb_default == non_pb["ACTION_GROUP"].astype(str)).mean())

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(reg), size=MODAL_SAMPLE_SIZE, replace=False)
    sample = reg.iloc[idx][BASELINE_FEATURES].copy().reset_index(drop=True)
    sample_modal = sample.copy()
    sample_modal["ACTION_GROUP"] = sample_modal["SHOT_DISTANCE_FT"].map(
        lambda d: modal_map.get(str(int(d)), MODAL_FALLBACK_GROUP)
    )

    p_actual = predict_make_probability(model, sample, clutch_shift=False)
    p_modal = predict_make_probability(model, sample_modal, clutch_shift=False)
    abs_diff = np.abs(p_modal - p_actual)

    return {
        "description": (
            "Step 1.4 comparison using distance-modal ACTION_GROUP "
            "(min_shots=30, fallback='jump_shot') over REGULAR shots and a 20,000-shot "
            "random sample (seed=42); retained for comparison only."
        ),
        "min_shots_per_distance": MODAL_MIN_SHOTS,
        "fallback_action_group": MODAL_FALLBACK_GROUP,
        "modal_by_distance_ft": modal_map,
        "agreement_share_all_regular": round(agree_all, 6),
        "agreement_share_non_putback": round(agree_non_pb, 6),
        "sample_size": MODAL_SAMPLE_SIZE,
        "sample_seed": seed,
        "sample_mean_abs_p_a_diff": round(float(np.mean(abs_diff)), 6),
        "sample_max_abs_p_a_diff": round(float(np.max(abs_diff)), 6),
    }


def build_court_defaults(
    model: ExportedModel | None = None,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Build the court defaults dictionary from REGULAR shots and exported Model A."""
    if model is None:
        model = load_export(DEFAULT_EXPORT_DIR)

    categories = list(model.action_group_categories)
    reg = load_regular_shots()
    non_pb = reg[reg["IS_PUTBACK"] == 0].copy().reset_index(drop=True)
    n_reg = len(reg)
    n_non_pb = len(non_pb)
    n_pb = n_reg - n_non_pb

    mix_payload, per_foot_min, per_foot_max = _build_action_group_mix(non_pb, categories)

    x_edges = np.arange(GRID_X_MIN, GRID_X_MAX + 1.0, CELL_SIZE_TENTHS)
    y_edges = np.arange(GRID_Y_MIN, GRID_Y_MAX + 1.0, CELL_SIZE_TENTHS)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    ix_all, iy_all, in_box = assign_2ft_grid_indices(
        reg["LOC_X"].to_numpy(dtype=float),
        reg["LOC_Y"].to_numpy(dtype=float),
        x_edges,
        y_edges,
    )
    counts_2d = np.zeros((nx, ny), dtype=int)
    np.add.at(counts_2d, (ix_all[in_box], iy_all[in_box]), 1)
    mask_2d = counts_2d >= MIN_MASK_SHOTS
    mask_sym_and = mask_2d & mask_2d[::-1, :]

    n_in_box = int(in_box.sum())
    n_in_mask = int(counts_2d[mask_2d].sum())
    n_in_mask_sym_and = int(counts_2d[mask_sym_and].sum())

    primary_check = _compute_real_coords_cell_check(
        non_pb=non_pb,
        x_edges=x_edges,
        y_edges=y_edges,
        mix_by_dist=mix_payload["by_distance_ft"],
        pooled_weights=mix_payload["pooled_ge_22ft"]["weights_per_1000"],
        categories=categories,
        model=model,
    )
    centre_diag = _compute_cell_centre_diagnostic(
        reg=reg,
        ix_all=ix_all,
        iy_all=iy_all,
        in_box=in_box,
        x_edges=x_edges,
        y_edges=y_edges,
        counts_2d=counts_2d,
        mix_by_dist=mix_payload["by_distance_ft"],
        pooled_weights=mix_payload["pooled_ge_22ft"]["weights_per_1000"],
        categories=categories,
        model=model,
    )
    modal_check = _compute_modal_comparison(reg, non_pb, model, seed=seed)

    active_groups = [
        cat for cat in categories if int((non_pb["ACTION_GROUP"].astype(str) == cat).sum()) > 0
    ]

    return {
        "meta": {
            "schema_version": "1.0.0",
            "seed": seed,
            "source": "REGULAR shots in data/processed/all_shots.parquet (shot context only; no SHOT_MADE_FLAG)",
            "no_outcomes_or_ids": True,
            "n_regular_shots": n_reg,
            "n_regular_non_putback_shots": n_non_pb,
            "n_regular_putback_shots": n_pb,
            "action_group_categories": categories,
            "active_action_groups": active_groups,
            "overall_non_putback_shares": _shares_summary(non_pb, categories),
            "overall_regular_shares": _shares_summary(reg, categories),
            "mix_rule": {
                "description": (
                    "Share of each ACTION_GROUP among REGULAR, non-putback (IS_PUTBACK == 0) "
                    "shots by integer SHOT_DISTANCE_FT, stored as integer weights out of 1,000 "
                    "per group via largest-remainder rounding (exact sum 1,000). Uses the "
                    "per-foot distribution when distance has >= 100 such shots, otherwise the "
                    "pooled distribution of all REGULAR non-putback shots at >= 22 ft."
                ),
                "min_distance_shots_for_per_foot": MIN_PER_FOOT_SHOTS,
                "fallback_pool_min_distance_ft": POOLED_MIN_DISTANCE_FT,
                "weight_scale": WEIGHT_SCALE,
                "rounding": "largest_remainder",
                "per_foot_distance_min_ft": per_foot_min,
                "per_foot_distance_max_ft": per_foot_max,
                "max_stored_distance_ft": int(non_pb["SHOT_DISTANCE_FT"].max()),
            },
        },
        "action_group_mix": mix_payload,
        "density_mask": {
            "description": (
                "25 x 24 grid of 2 ft (20 tenths) cells over LOC_X in [-250, 250] and "
                "LOC_Y in [-50, 430]. Raw mask keeps cells with >= 10 REGULAR shots "
                "(counts only). Under left-right smoothing, a cell is shown only if both "
                "it and its mirror cell across LOC_X = 0 are in the raw mask. Coverage "
                "shares divide by all n_regular_shots (shots outside the box remain in the "
                "denominator)."
            ),
            "x_min_tenths": GRID_X_MIN,
            "x_max_tenths": GRID_X_MAX,
            "y_min_tenths": GRID_Y_MIN,
            "y_max_tenths": GRID_Y_MAX,
            "cell_size_tenths": CELL_SIZE_TENTHS,
            "nx": nx,
            "ny": ny,
            "x_edges_tenths": [float(v) for v in x_edges],
            "y_edges_tenths": [float(v) for v in y_edges],
            "min_shots_threshold": MIN_MASK_SHOTS,
            "n_cells_total": int(nx * ny),
            "n_cells_in_mask": int(mask_2d.sum()),
            "n_cells_in_mask_mirror_and": int(mask_sym_and.sum()),
            "share_denominator": "n_regular_shots",
            "n_regular_shots_total": n_reg,
            "n_regular_shots_in_box": n_in_box,
            "n_regular_shots_outside_box": n_reg - n_in_box,
            "n_regular_shots_in_mask": n_in_mask,
            "share_regular_shots_in_mask": round(n_in_mask / n_reg, 6),
            "n_regular_shots_in_mask_mirror_and": n_in_mask_sym_and,
            "share_regular_shots_in_mask_mirror_and": round(n_in_mask_sym_and / n_reg, 6),
            "counts": counts_2d.tolist(),
            "mask": mask_2d.tolist(),
        },
        "checks": {
            "cell_level_typical_mix_vs_actual": primary_check,
            "cell_centre_diagnostic": centre_diag,
            "modal_comparison": modal_check,
        },
    }


def main() -> None:
    payload = build_court_defaults()
    serialized = json.dumps(payload, separators=(",", ":"))
    size_bytes = len(serialized.encode("utf-8"))
    if size_bytes > MAX_FILE_BYTES:
        raise RuntimeError(
            f"Serialized court_defaults.json is {size_bytes:,} bytes (> {MAX_FILE_BYTES:,} bytes)."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(serialized + "\n", encoding="utf-8")

    dm = payload["density_mask"]
    pc = payload["checks"]["cell_level_typical_mix_vs_actual"]
    cd = payload["checks"]["cell_centre_diagnostic"]
    mc = payload["checks"]["modal_comparison"]
    print(f"Wrote {OUT_PATH} ({size_bytes:,} bytes, {size_bytes / 1024:.1f} KB)")
    print(
        f"Mask coverage (>= {dm['min_shots_threshold']} REGULAR shots): "
        f"{dm['n_regular_shots_in_mask']:,}/{payload['meta']['n_regular_shots']:,} = "
        f"{dm['share_regular_shots_in_mask']:.6f} "
        f"({dm['n_cells_in_mask']}/{dm['n_cells_total']} cells; "
        f"mirror-AND {dm['share_regular_shots_in_mask_mirror_and']:.6f})"
    )
    print(
        f"Primary check (real-coords non-putback, >= {pc['min_cell_shots']} shots, "
        f"{pc['n_cells_checked']} cells, {pc['n_shots_covered']:,} shots): "
        f"weighted mean |diff|={pc['weighted_mean_abs_diff']:.6f}, "
        f"max |diff|={pc['max_abs_diff']:.6f}, cells >= {pc['diff_threshold']}: {pc['n_cells_ge_threshold']}"
    )
    print(
        f"Cell-centre diagnostic ({cd['n_cells_checked']} cells): "
        f"weighted mean |diff|={cd['weighted_mean_abs_diff']:.6f}, "
        f"max |diff|={cd['max_abs_diff']:.6f}, cells >= {cd['diff_threshold']}: {cd['n_cells_ge_threshold']}"
    )
    print(
        f"Modal comparison: agree_all={mc['agreement_share_all_regular']:.6f}, "
        f"sample mean |diff|={mc['sample_mean_abs_p_a_diff']:.6f}, "
        f"max |diff|={mc['sample_max_abs_p_a_diff']:.6f}"
    )


if __name__ == "__main__":
    main()
