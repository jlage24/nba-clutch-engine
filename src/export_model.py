"""Phase 4a — Step 2: Export portable simulator model artifacts.

Builds a self-contained export under ``export/`` from Phase 3b data/results
(without loading any Phase 3 clutch-only model from ``models/``):

- ``export/model_a.json``: Final Model A (XGBoost JSON) fitted on all 196,198
  REGULAR shots with the saved Phase 3b hyperparameters and median tree count
  (n_estimators=168).
- ``export/player_offsets.csv``: Final DerSimonian-Laird shrunk player offsets
  estimated from all 196,198 REGULAR out-of-fold residuals. Players with no
  regular-play shots (10 of 582 players) are absent; the loader returns 0.0.
- ``export/clutch_shift.json``: Constant logit-scale clutch shifts for Model A
  and Model B over the 7,104 CLUTCH shots (with 95% game-level cluster
  bootstrap intervals), plus exploratory 2PT and 3PT shifts.
- ``export/manifest.json``: Feature order, ACTION_GROUP categories, inference
  rules, model/data metadata, versions, and limitations.

Run from the repo root::

    python src/export_model.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_model import (  # noqa: E402
    BASELINE_FEATURES,
    RNG_SEED,
    TARGET,
    _X,
    assert_no_clutch_in_training,
)
from src.features import ACTION_CATEGORIES, ACTION_DTYPE  # noqa: E402
from src.predictive_comparison import fit_logit_shift  # noqa: E402
from src.shooter_offsets import (  # noqa: E402
    P_CLIP_HIGH,
    P_CLIP_LOW,
    compute_player_offsets,
)

DATA_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"
EXPORT_DIR = ROOT / "export"

N_BOOT = 2_000
LOGIT_CLIP_EPS = 1e-6


def load_phase3b_hyperparameters(
    quality_path: Path = DOCS_DIR / "baseline_quality.json",
) -> tuple[dict[str, Any], int, list[int]]:
    """Load Phase 3b Model A hyperparameters and median tree count."""
    with open(quality_path) as f:
        quality = json.load(f)
    best_cfg = dict(quality["best_config"])
    trees_per_fold = [int(t) for t in quality["trees_per_fold"]]
    n_estimators = int(np.median(trees_per_fold))
    return best_cfg, n_estimators, trees_per_fold


def fit_final_model_a(
    regular_df: pd.DataFrame,
    best_cfg: dict[str, Any],
    n_estimators: int,
    seed: int = RNG_SEED,
) -> xgb.XGBClassifier:
    """Fit one final Model A on all REGULAR shots with Phase 3b hyperparameters."""
    assert_no_clutch_in_training(regular_df, fold_k=-1, phase="final-export")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        enable_categorical=True,
        tree_method="hist",
        random_state=seed,
        eval_metric="logloss",
        n_jobs=6,
        **best_cfg,
    )
    model.fit(_X(regular_df), regular_df[TARGET])
    return model


def verify_double_fit_determinism(
    all_shots: pd.DataFrame,
    regular_df: pd.DataFrame,
    best_cfg: dict[str, Any],
    n_estimators: int,
    first_preds: np.ndarray,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Refit Model A with the same seed and verify identical predictions on all shots."""
    second_model = fit_final_model_a(regular_df, best_cfg, n_estimators, seed=seed)
    second_preds = second_model.predict_proba(_X(all_shots))[:, 1]
    max_abs_diff = float(np.max(np.abs(first_preds - second_preds)))
    exact_match = bool(np.array_equal(first_preds, second_preds))
    return {
        "n_shots_checked": int(len(all_shots)),
        "exact_match": exact_match,
        "max_abs_diff": max_abs_diff,
    }


def build_player_offsets_table(
    oof_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate shrunk per-player offsets from all REGULAR OOF residuals."""
    reg_oof = oof_df[oof_df["SET_LABEL"] == "REGULAR"].copy()
    result = compute_player_offsets(reg_oof)
    if not isinstance(result, tuple):
        raise RuntimeError("Unexpected return from compute_player_offsets; too few players.")
    offsets, tau2, rbar, stats_v = result

    player_meta = (
        reg_oof.groupby("PLAYER_ID", as_index=False)
        .agg(
            PLAYER_NAME=("PLAYER_NAME", "first"),
            n_regular_shots=(TARGET, "count"),
        )
        .sort_values("PLAYER_ID")
        .reset_index(drop=True)
    )
    player_meta["offset"] = player_meta["PLAYER_ID"].astype(str).map(offsets).astype(float)

    shrinkage = tau2 / (tau2 + stats_v["s2_p"].values)
    n_total_players = int(oof_df["PLAYER_ID"].nunique())
    n_table_players = int(len(player_meta))
    n_absent = n_total_players - n_table_players

    summary = {
        "estimator": "DerSimonian-Laird empirical-Bayes shrinkage on REGULAR OOF residuals (Model A)",
        "n_regular_shots": int(len(reg_oof)),
        "n_players_in_table": n_table_players,
        "n_players_total_dataset": n_total_players,
        "n_players_absent_zero_regular_shots": int(n_absent),
        "absent_player_rule": (
            f"The {n_absent} players with no regular-play shots (plus any unknown player) "
            "are absent from player_offsets.csv and receive offset 0.0."
        ),
        "tau2": float(tau2),
        "tau_pp": float(np.sqrt(tau2) * 100.0),
        "rbar": float(rbar),
        "avg_shrinkage_factor": float(np.mean(shrinkage)),
    }
    return player_meta[["PLAYER_ID", "PLAYER_NAME", "n_regular_shots", "offset"]], summary


def bootstrap_logit_shift(
    df: pd.DataFrame,
    prob_col: str,
    n_resamples: int = N_BOOT,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Estimate logit-scale clutch shift with 95% game-level cluster bootstrap CI."""
    game_ids = df["GAME_ID"].astype(str).values
    y = df[TARGET].astype(float).values
    p = df[prob_col].astype(float).values

    point_shift = fit_logit_shift(p, y)

    unique_games = np.sort(np.unique(game_ids))
    n_games = len(unique_games)
    game_groups = (
        pd.DataFrame({"game": game_ids, "idx": np.arange(len(game_ids))})
        .groupby("game")["idx"]
        .apply(np.array)
        .to_dict()
    )

    rng = np.random.default_rng(seed)
    boot_shifts = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        idx = np.concatenate([game_groups[g] for g in sampled_games])
        boot_shifts[b] = fit_logit_shift(p[idx], y[idx])

    ci_lower = float(np.percentile(boot_shifts, 2.5))
    ci_upper = float(np.percentile(boot_shifts, 97.5))

    return {
        "shift_logit": round(float(point_shift), 6),
        "ci_lower_logit": round(ci_lower, 6),
        "ci_upper_logit": round(ci_upper, 6),
        "std_err_logit": round(float(np.std(boot_shifts, ddof=1)), 6),
        "n_shots": int(len(df)),
        "n_games": int(n_games),
    }


def build_clutch_shift_payload(oof_df: pd.DataFrame) -> dict[str, Any]:
    """Estimate overall and exploratory 2PT/3PT clutch shifts for Model A and Model B."""
    clutch = oof_df[oof_df["SET_LABEL"] == "CLUTCH"].copy()
    clutch_2pt = clutch[clutch["SHOT_VALUE"] == 2].copy()
    clutch_3pt = clutch[clutch["SHOT_VALUE"] == 3].copy()

    overall_a = bootstrap_logit_shift(clutch, "p_A")
    overall_b = bootstrap_logit_shift(clutch, "p_B")

    exp_a_2pt = bootstrap_logit_shift(clutch_2pt, "p_A")
    exp_a_3pt = bootstrap_logit_shift(clutch_3pt, "p_A")
    exp_b_2pt = bootstrap_logit_shift(clutch_2pt, "p_B")
    exp_b_3pt = bootstrap_logit_shift(clutch_3pt, "p_B")

    with open(DOCS_DIR / "predictive_comparison.json") as f:
        pred_comp = json.load(f)
    fold_deltas = [float(row["delta_logit"]) for row in pred_comp["model_b_shift_folds"]]
    fold_min = min(fold_deltas)
    fold_max = max(fold_deltas)
    fold_mean = float(np.mean(fold_deltas))

    return {
        "meta": {
            "description": (
                "Descriptive constant logit-scale clutch shift: intercept of logistic "
                "regression of SHOT_MADE_FLAG on offset logit(clip(p_base, 1e-6, 1 - 1e-6)) "
                "over all 7,104 CLUTCH shots using out-of-fold baseline probabilities. "
                "This is a descriptive gap bundling defense, fatigue, shot selection "
                "within shot type, and shooter composition — not a causal pressure effect."
            ),
            "app_uses_single_shift_in_v1": True,
            "n_bootstrap": N_BOOT,
            "bootstrap_method": "game-level cluster bootstrap (resample GAME_IDs), 95% percentile CI",
            "rng_seed": RNG_SEED,
            "logit_clip_eps": LOGIT_CLIP_EPS,
        },
        "model_a": overall_a,
        "model_b": overall_b,
        "phase3b_fold_comparison_model_b": {
            "source": "docs/predictive_comparison.json (model_b_shift_folds)",
            "per_fold_deltas_logit": fold_deltas,
            "fold_min_logit": fold_min,
            "fold_max_logit": fold_max,
            "fold_mean_logit": round(fold_mean, 6),
            "within_fold_range": bool(fold_min <= overall_b["shift_logit"] <= fold_max),
        },
        "exploratory": {
            "warning": (
                "Exploratory breakdown by SHOT_VALUE; the simulator uses one overall "
                "shift in v1. If the 2PT or 3PT interval includes the overall shift, "
                "the data does not support separate shifts."
            ),
            "model_a_2pt": exp_a_2pt,
            "model_a_3pt": exp_a_3pt,
            "model_b_2pt": exp_b_2pt,
            "model_b_3pt": exp_b_3pt,
        },
    }


def build_manifest(
    all_shots: pd.DataFrame,
    best_cfg: dict[str, Any],
    n_estimators: int,
    trees_per_fold: list[int],
    det_check: dict[str, Any],
    offsets_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble export/manifest.json with full schema, rules, metadata, and limits."""
    return {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_description": (
            "Descriptive shot-make probability model for NBA simulator (Phase 4a). "
            "Baseline Model A fitted on regular-play shots only; Model B adds shrunk "
            "per-player regular-play offsets; optional clutch shift adds a descriptive "
            "constant logit-scale offset estimated from clutch out-of-fold residuals."
        ),
        "features": list(BASELINE_FEATURES),
        "numeric_features": [f for f in BASELINE_FEATURES if f != "ACTION_GROUP"],
        "categorical_features": ["ACTION_GROUP"],
        "action_group_categories": list(ACTION_CATEGORIES),
        "feature_rules": {
            "loc_units": "LOC_X and LOC_Y are in tenths of a foot with the hoop at (0, 0).",
            "shot_distance_ft": "Computed as hypot(LOC_X, LOC_Y) / 10.0 when not explicitly supplied.",
            "shot_angle": "Computed as degrees(arctan2(LOC_X, LOC_Y)) when not explicitly supplied.",
            "is_putback": (
                "1 if ACTION_TYPE contains 'tip' or 'putback' (case-insensitive), else 0, "
                "unless IS_PUTBACK is provided explicitly."
            ),
            "distance_clamp_rule": (
                "If IS_PUTBACK == 0 and SHOT_DISTANCE_FT < 1.0, clamp SHOT_DISTANCE_FT to 1.0 "
                "(matches src/predict.py; in training data, exact (0, 0) occurs almost only on "
                "tips and putbacks)."
            ),
            "unknown_action_group_policy": "Raise ValueError if any ACTION_GROUP value is not in action_group_categories.",
        },
        "prediction_formula": {
            "model_a": "p_A = Model A XGBoost predict_proba(X)[:, 1]",
            "model_b": "p_B = clip(p_A + b_player, 0.01, 0.99); unknown/absent player uses b_player = 0 so p_B = p_A",
            "final": "p_final = sigmoid(logit(clip(p_base, 1e-6, 1 - 1e-6)) + shift), with shift = 0 when clutch_shift=False",
            "p_b_clip_low": P_CLIP_LOW,
            "p_b_clip_high": P_CLIP_HIGH,
            "logit_clip_eps": LOGIT_CLIP_EPS,
        },
        "model_a_training": {
            "source_script": "src/export_model.py",
            "hyperparameters": best_cfg,
            "n_estimators": int(n_estimators),
            "trees_per_fold_phase3b": trees_per_fold,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "logloss",
            "random_seed": RNG_SEED,
            "n_regular_training_shots": int(all_shots["REGULAR"].sum()),
            "double_fit_determinism": det_check,
        },
        "player_offsets": offsets_summary,
        "data_facts": {
            "season": "2025-26 regular season",
            "total_shots": int(len(all_shots)),
            "regular_shots": int(all_shots["REGULAR"].sum()),
            "late_blowout_shots": int(all_shots["LATE_BLOWOUT"].sum()),
            "clutch_shots": int(all_shots["CLUTCH"].sum()),
            "n_games": int(all_shots["GAME_ID"].nunique()),
            "n_clutch_games": int(all_shots.loc[all_shots["CLUTCH"], "GAME_ID"].nunique()),
            "n_folds": 5,
        },
        "rng_seed": RNG_SEED,
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Single season only (2025-26 NBA regular season).",
            "No defender tracking, closest-defender distance, or shot-clock data.",
            (
                "The clutch shift is a descriptive gap, not a causal pressure effect; "
                "it bundles defensive intensity, fatigue, shot selection within shot type, "
                "and shooter composition."
            ),
            (
                "The clutch shift was estimated using out-of-fold p_A/p_B from models "
                "trained on 80% of games, whereas the exported Model A is fitted on 100% "
                "of regular-play shots (a second-order difference)."
            ),
            (
                "Player offsets in the export are estimated from all regular-play shots "
                "using out-of-fold residuals, so players with few shots are shrunk more "
                "toward the grand mean, and the 10 players with zero regular-play shots "
                "are absent from player_offsets.csv and receive offset 0.0."
            ),
        ],
    }


def main() -> None:
    print("=" * 64)
    print("Phase 4a — Step 2: Exporting portable simulator model artifacts")
    print("=" * 64)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading Phase 3b dataset and hyperparameters …")
    all_shots = pd.read_parquet(DATA_DIR / "all_shots.parquet")
    all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(str)
    all_shots["ACTION_GROUP"] = pd.Categorical(all_shots["ACTION_GROUP"], dtype=ACTION_DTYPE)
    regular = all_shots[all_shots["REGULAR"]].copy()

    oof = pd.read_parquet(DATA_DIR / "oof_predictions.parquet")
    oof["GAME_ID"] = oof["GAME_ID"].astype(str)

    best_cfg, n_estimators, trees_per_fold = load_phase3b_hyperparameters()
    print(f"    REGULAR shots : {len(regular):,} | All shots : {len(all_shots):,}")
    print(f"    Hyperparams   : {best_cfg}")
    print(f"    n_estimators  : {n_estimators} (median of {trees_per_fold})")

    print("\n[2] Fitting final Model A on all REGULAR shots …")
    model_a = fit_final_model_a(regular, best_cfg, n_estimators, seed=RNG_SEED)
    feature_names = model_a.get_booster().feature_names
    if feature_names != BASELINE_FEATURES:
        raise RuntimeError(
            f"Feature name mismatch: {feature_names} != {BASELINE_FEATURES}"
        )

    model_path = EXPORT_DIR / "model_a.json"
    model_a.save_model(str(model_path))
    print(f"    Saved -> {model_path.relative_to(ROOT)} ({model_path.stat().st_size:,} bytes)")

    first_preds = model_a.predict_proba(_X(all_shots))[:, 1]
    det_check = verify_double_fit_determinism(
        all_shots, regular, best_cfg, n_estimators, first_preds, seed=RNG_SEED
    )
    print(
        f"    Double-fit determinism on {det_check['n_shots_checked']:,} shots: "
        f"exact_match={det_check['exact_match']}, max_abs_diff={det_check['max_abs_diff']:.3e}"
    )

    print("\n[3] Computing final player offsets from all REGULAR OOF residuals …")
    offsets_df, offsets_summary = build_player_offsets_table(oof)
    offsets_path = EXPORT_DIR / "player_offsets.csv"
    offsets_df.to_csv(offsets_path, index=False)
    print(
        f"    Players in table: {offsets_summary['n_players_in_table']:,} "
        f"(absent with 0 REGULAR shots: {offsets_summary['n_players_absent_zero_regular_shots']})"
    )
    print(f"    Saved -> {offsets_path.relative_to(ROOT)} ({offsets_path.stat().st_size:,} bytes)")

    print("\n[4] Estimating clutch shifts (overall + exploratory 2PT/3PT) …")
    shift_payload = build_clutch_shift_payload(oof)
    shift_path = EXPORT_DIR / "clutch_shift.json"
    with open(shift_path, "w") as f:
        json.dump(shift_payload, f, indent=2)
    sa = shift_payload["model_a"]
    sb = shift_payload["model_b"]
    comp = shift_payload["phase3b_fold_comparison_model_b"]
    print(
        f"    Model A shift: {sa['shift_logit']:+.6f} "
        f"95% CI [{sa['ci_lower_logit']:+.6f}, {sa['ci_upper_logit']:+.6f}]"
    )
    print(
        f"    Model B shift: {sb['shift_logit']:+.6f} "
        f"95% CI [{sb['ci_lower_logit']:+.6f}, {sb['ci_upper_logit']:+.6f}] "
        f"(Phase 3b folds: [{comp['fold_min_logit']:+.5f}, {comp['fold_max_logit']:+.5f}], "
        f"within_range={comp['within_fold_range']})"
    )
    print(f"    Saved -> {shift_path.relative_to(ROOT)} ({shift_path.stat().st_size:,} bytes)")

    print("\n[5] Writing manifest.json …")
    manifest = build_manifest(
        all_shots, best_cfg, n_estimators, trees_per_fold, det_check, offsets_summary
    )
    manifest_path = EXPORT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    Saved -> {manifest_path.relative_to(ROOT)} ({manifest_path.stat().st_size:,} bytes)")

    total_bytes = sum(p.stat().st_size for p in EXPORT_DIR.iterdir() if p.is_file())
    print(f"\nTotal export/ size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    if total_bytes > 5 * 1024 * 1024:
        raise RuntimeError(f"Export exceeds 5 MB limit ({total_bytes} bytes).")
    print("✓ Step 2 export complete.")


if __name__ == "__main__":
    main()
