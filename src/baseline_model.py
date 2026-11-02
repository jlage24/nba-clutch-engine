"""Cross-fitted baseline model for Phase 3b (Model A).

Trains XGBoost on REGULAR shots only, using 5-fold GroupKFold on GAME_ID.
Hyperparameters are tuned from scratch on REGULAR shots (independent of any
Phase 3 artifacts); the brief §3.1 forbids reusing Phase 3 models or
hyperparameters because those were selected on clutch outcomes.

Outputs
-------
data/processed/fold_assignment.parquet
    One row per game: GAME_ID, fold (0–4).
data/processed/oof_predictions.parquet
    One row per shot: GAME_ID, GAME_EVENT_ID, PLAYER_ID, PLAYER_NAME,
    SET_LABEL, SHOT_VALUE, PERIOD, fold, SHOT_MADE_FLAG, p_A.
docs/baseline_tuning.json
    All tuning configurations with their grouped-CV log loss.
docs/baseline_quality.json
    Quality metrics on REGULAR out-of-fold predictions.

Run from the repo root::

    python src/baseline_model.py

Hard rules enforced here (Phase 3b brief §3)
--------------------------------------------
- No CLUTCH or LATE_BLOWOUT shot ever enters any training or early-stopping
  split of Model A.  ``assert_no_clutch_in_training`` is called before every
  fold fit and raises ValueError otherwise.
- Features used: LOC_X, LOC_Y, SHOT_DISTANCE_FT, SHOT_ANGLE, SHOT_VALUE,
  ACTION_GROUP, IS_PUTBACK  (shot context only, no game-state).
- NaN SCORE_MARGIN on the 1 unmatched REGULAR shot is a metadata column only;
  Model A features do not include SCORE_MARGIN so no NaN reaches XGBoost.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics, expected_calibration_error  # noqa: E402
from src.features import ACTION_DTYPE  # noqa: E402

DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# Baseline feature set — shot context only, NO game-state (brief §5)
BASELINE_FEATURES = [
    "LOC_X", "LOC_Y", "SHOT_DISTANCE_FT", "SHOT_ANGLE",
    "SHOT_VALUE", "ACTION_GROUP", "IS_PUTBACK",
]
TARGET = "SHOT_MADE_FLAG"
N_FOLDS = 5
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Safety guard: no CLUTCH or LATE_BLOWOUT in training data
# ---------------------------------------------------------------------------

def assert_no_clutch_in_training(
    train_df: pd.DataFrame,
    fold_k: int,
    phase: str = "outer-train",
) -> None:
    """Raise ValueError if any CLUTCH or LATE_BLOWOUT shot is in train_df.

    Must be called before every model fit.  The brief (§3.1) forbids clutch
    outcomes from influencing the baseline model in any way.

    Parameters
    ----------
    train_df:
        The DataFrame being passed to a model fit call.
    fold_k:
        Current fold index (for the error message).
    phase:
        "outer-train" or "inner-train" (for the error message).
    """
    if "CLUTCH" not in train_df.columns and "LATE_BLOWOUT" not in train_df.columns:
        # No set-label columns present — skip (e.g. in unit tests with minimal data)
        return
    clutch_count = int(train_df.get("CLUTCH", pd.Series(False)).sum())
    blowout_count = int(train_df.get("LATE_BLOWOUT", pd.Series(False)).sum())
    if clutch_count > 0 or blowout_count > 0:
        raise ValueError(
            f"STOP [fold {fold_k}, {phase}]: training data contains "
            f"{clutch_count} CLUTCH and {blowout_count} LATE_BLOWOUT shots. "
            "These must never influence Model A."
        )


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------

def make_fold_assignment(all_shots: pd.DataFrame) -> pd.DataFrame:
    """Create a 5-fold GroupKFold assignment over all 1,230 games.

    Returns a DataFrame with columns [GAME_ID, fold] where fold in 0..4.
    The assignment is over ALL games (not just REGULAR), so that the same
    fold boundary applies consistently to REGULAR, CLUTCH and LATE_BLOWOUT
    when producing out-of-fold predictions.
    """
    games = all_shots[["GAME_ID"]].drop_duplicates().copy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    # GKF assigns each group to exactly one validation fold
    fold_label = np.empty(len(games), dtype=int)
    # Use a dummy X and y (values don't matter for GroupKFold)
    dummy_X = np.zeros((len(games), 1))
    dummy_y = np.zeros(len(games))
    for k, (_, val_idx) in enumerate(gkf.split(dummy_X, dummy_y, groups=games["GAME_ID"])):
        fold_label[val_idx] = k
    games["fold"] = fold_label
    return games.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def build_tuning_grid() -> list[dict[str, Any]]:
    """Return ≤30 hyperparameter configurations to search.

    The search space is wider than Phase 3 (depth up to 8, as recommended by
    the brief for ~196k shots) and is defined independently — it must not
    replicate Phase 3 hyperparameters, which were selected on clutch outcomes.
    """
    grid: dict[str, list] = {
        "max_depth":       [4, 6, 8],
        "learning_rate":   [0.03, 0.06],
        "min_child_weight":[20, 50],
        "subsample":       [0.8],
        "colsample_bytree":[0.8],
        "reg_lambda":      [1.0, 3.0],
    }
    keys, values = zip(*grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return configs


def tune_hyperparameters(
    regular_df: pd.DataFrame,
    fold_assign: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grid-search XGBoost hyperparameters on REGULAR shots with grouped CV.

    For each configuration, runs 5-fold grouped CV (same folds as the outer
    loop) on REGULAR shots only, using early stopping on an inner grouped
    split of the training fold (no data leaks to the outer OOF set).

    Parameters
    ----------
    regular_df:
        REGULAR shots only (no CLUTCH, no LATE_BLOWOUT).
    fold_assign:
        DataFrame[GAME_ID, fold] for all games.

    Returns
    -------
    best_cfg:
        The configuration with the lowest mean grouped-CV log loss.
    all_results:
        List of dicts with config + cv_log_loss + cv_brier, sorted ascending
        by cv_log_loss.
    """
    # Map each REGULAR shot to its fold
    reg = regular_df.merge(fold_assign, on="GAME_ID", how="left")
    y = reg[TARGET].values
    configs = build_tuning_grid()
    print(f"    Tuning over {len(configs)} configurations …")

    all_results: list[dict[str, Any]] = []

    for cfg in configs:
        oof = np.full(len(reg), np.nan)

        for k in range(N_FOLDS):
            tr_mask = reg["fold"] != k
            val_mask = reg["fold"] == k
            tr = reg[tr_mask].copy()
            val = reg[val_mask].copy()

            # Safety: must be REGULAR only
            assert_no_clutch_in_training(tr, fold_k=k, phase="tuning-outer-train")

            # Inner split for early stopping (80/20 games within training fold)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG_SEED + k)
            inner_tr_idx, inner_val_idx = next(gss.split(tr, groups=tr["GAME_ID"]))
            inner_tr = tr.iloc[inner_tr_idx]
            inner_val = tr.iloc[inner_val_idx]

            assert_no_clutch_in_training(inner_tr, fold_k=k, phase="tuning-inner-train")

            m = xgb.XGBClassifier(
                n_estimators=400,
                enable_categorical=True,
                tree_method="hist",
                random_state=RNG_SEED,
                eval_metric="logloss",
                early_stopping_rounds=20,
                n_jobs=6,
                **cfg,
            )
            m.fit(
                _X(inner_tr), inner_tr[TARGET],
                eval_set=[(_X(inner_val), inner_val[TARGET])],
                verbose=False,
            )
            n_trees = m.best_iteration + 1

            # Refit on full outer fold with n_trees
            m2 = xgb.XGBClassifier(
                n_estimators=n_trees,
                enable_categorical=True,
                tree_method="hist",
                random_state=RNG_SEED,
                eval_metric="logloss",
                n_jobs=6,
                **cfg,
            )
            assert_no_clutch_in_training(tr, fold_k=k, phase="tuning-refit")
            m2.fit(_X(tr), tr[TARGET])
            oof[val_mask.values] = m2.predict_proba(_X(val))[:, 1]

        valid = ~np.isnan(oof)
        cv_ll = float(log_loss(y[valid], np.clip(oof[valid], 1e-15, 1 - 1e-15)))
        cv_brier = float(brier_score_loss(y[valid], oof[valid]))
        all_results.append({"config": cfg, "cv_log_loss": cv_ll, "cv_brier": cv_brier})

    all_results.sort(key=lambda x: x["cv_log_loss"])
    best_cfg = all_results[0]["config"]
    return best_cfg, all_results


# ---------------------------------------------------------------------------
# OOF prediction loop
# ---------------------------------------------------------------------------

def _X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the baseline feature columns, ensuring ACTION_GROUP is categorical."""
    out = df[BASELINE_FEATURES].copy()
    out["ACTION_GROUP"] = pd.Categorical(out["ACTION_GROUP"], dtype=ACTION_DTYPE)
    return out


def run_oof_predictions(
    all_shots: pd.DataFrame,
    fold_assign: pd.DataFrame,
    best_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Produce out-of-fold p_A predictions for ALL shots using Model A.

    For each fold k:
    - Fit Model A on REGULAR shots from games **outside** fold k,
      with early stopping on an inner 80/20 grouped split of those REGULAR shots.
    - Predict p_A for ALL shots in games of fold k
      (REGULAR, CLUTCH and LATE_BLOWOUT alike).

    No CLUTCH or LATE_BLOWOUT shot is ever in a training or early-stopping set.
    """
    shots = all_shots.merge(fold_assign, on="GAME_ID", how="left")

    p_A = np.full(len(shots), np.nan)
    fold_out = np.full(len(shots), -1, dtype=int)
    trees_per_fold: list[int] = []

    for k in range(N_FOLDS):
        tr_mask = (shots["fold"] != k) & shots["REGULAR"]
        val_mask = shots["fold"] == k

        tr = shots[tr_mask].copy()
        val = shots[val_mask].copy()

        # Hard rule: training set must be REGULAR only
        assert_no_clutch_in_training(tr, fold_k=k, phase="oof-train")

        # Inner split for early stopping
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG_SEED + k)
        inner_tr_idx, inner_val_idx = next(gss.split(tr, groups=tr["GAME_ID"]))
        inner_tr = tr.iloc[inner_tr_idx]
        inner_val = tr.iloc[inner_val_idx]

        assert_no_clutch_in_training(inner_tr, fold_k=k, phase="oof-inner-train")

        m_inner = xgb.XGBClassifier(
            n_estimators=600,
            enable_categorical=True,
            tree_method="hist",
            random_state=RNG_SEED,
            eval_metric="logloss",
            early_stopping_rounds=25,
            n_jobs=6,
            **best_cfg,
        )
        m_inner.fit(
            _X(inner_tr), inner_tr[TARGET],
            eval_set=[(_X(inner_val), inner_val[TARGET])],
            verbose=False,
        )
        n_trees = m_inner.best_iteration + 1
        trees_per_fold.append(n_trees)
        print(f"      fold {k}: {len(tr):,} REGULAR training shots | {n_trees} trees (early stopping)")

        # Refit on full outer training fold
        m_fold = xgb.XGBClassifier(
            n_estimators=n_trees,
            enable_categorical=True,
            tree_method="hist",
            random_state=RNG_SEED,
            eval_metric="logloss",
            n_jobs=6,
            **best_cfg,
        )
        assert_no_clutch_in_training(tr, fold_k=k, phase="oof-refit")
        m_fold.fit(_X(tr), tr[TARGET])

        val_idx_pos = np.where(val_mask.values)[0]
        p_A[val_idx_pos] = m_fold.predict_proba(_X(val))[:, 1]
        fold_out[val_idx_pos] = k

    shots = shots.copy()
    shots["p_A"] = p_A
    shots["fold"] = fold_out

    print(f"    Trees per fold: {trees_per_fold} | median: {int(np.median(trees_per_fold))}")

    # Build SET_LABEL column
    def _label(row: pd.Series) -> str:
        if row["CLUTCH"]:
            return "CLUTCH"
        if row["LATE_BLOWOUT"]:
            return "LATE_BLOWOUT"
        return "REGULAR"

    shots["SET_LABEL"] = shots.apply(_label, axis=1)

    return shots, trees_per_fold


# ---------------------------------------------------------------------------
# Baseline quality metrics
# ---------------------------------------------------------------------------

def compute_baseline_quality(oof_df: pd.DataFrame) -> dict[str, Any]:
    """Compute quality metrics on REGULAR out-of-fold predictions.

    Checks:
    - Brier, log loss, ROC AUC, ECE vs constant baseline
    - Mean residual overall, by SHOT_VALUE, by ACTION_GROUP, by distance bucket
    Expected: residuals near zero (within ~0.5 pp) in each subgroup.
    AUC should be in 0.65–0.70; above 0.75 indicates leakage.
    """
    reg = oof_df[oof_df["SET_LABEL"] == "REGULAR"].copy()
    y = reg[TARGET].values
    p = reg["p_A"].values
    game_ids = reg["GAME_ID"].values

    constant_p = float(y.mean())
    p_const = np.full_like(y, constant_p, dtype=float)

    model_metrics = compute_metrics(y, p)
    const_metrics = compute_metrics(y, p_const)

    # Residuals by subgroup
    reg["residual"] = y - p

    def _residual_stats(mask: pd.Series, label: str) -> dict:
        sub = reg[mask]
        r = sub["residual"].values
        return {
            "label": label,
            "n": int(len(sub)),
            "mean_residual_pp": round(float(r.mean()) * 100, 3),
            "std_residual_pp": round(float(r.std()) * 100, 3),
        }

    # By SHOT_VALUE
    by_value = [
        _residual_stats(reg["SHOT_VALUE"] == v, f"SHOT_VALUE={v}")
        for v in [2, 3]
    ]

    # By ACTION_GROUP
    by_action = [
        _residual_stats(reg["ACTION_GROUP"] == ag, f"ACTION_GROUP={ag}")
        for ag in reg["ACTION_GROUP"].cat.categories
    ]

    # By distance bucket (0-5, 5-15, 15-25, 25+ feet)
    bins = [0, 5, 15, 25, 100]
    labels_d = ["0-5ft", "5-15ft", "15-25ft", "25+ft"]
    reg["dist_bucket"] = pd.cut(reg["SHOT_DISTANCE_FT"], bins=bins, labels=labels_d, right=False)
    by_distance = [
        _residual_stats(reg["dist_bucket"] == b, f"distance={b}")
        for b in labels_d
    ]

    # By period (Q1/Q2/Q3 for REGULAR — Q4 is never REGULAR by definition)
    by_period = [
        _residual_stats(reg["PERIOD"] == p_val, f"PERIOD={p_val}")
        for p_val in sorted(reg["PERIOD"].unique())
    ]

    return {
        "n_regular_oof": int(len(reg)),
        "model_A_metrics": model_metrics,
        "constant_baseline_metrics": const_metrics,
        "auc_leakage_check": {
            "auc": model_metrics["roc_auc"],
            "threshold": 0.75,
            "ok": model_metrics["roc_auc"] < 0.75,
        },
        "residuals_by_shot_value": by_value,
        "residuals_by_action_group": by_action,
        "residuals_by_distance": by_distance,
        "residuals_by_period": by_period,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 60)
    print("Phase 3b — Step 2: baseline_model.py (Model A)")
    print("=" * 60)

    # Load all_shots
    print("\n[1] Loading data/processed/all_shots.parquet …")
    all_shots = pd.read_parquet(DATA / "all_shots.parquet")
    all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(str)
    # Ensure ACTION_GROUP is categorical with fixed dtype
    all_shots["ACTION_GROUP"] = pd.Categorical(all_shots["ACTION_GROUP"], dtype=ACTION_DTYPE)
    regular = all_shots[all_shots["REGULAR"]].copy()
    print(f"    Total shots   : {len(all_shots):,}")
    print(f"    REGULAR shots : {len(regular):,}")
    print(f"    CLUTCH shots  : {all_shots['CLUTCH'].sum():,}")
    print(f"    LATE_BLOWOUT  : {all_shots['LATE_BLOWOUT'].sum():,}")

    # Verify NaN margin does not touch features (brief safety requirement)
    nan_margin_rows = all_shots[all_shots["SCORE_MARGIN"].isna()]
    if not nan_margin_rows.empty:
        nan_feats = nan_margin_rows[BASELINE_FEATURES].isna().any().any()
        if nan_feats:
            raise ValueError("STOP: NaN SCORE_MARGIN leaked into baseline features!")
        print(f"    NaN SCORE_MARGIN rows: {len(nan_margin_rows)} — none in baseline features ✓")

    # Step 2.1 — fold assignment
    print("\n[2] Creating 5-fold GroupKFold assignment over all games …")
    fold_assign = make_fold_assignment(all_shots)
    fold_assign.to_parquet(DATA / "fold_assignment.parquet", index=False)
    print(f"    Games assigned: {len(fold_assign):,}")
    print("    Fold sizes:")
    print("   ", fold_assign["fold"].value_counts().sort_index().to_dict())
    print(f"    Saved → data/processed/fold_assignment.parquet")

    # Step 2.2 — hyperparameter tuning on REGULAR shots
    print("\n[3] Tuning hyperparameters on REGULAR shots only …")
    best_cfg, tuning_results = tune_hyperparameters(regular, fold_assign)
    print(f"    Best config : {best_cfg}")
    print(f"    Best CV log loss: {tuning_results[0]['cv_log_loss']:.6f}")
    print("    Top 5 configs:")
    for r in tuning_results[:5]:
        print(f"      cv_ll={r['cv_log_loss']:.6f}  cv_brier={r['cv_brier']:.6f}  {r['config']}")
    print("    Bottom 3 configs (worst):")
    for r in tuning_results[-3:]:
        print(f"      cv_ll={r['cv_log_loss']:.6f}  cv_brier={r['cv_brier']:.6f}  {r['config']}")

    # Save tuning results
    tuning_json = {
        "n_configs": len(tuning_results),
        "best_config": best_cfg,
        "best_cv_log_loss": tuning_results[0]["cv_log_loss"],
        "all_configs_sorted": [
            {
                "rank": i + 1,
                "config": r["config"],
                "cv_log_loss": r["cv_log_loss"],
                "cv_brier": r["cv_brier"],
            }
            for i, r in enumerate(tuning_results)
        ],
    }
    with open(DOCS / "baseline_tuning.json", "w") as f:
        json.dump(tuning_json, f, indent=2)
    print(f"    Tuning results saved → docs/baseline_tuning.json")

    # Step 2.3 — OOF predictions
    print("\n[4] Generating out-of-fold predictions for all 219,160 shots …")
    oof_df, trees_per_fold = run_oof_predictions(all_shots, fold_assign, best_cfg)

    # Save OOF predictions
    save_cols = [
        "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME",
        "SET_LABEL", "SHOT_VALUE", "PERIOD", "fold",
        TARGET, "p_A",
    ]
    oof_out = oof_df[[c for c in save_cols if c in oof_df.columns]].copy()
    oof_out.to_parquet(DATA / "oof_predictions.parquet", index=False)
    print(f"    OOF predictions saved → data/processed/oof_predictions.parquet")
    print(f"    Rows: {len(oof_out):,} | NaN p_A: {oof_out['p_A'].isna().sum()}")

    # Step 2.5 — baseline quality
    print("\n[5] Computing baseline quality metrics on REGULAR OOF predictions …")
    quality = compute_baseline_quality(oof_df)
    m = quality["model_A_metrics"]
    c = quality["constant_baseline_metrics"]
    print(f"    Model A  : Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  AUC={m['roc_auc']:.4f}  ECE={m['ece']:.4f}")
    print(f"    Constant : Brier={c['brier']:.4f}  LogLoss={c['log_loss']:.4f}  AUC={c['roc_auc']:.4f}  ECE={c['ece']:.4f}")
    auc_ok = quality["auc_leakage_check"]["ok"]
    print(f"    AUC leakage check (<0.75): {'✓ PASS' if auc_ok else '⚠ FAIL — investigate!'}")
    if not auc_ok:
        raise ValueError(f"STOP: AUC={m['roc_auc']:.4f} >= 0.75 — possible leakage.")

    print("\n    Residuals by SHOT_VALUE:")
    for r in quality["residuals_by_shot_value"]:
        print(f"      {r['label']}: mean={r['mean_residual_pp']:+.3f} pp  (n={r['n']:,})")
    print("\n    Residuals by ACTION_GROUP:")
    for r in quality["residuals_by_action_group"]:
        print(f"      {r['label']}: mean={r['mean_residual_pp']:+.3f} pp  (n={r['n']:,})")
    print("\n    Residuals by distance bucket:")
    for r in quality["residuals_by_distance"]:
        print(f"      {r['label']}: mean={r['mean_residual_pp']:+.3f} pp  (n={r['n']:,})")
    print("\n    Residuals by PERIOD:")
    for r in quality["residuals_by_period"]:
        print(f"      {r['label']}: mean={r['mean_residual_pp']:+.3f} pp  (n={r['n']:,})")

    quality["trees_per_fold"] = trees_per_fold
    quality["best_config"] = best_cfg
    with open(DOCS / "baseline_quality.json", "w") as f:
        json.dump(quality, f, indent=2)
    print(f"\n    Baseline quality saved → docs/baseline_quality.json")
    print("\n✓ Step 2 complete.")


if __name__ == "__main__":
    main()
