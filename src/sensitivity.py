"""Sensitivity analyses for the clutch residual (sensitivity, not pre-specified).

S1: Two alternative XGBoost configurations:
    - S1a: beyond the edges where the best tuning config sat
      (max_depth=6, learning_rate=0.02, min_child_weight=100, reg_lambda=10.0)
    - S1b: the 5th-best configuration from docs/baseline_tuning.json
S2: A logistic-regression baseline on REGULAR shots only, cross-fitted
    with the same 5-fold assignment, using shot-context features only
    (distance, distance^2, distance^3, |angle|, SHOT_VALUE, one-hot
    ACTION_GROUP, IS_PUTBACK; NO time, margin, home, or overtime).

For each alternative, we recompute Model A (cross-fitted OOF predictions on
REGULAR only), Model B (DerSimonian-Laird player offsets from REGULAR OOF
residuals), and the overall clutch residual with 2,000-resample game-level
cluster bootstrap CIs (seed 42).

Hard rules:
- CLUTCH and LATE_BLOWOUT shots never enter any training or early-stopping set.
- Same fold assignment (data/processed/fold_assignment.parquet) as the primary
  analysis.
- Results are labelled "sensitivity, not pre-specified" and must not change
  the primary result.

Outputs
-------
docs/sensitivity.json
    Clutch residuals (Model A and Model B), differences vs reference XGBoost,
    and REGULAR OOF quality metrics for each sensitivity model.

Run from the repo root::

    python src/sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_model import (
    N_FOLDS,
    RNG_SEED,
    TARGET,
    _X,
    assert_no_clutch_in_training,
)
from src.clutch_effect import _residual_row
from src.evaluate import compute_metrics
from src.features import ACTION_CATEGORIES
from src.shooter_offsets import (
    P_CLIP_HIGH,
    P_CLIP_LOW,
    compute_player_offsets,
)

DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

LR_ALLOWED_RAW_COLS = {
    "SHOT_DISTANCE_FT",
    "SHOT_ANGLE",
    "SHOT_VALUE",
    "IS_PUTBACK",
    "ACTION_GROUP",
}


# ---------------------------------------------------------------------------
# Shared: apply shooter offsets and compute clutch residual
# ---------------------------------------------------------------------------

def _apply_offsets_and_residual(
    shots: pd.DataFrame,
    p_A_col: str,
    ref_A_pp: float,
    ref_B_pp: float,
) -> Dict[str, Any]:
    """Given cross-fitted p_A predictions, fit Model B offsets and evaluate clutch residual."""
    df = shots.copy()
    if "SET_LABEL" not in df.columns:
        df["SET_LABEL"] = "REGULAR"
        df.loc[df["CLUTCH"], "SET_LABEL"] = "CLUTCH"
        df.loc[df["LATE_BLOWOUT"], "SET_LABEL"] = "LATE_BLOWOUT"

    df["p_A"] = df[p_A_col].values
    df["b_player"] = 0.0
    df["p_B"] = df["p_A"].copy()

    fold_tau_pp: list[float] = []
    for k in range(N_FOLDS):
        tr_mask = (df["fold"] != k) & (df["SET_LABEL"] == "REGULAR")
        val_mask = df["fold"] == k
        tr = df[tr_mask].copy()
        val_idx = np.where(val_mask.values)[0]

        result = compute_player_offsets(tr)
        if isinstance(result, tuple):
            offsets, tau2, _rbar, _stats_v = result
            fold_tau_pp.append(round(float(np.sqrt(tau2)) * 100.0, 3))
        else:
            offsets = result
            fold_tau_pp.append(0.0)

        val_rows = df.iloc[val_idx]
        b_vals = val_rows["PLAYER_ID"].astype(str).map(
            lambda pid: offsets.get(str(pid), 0.0)
        ).values
        df.iloc[val_idx, df.columns.get_loc("b_player")] = b_vals
        df.iloc[val_idx, df.columns.get_loc("p_B")] = np.clip(
            val_rows["p_A"].values + b_vals, P_CLIP_LOW, P_CLIP_HIGH
        )

    reg = df[df["SET_LABEL"] == "REGULAR"]
    reg_metrics_A = compute_metrics(reg[TARGET].values, reg["p_A"].values)
    reg_metrics_B = compute_metrics(reg[TARGET].values, reg["p_B"].values)

    clutch = df[df["SET_LABEL"] == "CLUTCH"]
    rng = np.random.default_rng(RNG_SEED)
    resid_A = _residual_row("CLUTCH overall (Model A)", clutch, "p_A", rng)
    resid_B = _residual_row("CLUTCH overall (Model B)", clutch, "p_B", rng)

    return {
        "regular_oof_metrics": {
            "model_A": reg_metrics_A,
            "model_B": reg_metrics_B,
        },
        "fold_tau_pp": fold_tau_pp,
        "clutch_residual_A": resid_A,
        "clutch_residual_B": resid_B,
        "diff_vs_reference_pp": {
            "model_A": round(resid_A["residual_pp"] - ref_A_pp, 3),
            "model_B": round(resid_B["residual_pp"] - ref_B_pp, 3),
        },
    }


# ---------------------------------------------------------------------------
# S1: Alternative XGBoost configurations
# ---------------------------------------------------------------------------

def run_xgb_alternative(
    all_shots: pd.DataFrame,
    fold_assign: pd.DataFrame,
    cfg: Dict[str, Any],
    cfg_label: str,
    ref_A_pp: float,
    ref_B_pp: float,
) -> Dict[str, Any]:
    """Cross-fit an alternative XGBoost config on REGULAR shots only."""
    print(f"\n  [{cfg_label}] Config: {cfg}")
    shots = all_shots.merge(fold_assign, on="GAME_ID", how="left")
    p_A = np.full(len(shots), np.nan)
    trees_per_fold: list[int] = []

    for k in range(N_FOLDS):
        tr_mask = (shots["fold"] != k) & shots["REGULAR"]
        val_mask = shots["fold"] == k
        tr = shots[tr_mask].copy()
        val = shots[val_mask].copy()

        assert_no_clutch_in_training(tr, fold_k=k, phase=f"{cfg_label}-outer-train")

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG_SEED + k)
        inner_tr_idx, inner_val_idx = next(gss.split(tr, groups=tr["GAME_ID"]))
        inner_tr = tr.iloc[inner_tr_idx]
        inner_val = tr.iloc[inner_val_idx]

        assert_no_clutch_in_training(inner_tr, fold_k=k, phase=f"{cfg_label}-inner-train")

        m_inner = xgb.XGBClassifier(
            n_estimators=1000,
            enable_categorical=True,
            tree_method="hist",
            random_state=RNG_SEED,
            eval_metric="logloss",
            early_stopping_rounds=25,
            n_jobs=6,
            **cfg,
        )
        m_inner.fit(
            _X(inner_tr), inner_tr[TARGET],
            eval_set=[(_X(inner_val), inner_val[TARGET])],
            verbose=False,
        )
        n_trees = m_inner.best_iteration + 1
        trees_per_fold.append(int(n_trees))

        m_fold = xgb.XGBClassifier(
            n_estimators=n_trees,
            enable_categorical=True,
            tree_method="hist",
            random_state=RNG_SEED,
            eval_metric="logloss",
            n_jobs=6,
            **cfg,
        )
        assert_no_clutch_in_training(tr, fold_k=k, phase=f"{cfg_label}-refit")
        m_fold.fit(_X(tr), tr[TARGET])
        val_idx_pos = np.where(val_mask.values)[0]
        p_A[val_idx_pos] = m_fold.predict_proba(_X(val))[:, 1]
        print(f"    fold {k}: {len(tr):,} REGULAR train shots, {n_trees} trees")

    shots["p_A_alt"] = p_A
    result = _apply_offsets_and_residual(shots, "p_A_alt", ref_A_pp, ref_B_pp)
    result["label"] = cfg_label
    result["config"] = cfg
    result["trees_per_fold"] = trees_per_fold
    return result


# ---------------------------------------------------------------------------
# S2: Logistic regression baseline (shot-context features only)
# ---------------------------------------------------------------------------

def build_lr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build shot-context-only feature DataFrame for logistic regression.

    Features:
    - SHOT_DISTANCE_FT, distance^2, distance^3
    - |SHOT_ANGLE|
    - SHOT_VALUE
    - IS_PUTBACK
    - One-hot ACTION_GROUP (fixed categories)

    Deliberately excluded: TIME_REMAINING_SEC, SCORE_MARGIN, IS_HOME, IS_OT,
    PERIOD, and any player/team identifier.
    """
    dist = df["SHOT_DISTANCE_FT"].values.astype(float)
    angle = np.abs(df["SHOT_ANGLE"].values.astype(float))
    sv = df["SHOT_VALUE"].values.astype(float)
    putback = df["IS_PUTBACK"].values.astype(float)

    ag = df["ACTION_GROUP"].astype(str).to_numpy(dtype=object).reshape(-1, 1)
    enc = OneHotEncoder(
        categories=[ACTION_CATEGORIES],
        sparse_output=False,
        handle_unknown="ignore",
    )
    ag_onehot = enc.fit_transform(ag)

    cols = ["dist", "dist_sq", "dist_cube", "abs_angle", "shot_value", "is_putback"] + [
        f"action_{c}" for c in ACTION_CATEGORIES
    ]
    data = np.column_stack([dist, dist ** 2, dist ** 3, angle, sv, putback, ag_onehot])
    return pd.DataFrame(data, columns=cols, index=df.index)


def run_logistic_regression(
    all_shots: pd.DataFrame,
    fold_assign: pd.DataFrame,
    ref_A_pp: float,
    ref_B_pp: float,
) -> Dict[str, Any]:
    """Cross-fit logistic regression baseline on REGULAR shots only."""
    print("\n  [S2_logistic_regression] Shot-context features only, StandardScaler per fold")
    shots = all_shots.merge(fold_assign, on="GAME_ID", how="left")
    X_all = build_lr_features(shots)
    p_A = np.full(len(shots), np.nan)

    for k in range(N_FOLDS):
        tr_mask = (shots["fold"] != k) & shots["REGULAR"]
        val_mask = shots["fold"] == k
        tr = shots[tr_mask]

        assert_no_clutch_in_training(tr, fold_k=k, phase="S2-LR-train")

        X_tr_raw = X_all[tr_mask]
        X_val_raw = X_all[val_mask]
        y_tr = tr[TARGET].values

        # Exclude zero-variance columns in the training fold (e.g., unused categories)
        non_const = [c for c in X_tr_raw.columns if X_tr_raw[c].nunique() > 1]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw[non_const])
        X_val_scaled = scaler.transform(X_val_raw[non_const])

        lr = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RNG_SEED,
            C=1.0,
        )
        lr.fit(X_tr_scaled, y_tr)
        val_idx_pos = np.where(val_mask.values)[0]
        p_A[val_idx_pos] = lr.predict_proba(X_val_scaled)[:, 1]
        print(f"    fold {k}: {len(tr):,} REGULAR train shots, {len(non_const)} active features, n_iter={int(lr.n_iter_[0])}")

    shots["p_A_lr"] = p_A
    result = _apply_offsets_and_residual(shots, "p_A_lr", ref_A_pp, ref_B_pp)
    result["label"] = "S2_logistic_regression"
    result["features"] = list(X_all.columns)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 3b — Sensitivity analyses (sensitivity, not pre-specified)")
    print("=" * 60)

    all_shots = pd.read_parquet(DATA / "all_shots.parquet")
    all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(str)
    fold_assign = pd.read_parquet(DATA / "fold_assignment.parquet")
    fold_assign["GAME_ID"] = fold_assign["GAME_ID"].astype(str)

    with open(DOCS / "clutch_effect.json") as f:
        ref = json.load(f)
    ref_A = ref["primary_result"]["model_A"]
    ref_B = ref["primary_result"]["model_B"]
    ref_A_pp = float(ref_A["residual_pp"])
    ref_B_pp = float(ref_B["residual_pp"])

    print(f"\n  Reference (original XGBoost):")
    print(f"    Model A: {ref_A_pp:+.3f} pp  95% CI [{ref_A['ci_lower_pp']:+.3f}, {ref_A['ci_upper_pp']:+.3f}]")
    print(f"    Model B: {ref_B_pp:+.3f} pp  95% CI [{ref_B['ci_lower_pp']:+.3f}, {ref_B['ci_upper_pp']:+.3f}]")

    results: Dict[str, Any] = {
        "status": "sensitivity, not pre-specified",
        "note": (
            "Sensitivity analyses (not pre-specified). They check robustness of "
            "the clutch residual to XGBoost hyperparameters beyond the tuning grid "
            "edges (S1) and to a parametric logistic-regression baseline (S2). "
            "They do not change the pre-specified primary result."
        ),
        "reference": {"model_A": ref_A, "model_B": ref_B},
    }

    # S1a: Beyond grid edges
    cfg_s1a = {
        "max_depth": 6,
        "learning_rate": 0.02,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
    }
    results["S1a_beyond_edges"] = run_xgb_alternative(
        all_shots, fold_assign, cfg_s1a, "S1a_beyond_edges", ref_A_pp, ref_B_pp
    )

    # S1b: 5th-best config from original tuning
    with open(DOCS / "baseline_tuning.json") as f:
        tuning = json.load(f)
    cfg_s1b = tuning["all_configs_sorted"][4]["config"]  # rank 5
    results["S1b_5th_best"] = run_xgb_alternative(
        all_shots, fold_assign, cfg_s1b, "S1b_5th_best", ref_A_pp, ref_B_pp
    )

    # S2: Logistic regression baseline
    results["S2_logistic_regression"] = run_logistic_regression(
        all_shots, fold_assign, ref_A_pp, ref_B_pp
    )

    # Summary printout
    print("\n" + "=" * 60)
    print("Sensitivity summary (sensitivity, not pre-specified)")
    print("=" * 60)
    print(f"\n  {'Model':<26} {'Model A (pp)':>22} {'Model B (pp)':>22} {'ΔB vs ref':>10}")
    print("  " + "-" * 84)
    print(
        f"  {'Reference (XGB best)':<26} "
        f"{ref_A_pp:>+6.3f} [{ref_A['ci_lower_pp']:>+6.3f},{ref_A['ci_upper_pp']:>+6.3f}] "
        f"{ref_B_pp:>+6.3f} [{ref_B['ci_lower_pp']:>+6.3f},{ref_B['ci_upper_pp']:>+6.3f}] "
        f"{0.0:>+10.3f}"
    )
    for key in ["S1a_beyond_edges", "S1b_5th_best", "S2_logistic_regression"]:
        r = results[key]
        ra = r["clutch_residual_A"]
        rb = r["clutch_residual_B"]
        db = r["diff_vs_reference_pp"]["model_B"]
        print(
            f"  {key:<26} "
            f"{ra['residual_pp']:>+6.3f} [{ra['ci_lower_pp']:>+6.3f},{ra['ci_upper_pp']:>+6.3f}] "
            f"{rb['residual_pp']:>+6.3f} [{rb['ci_lower_pp']:>+6.3f},{rb['ci_upper_pp']:>+6.3f}] "
            f"{db:>+10.3f}"
        )

    with open(DOCS / "sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → docs/sensitivity.json")
    print("\n✓ Sensitivity analyses complete.")


if __name__ == "__main__":
    main()
