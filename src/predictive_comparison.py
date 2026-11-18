"""Phase 3b Step 7: predictive comparison on the 7,104 clutch shots.

Compares four models out-of-fold on the exact same 7,104 CLUTCH shots and the
exact same 5-fold game-level fold assignment (data/processed/fold_assignment.parquet):

    (i)   Model A: shot-only XGBoost baseline trained on REGULAR shots only.
    (ii)  Model B: Model A + shrunk per-player offset estimated on REGULAR shots only.
    (iii) Model B + clutch shift: Model B plus a single constant logit-scale shift
          delta_k estimated ONLY from CLUTCH shots in folds != k.
    (iv)  Clutch-only XGBoost: XGBoost with the Phase 3 hyperparameters, cross-fitted
          from scratch on CLUTCH shots only (no Phase 3 model artifacts are loaded).

EXCEPTION TO HARD RULE §3.1 (explicitly scoped to Step 7):
    For model (iii) in this predictive comparison ONLY, clutch outcomes from the
    OTHER folds (fold != k) are used to estimate the scalar logit-scale shift
    delta_k, and for model (iv) clutch outcomes from folds != k are used to fit
    the clutch-only XGBoost benchmark.  Fold-k clutch outcomes are NEVER used to
    fit, tune, or shift fold-k predictions.  Models (i) and (ii) remain strictly
    fitted on REGULAR shots only.

Metrics: Brier score, log loss, ROC AUC (plus ECE and mean residual for context),
with 2,000-resample game-level cluster bootstrap CIs and paired game-level
bootstrap differences (default_rng(42)) across the 543 clutch games.

Outputs
-------
docs/predictive_comparison.json
    Individual model metrics with 95% CIs, per-fold shift/tree diagnostics,
    and all paired game-level bootstrap differences.

Run from the repo root::

    python src/predictive_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics
from src.features import ACTION_DTYPE, FEATURES, TARGET

DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
N_BOOT = 2_000
RNG_SEED = 42
ALPHA = 0.05

# Phase 3 final XGBoost hyperparameters (copied from Phase 3 docs/metrics.json;
# we never load any Phase 3 model artifact from models/).
PHASE3_XGB_PARAMS: Dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
}


# ---------------------------------------------------------------------------
# Model (iii): Model B + out-of-fold constant logit-scale clutch shift
# ---------------------------------------------------------------------------

def fit_logit_shift(p: np.ndarray, y: np.ndarray) -> float:
    """Find scalar delta minimizing log loss of expit(logit(p) + delta).

    Solves sum(expit(logit(p) + delta)) == sum(y) via brentq (globally
    convex MLE; calibrates mean predicted probability to observed make rate).
    """
    p_clip = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y_arr = np.asarray(y, dtype=float)
    eta = logit(p_clip)
    target_sum = float(y_arr.sum())

    def _grad(delta: float) -> float:
        return float(expit(eta + delta).sum() - target_sum)

    return float(brentq(_grad, -10.0, 10.0))


def compute_oof_shifted_model_b(
    clutch_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Compute OOF predictions for Model B + constant logit-scale clutch shift.

    EXCEPTION (Step 7 only): Uses CLUTCH outcomes from folds != k to estimate
    the scalar logit-scale shift delta_k.  Fold-k CLUTCH outcomes are NEVER
    used for fold-k predictions.
    """
    p_b = clutch_df["p_B"].values.astype(float)
    y = clutch_df[TARGET].values.astype(float)
    folds = clutch_df["fold"].values
    games = clutch_df["GAME_ID"].values.astype(str)

    p_out = np.full(len(clutch_df), np.nan, dtype=float)
    fold_info: List[Dict[str, Any]] = []

    for k in range(N_FOLDS):
        tr_mask = folds != k
        val_mask = folds == k

        # Strict no-leakage check: zero game overlap between train and val
        overlap = set(games[tr_mask]) & set(games[val_mask])
        if overlap:
            raise ValueError(f"Fold {k} game overlap in clutch shift fit: {len(overlap)} games")

        delta_k = fit_logit_shift(p_b[tr_mask], y[tr_mask])
        eta_val = logit(np.clip(p_b[val_mask], 1e-6, 1.0 - 1e-6))
        p_out[val_mask] = expit(eta_val + delta_k)

        fold_info.append({
            "fold": int(k),
            "n_train_clutch": int(tr_mask.sum()),
            "n_val_clutch": int(val_mask.sum()),
            "delta_logit": round(delta_k, 5),
        })

    if np.isnan(p_out).any():
        raise RuntimeError("NaN predictions encountered in Model B + clutch shift.")

    return p_out, fold_info


# ---------------------------------------------------------------------------
# Model (iv): Clutch-only XGBoost (Phase 3 hyperparameters, cross-fitted)
# ---------------------------------------------------------------------------

def _X_phase3(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Phase 3 feature columns with fixed categorical ACTION_GROUP."""
    out = df[FEATURES].copy()
    out["ACTION_GROUP"] = pd.Categorical(out["ACTION_GROUP"], dtype=ACTION_DTYPE)
    return out


def compute_oof_clutch_only_xgb(
    clutch_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Cross-fit clutch-only XGBoost with Phase 3 hyperparameters on the same folds.

    Uses inner 80/20 grouped early stopping (same protocol as src/train.py) so
    fold-k games are never seen during training or early stopping.  Does NOT
    load any Phase 3 artifact from models/.
    """
    folds = clutch_df["fold"].values
    games = clutch_df["GAME_ID"].values.astype(str)
    p_out = np.full(len(clutch_df), np.nan, dtype=float)
    fold_info: List[Dict[str, Any]] = []

    for k in range(N_FOLDS):
        tr_mask = folds != k
        val_mask = folds == k

        overlap = set(games[tr_mask]) & set(games[val_mask])
        if overlap:
            raise ValueError(f"Fold {k} game overlap in clutch-only XGBoost: {len(overlap)} games")

        tr = clutch_df[tr_mask].copy()
        val = clutch_df[val_mask].copy()

        # Inner 80/20 grouped split of the outer training fold for early stopping
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG_SEED + k)
        inner_tr_idx, inner_val_idx = next(gss.split(tr, groups=tr["GAME_ID"]))
        inner_tr = tr.iloc[inner_tr_idx]
        inner_val = tr.iloc[inner_val_idx]

        m_inner = xgb.XGBClassifier(
            n_estimators=300,
            enable_categorical=True,
            tree_method="hist",
            random_state=RNG_SEED,
            eval_metric="logloss",
            early_stopping_rounds=25,
            n_jobs=6,
            **PHASE3_XGB_PARAMS,
        )
        m_inner.fit(
            _X_phase3(inner_tr), inner_tr[TARGET],
            eval_set=[(_X_phase3(inner_val), inner_val[TARGET])],
            verbose=False,
        )
        n_trees = int(m_inner.best_iteration + 1)

        m_fold = xgb.XGBClassifier(
            n_estimators=n_trees,
            enable_categorical=True,
            tree_method="hist",
            random_state=RNG_SEED,
            eval_metric="logloss",
            n_jobs=6,
            **PHASE3_XGB_PARAMS,
        )
        m_fold.fit(_X_phase3(tr), tr[TARGET])
        p_out[val_mask] = m_fold.predict_proba(_X_phase3(val))[:, 1]

        fold_info.append({
            "fold": int(k),
            "n_train_clutch": int(len(tr)),
            "n_val_clutch": int(len(val)),
            "n_trees": n_trees,
        })

    if np.isnan(p_out).any():
        raise RuntimeError("NaN predictions encountered in clutch-only XGBoost.")

    return p_out, fold_info


# ---------------------------------------------------------------------------
# Joint game-level cluster bootstrap for metrics and paired differences
# ---------------------------------------------------------------------------

def bootstrap_model_comparison(
    game_ids: np.ndarray,
    y: np.ndarray,
    models: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    n_resamples: int = N_BOOT,
    random_state: int = RNG_SEED,
    alpha: float = ALPHA,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute individual CIs and paired differences on the same resampled games.

    Pairs are specified as ``(model_b, model_a)`` and differences are reported as
    ``model_b - model_a`` (for Brier and log loss, negative means ``model_b``
    is better; for ROC AUC, positive means ``model_b`` is better).
    """
    g_arr = np.asarray(game_ids, dtype=str)
    y_t = np.asarray(y, dtype=int)

    point_metrics = {name: compute_metrics(y_t, p) for name, p in models.items()}
    mean_resid_pp = {
        name: round(float((y_t - p).mean() * 100.0), 3)
        for name, p in models.items()
    }

    unique_games = np.sort(np.unique(g_arr))
    n_games = len(unique_games)
    game_df = pd.DataFrame({"game": g_arr, "idx": np.arange(len(g_arr))})
    game_groups = game_df.groupby("game")["idx"].apply(np.array).to_dict()

    rng = np.random.default_rng(random_state)
    tracked = ("brier", "log_loss", "roc_auc")
    boot_vals = {
        name: {m: np.empty(n_resamples, dtype=float) for m in tracked}
        for name in models
    }

    p_clipped = {
        name: np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
        for name, p in models.items()
    }
    sse_all = {name: (y_t - p_clipped[name]) ** 2 for name in models}
    ll_all = {
        name: -(y_t * np.log(p_clipped[name]) + (1 - y_t) * np.log(1.0 - p_clipped[name]))
        for name in models
    }

    for b in range(n_resamples):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        idx = np.concatenate([game_groups[g] for g in sampled_games])
        y_b = y_t[idx]
        for name in models:
            boot_vals[name]["brier"][b] = float(sse_all[name][idx].mean())
            boot_vals[name]["log_loss"][b] = float(ll_all[name][idx].mean())
            boot_vals[name]["roc_auc"][b] = float(roc_auc_score(y_b, p_clipped[name][idx]))

    lo_p = 100.0 * (alpha / 2.0)
    hi_p = 100.0 * (1.0 - alpha / 2.0)

    indiv_out: Dict[str, Any] = {}
    for name in models:
        entry: Dict[str, Any] = {}
        for m in tracked:
            arr = boot_vals[name][m]
            entry[m] = {
                "point": round(point_metrics[name][m], 6),
                "ci_lower": round(float(np.percentile(arr, lo_p)), 6),
                "ci_upper": round(float(np.percentile(arr, hi_p)), 6),
                "std_err": round(float(np.std(arr, ddof=1)), 6),
            }
        entry["ece_point"] = round(point_metrics[name]["ece"], 6)
        entry["mean_residual_pp"] = mean_resid_pp[name]
        indiv_out[name] = entry

    paired_out: Dict[str, Any] = {}
    for m_b, m_a in pairs:
        pair_key = f"{m_b} minus {m_a}"
        pair_entry: Dict[str, Any] = {}
        for m in tracked:
            point_diff = point_metrics[m_b][m] - point_metrics[m_a][m]
            arr_diff = boot_vals[m_b][m] - boot_vals[m_a][m]
            p_pos = float(np.mean(arr_diff >= 0.0))
            p_neg = float(np.mean(arr_diff <= 0.0))
            raw_p = min(1.0, 2.0 * min(p_pos, p_neg))
            p_str = f"< {1.0 / n_resamples:.4f}" if raw_p < (1.0 / n_resamples) else f"{raw_p:.4f}"
            pair_entry[m] = {
                "point_diff": round(float(point_diff), 6),
                "ci_lower": round(float(np.percentile(arr_diff, lo_p)), 6),
                "ci_upper": round(float(np.percentile(arr_diff, hi_p)), 6),
                "std_err": round(float(np.std(arr_diff, ddof=1)), 6),
                "p_boot_display": p_str,
            }
        paired_out[pair_key] = pair_entry

    return indiv_out, paired_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Phase 3b — Step 7: predictive_comparison.py (7,104 CLUTCH shots)")
    print("=" * 65)

    print("\n[1] Loading OOF predictions and Phase 3 features for CLUTCH shots …")
    oof = pd.read_parquet(DATA / "oof_predictions.parquet")
    oof["GAME_ID"] = oof["GAME_ID"].astype(str)
    all_shots = pd.read_parquet(DATA / "all_shots.parquet")
    all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(str)

    clutch_oof = oof[oof["SET_LABEL"] == "CLUTCH"].copy()
    feat_cols = [c for c in FEATURES if c not in clutch_oof.columns]
    clutch_df = clutch_oof.merge(
        all_shots[["GAME_ID", "GAME_EVENT_ID"] + feat_cols],
        on=["GAME_ID", "GAME_EVENT_ID"],
        how="left",
        validate="one_to_one",
    )
    print(f"    CLUTCH shots: {len(clutch_df):,} across {clutch_df['GAME_ID'].nunique()} games")
    assert len(clutch_df) == 7104, f"Expected 7,104 clutch shots, got {len(clutch_df)}"

    print("\n[2] Fitting Model (iii): Model B + OOF constant logit-scale clutch shift …")
    print("    [Step 7 exception: clutch outcomes from folds != k estimate delta_k; fold k untouched]")
    p_b_shift, shift_folds = compute_oof_shifted_model_b(clutch_df)
    for fi in shift_folds:
        print(f"    fold {fi['fold']}: n_train={fi['n_train_clutch']:,}, "
              f"n_val={fi['n_val_clutch']:,}, delta_logit={fi['delta_logit']:+.5f}")

    print("\n[3] Fitting Model (iv): Clutch-only XGBoost (Phase 3 hyperparams, same 5 folds) …")
    p_clutch_xgb, xgb_folds = compute_oof_clutch_only_xgb(clutch_df)
    for fi in xgb_folds:
        print(f"    fold {fi['fold']}: n_train={fi['n_train_clutch']:,}, "
              f"n_val={fi['n_val_clutch']:,}, trees={fi['n_trees']}")

    models = {
        "Model A (regular shot-only)": clutch_df["p_A"].values.astype(float),
        "Model B (regular + shooter)": clutch_df["p_B"].values.astype(float),
        "Model B + clutch shift": p_b_shift,
        "Clutch-only XGBoost (Phase 3)": p_clutch_xgb,
    }

    pairs = [
        ("Model B (regular + shooter)", "Model A (regular shot-only)"),
        ("Model B + clutch shift", "Model A (regular shot-only)"),
        ("Model B + clutch shift", "Model B (regular + shooter)"),
        ("Model B + clutch shift", "Clutch-only XGBoost (Phase 3)"),
        ("Model B (regular + shooter)", "Clutch-only XGBoost (Phase 3)"),
        ("Model A (regular shot-only)", "Clutch-only XGBoost (Phase 3)"),
    ]

    print(f"\n[4] Running game-level cluster bootstrap ({N_BOOT:,} resamples, seed {RNG_SEED}) …")
    indiv, paired = bootstrap_model_comparison(
        clutch_df["GAME_ID"].values,
        clutch_df[TARGET].values,
        models,
        pairs,
        n_resamples=N_BOOT,
        random_state=RNG_SEED,
    )

    print("\n--- Individual Model Metrics on 7,104 CLUTCH Shots (5-fold OOF) ---")
    for name, m in indiv.items():
        print(
            f"  {name:<32} "
            f"Brier={m['brier']['point']:.4f} [{m['brier']['ci_lower']:.4f},{m['brier']['ci_upper']:.4f}]  "
            f"LL={m['log_loss']['point']:.4f} [{m['log_loss']['ci_lower']:.4f},{m['log_loss']['ci_upper']:.4f}]  "
            f"AUC={m['roc_auc']['point']:.4f} [{m['roc_auc']['ci_lower']:.4f},{m['roc_auc']['ci_upper']:.4f}]  "
            f"resid={m['mean_residual_pp']:+.2f}pp"
        )

    print("\n--- Paired Game-Level Bootstrap Differences (2,000 resamples) ---")
    for pair_key, pm in paired.items():
        print(f"  {pair_key}:")
        for metric in ("brier", "log_loss", "roc_auc"):
            s = pm[metric]
            print(
                f"    {metric:<8}: {s['point_diff']:+.5f}  "
                f"95% CI [{s['ci_lower']:+.5f}, {s['ci_upper']:+.5f}]  "
                f"(p_boot {s['p_boot_display']})"
            )

    out = {
        "meta": {
            "description": (
                "Out-of-fold predictive comparison on all 7,104 CLUTCH shots across "
                "543 games using the fixed 5-fold game-level fold assignment."
            ),
            "n_clutch_shots": int(len(clutch_df)),
            "n_clutch_games": int(clutch_df["GAME_ID"].nunique()),
            "n_bootstrap": N_BOOT,
            "rng_seed": RNG_SEED,
            "step_7_exception": (
                "For Step 7 ONLY, clutch outcomes from folds != k are used to estimate "
                "the constant logit-scale shift delta_k in 'Model B + clutch shift' "
                "and to fit the 'Clutch-only XGBoost' benchmark. Fold-k clutch outcomes "
                "are NEVER used for fold-k predictions. Model A and Model B themselves "
                "remain fitted exclusively on REGULAR shots."
            ),
            "phase3_artifacts_loaded": False,
            "phase3_hyperparameters_used": PHASE3_XGB_PARAMS,
            "phase3_features_used": FEATURES,
        },
        "model_b_shift_folds": shift_folds,
        "clutch_only_xgb_folds": xgb_folds,
        "individual_metrics": indiv,
        "paired_differences": paired,
    }

    with open(DOCS / "predictive_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n    Saved → docs/predictive_comparison.json")
    print("\n✓ Step 7 complete.")


if __name__ == "__main__":
    main()
