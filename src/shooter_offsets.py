"""Shooter-skill offsets (Model B) for Phase 3b.

Estimates per-player shrunk offsets using the DerSimonian–Laird (DL)
empirical-Bayes estimator, fitted exclusively on REGULAR out-of-fold
residuals from Model A.

The offset b_p for player p shrinks the raw per-player residual r_p toward
the weighted grand mean rbar.  Shrinkage is heavier for players with few
regular-play shots (large s2_p) and lighter when between-player variance
tau2 is large.  Players with no regular-play shots in the training fold
receive b_p = 0.

Model B:  p_B = clip(p_A + b_p, 0.01, 0.99)

Documented limitation (brief §7 step 5):
    The out-of-fold residuals used in step 1 come from models trained on
    folds that may have seen fold k's games during hyperparameter tuning.
    This is a second-order effect (tuning did not observe outcomes directly)
    and is expected to be negligible, but it means Model B's uncertainty is
    slightly understated.

Outputs
-------
data/processed/oof_predictions.parquet
    Updated in-place with columns p_B and b_player.
docs/shooter_offsets.json
    Per-fold DL statistics and Model B validity check.

Run from the repo root::

    python src/shooter_offsets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import bootstrap_paired_difference, compute_metrics  # noqa: E402

DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

TARGET = "SHOT_MADE_FLAG"
N_FOLDS = 5
RNG_SEED = 42
P_CLIP_LOW = 0.01
P_CLIP_HIGH = 0.99


# ---------------------------------------------------------------------------
# DerSimonian–Laird estimator
# ---------------------------------------------------------------------------

def dersimonian_laird(
    r_p: np.ndarray,
    n_p: np.ndarray,
    s2_p: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """DerSimonian–Laird empirical-Bayes estimator for between-player variance.

    Parameters
    ----------
    r_p  : shape (P,)  — per-player raw residual mean
    n_p  : shape (P,)  — per-player shot count
    s2_p : shape (P,)  — per-player sampling variance of r_p
               = sum(p_A*(1-p_A)) / n_p^2  (binomial variance of mean)

    Returns
    -------
    tau2  : between-player variance (>= 0)
    rbar  : weighted grand mean
    b_p   : shrunk offsets, shape (P,)

    Reference: DerSimonian & Laird (1986) "Meta-analysis in clinical trials."
    """
    P = len(r_p)
    w_p = 1.0 / s2_p  # inverse-variance weights

    rbar = float(np.sum(w_p * r_p) / np.sum(w_p))
    Q = float(np.sum(w_p * (r_p - rbar) ** 2))

    # DL tau2 estimate
    denom = np.sum(w_p) - np.sum(w_p ** 2) / np.sum(w_p)
    tau2 = max(0.0, (Q - (P - 1)) / denom) if denom > 0 else 0.0

    # Shrinkage factor: tau2 / (tau2 + s2_p)
    shrinkage = tau2 / (tau2 + s2_p)
    b_p = rbar + (r_p - rbar) * shrinkage

    return tau2, rbar, b_p


def compute_player_offsets(
    regular_train_df: pd.DataFrame,
) -> dict[str, float]:
    """Estimate per-player shrunk offsets from REGULAR out-of-fold residuals.

    Parameters
    ----------
    regular_train_df:
        REGULAR shots with columns PLAYER_ID, SHOT_MADE_FLAG, p_A.
        This DataFrame represents the REGULAR shots from all games
        *outside* fold k (the training set for fold k).

    Returns
    -------
    offsets:
        dict mapping PLAYER_ID (as str) -> b_p.
        Players with fewer than 1 regular-play shot get 0.0.
    """
    df = regular_train_df.copy()
    df["residual"] = df[TARGET] - df["p_A"]

    players = df.groupby("PLAYER_ID")
    stats = players.agg(
        r_p=("residual", "mean"),
        n_p=(TARGET, "count"),
        p_A_sum=("p_A", lambda x: (x * (1 - x)).sum()),
    ).reset_index()

    # Sampling variance of r_p under no true skill differences:
    # Var(r_p) = sum(p_A*(1-p_A)) / n_p^2
    stats["s2_p"] = stats["p_A_sum"] / (stats["n_p"] ** 2)

    # Exclude players with 0 variance (constant p_A, numerically unstable)
    valid = stats["s2_p"] > 0
    stats_v = stats[valid].copy()

    if len(stats_v) < 2:
        # Not enough players to estimate between-player variance
        return {str(pid): 0.0 for pid in stats["PLAYER_ID"]}

    tau2, rbar, b_p_arr = dersimonian_laird(
        stats_v["r_p"].values,
        stats_v["n_p"].values,
        stats_v["s2_p"].values,
    )

    offsets: dict[str, float] = {}
    for pid, bp in zip(stats_v["PLAYER_ID"], b_p_arr):
        offsets[str(pid)] = float(bp)
    # Players with s2_p == 0 get 0
    for pid in stats[~valid]["PLAYER_ID"]:
        offsets[str(pid)] = 0.0

    return offsets, tau2, rbar, stats_v


# ---------------------------------------------------------------------------
# Main offset pipeline
# ---------------------------------------------------------------------------

def run_shooter_offsets(
    oof_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute per-fold shooter offsets and produce p_B for every shot.

    For fold k:
    1. Collect REGULAR shots from games outside fold k (training fold).
    2. Estimate per-player DL offsets from their OOF residuals.
    3. Apply offsets to ALL shots in fold k (REGULAR + CLUTCH + LATE_BLOWOUT).

    Note on the second-order effect: the REGULAR OOF residuals used in step 1
    were produced by models that may have been tuned using fold k's games.
    Hyperparameter tuning never observed individual shot outcomes directly,
    so this is expected to be negligible, but it means Model B's uncertainty
    is slightly understated.

    Parameters
    ----------
    oof_df:
        Full OOF predictions table with columns:
        GAME_ID, GAME_EVENT_ID, PLAYER_ID, PLAYER_NAME, SET_LABEL,
        SHOT_VALUE, PERIOD, fold, SHOT_MADE_FLAG, p_A.

    Returns
    -------
    oof_df:
        Updated with columns p_B and b_player.
    fold_stats:
        List of per-fold DL statistics.
    """
    oof_df = oof_df.copy()
    oof_df["b_player"] = 0.0
    oof_df["p_B"] = oof_df["p_A"].copy()

    fold_stats: list[dict[str, Any]] = []

    for k in range(N_FOLDS):
        # Training fold: REGULAR shots outside fold k
        tr_mask = (oof_df["fold"] != k) & (oof_df["SET_LABEL"] == "REGULAR")
        val_mask = oof_df["fold"] == k

        tr = oof_df[tr_mask].copy()
        val_idx = np.where(val_mask.values)[0]

        # Estimate offsets
        result = compute_player_offsets(tr)
        if isinstance(result, tuple):
            offsets, tau2, rbar, stats_v = result
        else:
            offsets = result
            tau2, rbar = 0.0, 0.0
            stats_v = pd.DataFrame()

        tau = float(np.sqrt(tau2))
        n_players = len(offsets)

        if not stats_v.empty:
            shrinkage_factors = tau2 / (tau2 + stats_v["s2_p"].values)
            avg_shrinkage = float(np.mean(shrinkage_factors))
        else:
            avg_shrinkage = np.nan

        print(f"    fold {k}: {len(tr):,} REGULAR train shots | "
              f"{n_players} players | tau2={tau2:.6f} | tau={tau*100:.3f} pp | "
              f"avg_shrinkage={avg_shrinkage:.3f}")

        fold_stats.append({
            "fold": k,
            "n_regular_train": int(len(tr)),
            "n_players": n_players,
            "tau2": tau2,
            "tau_pp": tau * 100,
            "rbar": float(rbar),
            "avg_shrinkage_factor": avg_shrinkage,
        })

        # Apply offsets to all shots in this fold's validation set
        val_rows = oof_df.iloc[val_idx]
        b_vals = val_rows["PLAYER_ID"].astype(str).map(
            lambda pid: offsets.get(str(pid), 0.0)
        ).values
        oof_df.iloc[val_idx, oof_df.columns.get_loc("b_player")] = b_vals
        p_B_vals = np.clip(val_rows["p_A"].values + b_vals, P_CLIP_LOW, P_CLIP_HIGH)
        oof_df.iloc[val_idx, oof_df.columns.get_loc("p_B")] = p_B_vals

    return oof_df, fold_stats


# ---------------------------------------------------------------------------
# Model B validity check
# ---------------------------------------------------------------------------

def paired_game_bootstrap_brier_logloss(
    game_ids: np.ndarray,
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Fast, exact game-level paired cluster bootstrap for Brier score and Log Loss.

    Pre-aggregates per-game sums of squared errors, log losses, and shot counts.
    Mathematically identical to concatenating resampled games, but vectorized across games
    to run in milliseconds on large datasets (~200k shots).
    """
    p_a_clip = np.clip(p_a, 1e-15, 1.0 - 1e-15)
    p_b_clip = np.clip(p_b, 1e-15, 1.0 - 1e-15)

    df_game = pd.DataFrame({
        "game": game_ids,
        "n": 1,
        "sse_a": (y - p_a) ** 2,
        "sse_b": (y - p_b) ** 2,
        "ll_a": -(y * np.log(p_a_clip) + (1 - y) * np.log(1 - p_a_clip)),
        "ll_b": -(y * np.log(p_b_clip) + (1 - y) * np.log(1 - p_b_clip)),
    })
    agg = df_game.groupby("game", sort=False).sum()

    n_games = len(agg)
    n_vec = agg["n"].values
    sse_a_vec = agg["sse_a"].values
    sse_b_vec = agg["sse_b"].values
    ll_a_vec = agg["ll_a"].values
    ll_b_vec = agg["ll_b"].values

    total_n = n_vec.sum()
    point_diff_brier = float((sse_b_vec.sum() - sse_a_vec.sum()) / total_n)
    point_diff_ll = float((ll_b_vec.sum() - ll_a_vec.sum()) / total_n)

    rng = np.random.default_rng(random_state)
    sampled = rng.choice(n_games, size=(n_resamples, n_games), replace=True)

    N_boot = n_vec[sampled].sum(axis=1)
    diff_brier_boot = (sse_b_vec[sampled].sum(axis=1) - sse_a_vec[sampled].sum(axis=1)) / N_boot
    diff_ll_boot = (ll_b_vec[sampled].sum(axis=1) - ll_a_vec[sampled].sum(axis=1)) / N_boot

    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)

    def _stats(point_diff: float, arr: np.ndarray) -> dict[str, float]:
        p_pos = np.mean(arr >= 0)
        p_neg = np.mean(arr <= 0)
        two_sided_p = float(min(1.0, 2.0 * min(p_pos, p_neg)))
        return {
            "point_diff": point_diff,
            "ci_lower": float(np.percentile(arr, lower_p)),
            "ci_upper": float(np.percentile(arr, upper_p)),
            "std_err": float(np.std(arr, ddof=1)),
            "p_value": two_sided_p,
        }

    return {
        "brier": _stats(point_diff_brier, diff_brier_boot),
        "log_loss": _stats(point_diff_ll, diff_ll_boot),
    }


def validate_model_b(
    oof_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compare Model B vs Model A on held-out REGULAR shots (paired game bootstrap).

    If B does not beat A (lower Brier / log loss), tau2 is near zero and
    the offsets carry no usable information.  This is reported honestly.

    Bootstrap: 2,000 resamples, default_rng(42), as specified in brief §6.
    """
    reg = oof_df[oof_df["SET_LABEL"] == "REGULAR"].copy()
    y = reg[TARGET].values
    p_a = reg["p_A"].values
    p_b = reg["p_B"].values
    game_ids = reg["GAME_ID"].values

    point_a = compute_metrics(y, p_a)
    point_b = compute_metrics(y, p_b)

    paired = paired_game_bootstrap_brier_logloss(
        game_ids, y, p_a, p_b,
        n_resamples=2000,
        random_state=RNG_SEED,
    )

    b_beats_a_brier = point_b["brier"] < point_a["brier"]
    b_beats_a_ll = point_b["log_loss"] < point_a["log_loss"]
    b_useful = b_beats_a_brier and b_beats_a_ll

    result = {
        "model_A_regular": point_a,
        "model_B_regular": point_b,
        "paired_diff_B_minus_A": paired,
        "b_beats_a_on_brier": bool(b_beats_a_brier),
        "b_beats_a_on_log_loss": bool(b_beats_a_ll),
        "b_carries_usable_information": bool(b_useful),
        "note": (
            "Model B beats Model A on REGULAR held-out shots."
            if b_useful
            else "Model B does not beat Model A on REGULAR held-out shots. "
                 "tau2 is near zero; offsets carry little usable information. "
                 "B is effectively A for this dataset."
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 60)
    print("Phase 3b — Step 3: shooter_offsets.py (Model B)")
    print("=" * 60)

    # Load OOF predictions from Step 2
    print("\n[1] Loading data/processed/oof_predictions.parquet …")
    oof_df = pd.read_parquet(DATA / "oof_predictions.parquet")
    oof_df["GAME_ID"] = oof_df["GAME_ID"].astype(str)
    print(f"    Rows: {len(oof_df):,} | REGULAR: {(oof_df['SET_LABEL']=='REGULAR').sum():,} | "
          f"CLUTCH: {(oof_df['SET_LABEL']=='CLUTCH').sum():,} | "
          f"LATE_BLOWOUT: {(oof_df['SET_LABEL']=='LATE_BLOWOUT').sum():,}")

    # Step 3.1–3.4 — compute per-fold offsets and p_B
    print("\n[2] Computing per-fold DL shooter offsets …")
    oof_df, fold_stats = run_shooter_offsets(oof_df)

    # Step 3.6 — Model B validity check
    print("\n[3] Model B validity check on held-out REGULAR shots …")
    validity = validate_model_b(oof_df)
    ma = validity["model_A_regular"]
    mb = validity["model_B_regular"]
    pd_res = validity["paired_diff_B_minus_A"]
    print(f"    Model A (REGULAR OOF): Brier={ma['brier']:.5f}  LogLoss={ma['log_loss']:.5f}")
    print(f"    Model B (REGULAR OOF): Brier={mb['brier']:.5f}  LogLoss={mb['log_loss']:.5f}")
    print(f"    Paired diff (B-A) Brier : {pd_res['brier']['point_diff']:+.5f}  "
          f"95%CI [{pd_res['brier']['ci_lower']:+.5f}, {pd_res['brier']['ci_upper']:+.5f}]")
    print(f"    Paired diff (B-A) LogLoss: {pd_res['log_loss']['point_diff']:+.5f}  "
          f"95%CI [{pd_res['log_loss']['ci_lower']:+.5f}, {pd_res['log_loss']['ci_upper']:+.5f}]")
    print(f"    B beats A: {validity['b_carries_usable_information']}")
    print(f"    Note: {validity['note']}")

    # Save updated OOF predictions (adds p_B, b_player)
    oof_df.to_parquet(DATA / "oof_predictions.parquet", index=False)
    print(f"\n    OOF predictions updated → data/processed/oof_predictions.parquet")

    # Save offset statistics
    out = {
        "fold_stats": fold_stats,
        "validity_check": validity,
        "documented_limitation": (
            "The out-of-fold residuals used to estimate player offsets come from "
            "models whose hyperparameters were tuned using all folds' games. "
            "This is a second-order effect (tuning never observed individual shot "
            "outcomes directly), so the impact on offset estimates is expected to "
            "be negligible, but it means Model B's uncertainty is slightly understated."
        ),
    }
    with open(DOCS / "shooter_offsets.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"    Shooter offsets saved → docs/shooter_offsets.json")
    print("\n✓ Step 3 complete.")


if __name__ == "__main__":
    main()
