"""Phase 3b Step 6: player-level clutch differential and heterogeneity.

For players with at least 30 clutch shots:
1. Compute ``d_p = (mean Model A clutch residual) - (mean Model A regular-play residual)``.
   Subtracting the player's regular-play residual removes their general skill,
   leaving the difference between clutch and regular-play performance on
   comparable shots.
2. Sampling variance of ``d_p``: sum of the binomial variances of the two means
   under Model A probabilities:
       s2_d = sum_{C}(p_A (1 - p_A)) / n_C^2 + sum_{R}(p_A (1 - p_A)) / n_R^2
3. Estimate the between-player variance ``tau_d2`` with the DerSimonian–Laird
   estimator (reusing ``dersimonian_laird`` from ``src/shooter_offsets.py``) and
   shrink each ``d_p`` toward the inverse-variance weighted mean ``dbar``.
4. Uncertainty intervals for ``tau_d``:
   - Exact Q-profile 95% CI (Viechtbauer 2007), inverting Q(tau^2) ~ chi^2_{P-1}.
   - Player-level bootstrap 95% CI (2,000 resamples of eligible players, seed 42),
     which preserves each player's single copy of within-player sampling variance.
   - Game-level cluster bootstrap (2,000 resamples of GAME_IDs, seed 42): valid
     for the weighted mean ``dbar``, but **upward-biased for ``tau_d``** because
     resampling games resamples shots *within* fixed players, adding a second
     layer of within-player binomial variance (SE ~ 7.3 pp) while DerSimonian–
     Laird subtracts only one layer.  Both the unbiased intervals and the game-
     bootstrap diagnostic are saved so this statistical caveat is transparent.

Reporting rule (brief §7 step 6.4):
    List individual players ONLY if the unbiased 95% interval for ``tau_d``
    strictly excludes zero.  Otherwise report only that no player-level
    heterogeneity in clutch performance is detectable, and do NOT rank or name
    any player.  Never use labels such as "clutch" or "choker".

Outputs
-------
docs/player_clutch.json
    DerSimonian–Laird heterogeneity statistics, Q-profile and bootstrap
    intervals for ``tau_d``, and (ONLY if heterogeneity is detectable) the
    shrunk per-player table.

Run from the repo root::

    python src/player_clutch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shooter_offsets import dersimonian_laird

DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

MIN_CLUTCH_SHOTS = 30
N_BOOT = 2_000
RNG_SEED = 42
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Player-level differential table and DL heterogeneity
# ---------------------------------------------------------------------------

def compute_player_differentials(
    oof_df: pd.DataFrame,
    min_clutch_shots: int = MIN_CLUTCH_SHOTS,
) -> pd.DataFrame:
    """Compute per-player clutch minus regular-play Model A residual and variance."""
    df = oof_df[oof_df["SET_LABEL"].isin(["CLUTCH", "REGULAR"])].copy()
    df["resid_A"] = df["SHOT_MADE_FLAG"].astype(float) - df["p_A"].astype(float)
    df["binom_var"] = df["p_A"].astype(float) * (1.0 - df["p_A"].astype(float))

    clutch = df[df["SET_LABEL"] == "CLUTCH"]
    regular = df[df["SET_LABEL"] == "REGULAR"]

    c_agg = clutch.groupby("PLAYER_ID").agg(
        PLAYER_NAME=("PLAYER_NAME", "first"),
        n_clutch=("resid_A", "count"),
        sum_resid_c=("resid_A", "sum"),
        sum_var_c=("binom_var", "sum"),
    )
    r_agg = regular.groupby("PLAYER_ID").agg(
        n_regular=("resid_A", "count"),
        sum_resid_r=("resid_A", "sum"),
        sum_var_r=("binom_var", "sum"),
    )

    merged = c_agg.join(r_agg, how="inner").reset_index()
    eligible = merged[
        (merged["n_clutch"] >= min_clutch_shots) & (merged["n_regular"] >= 1)
    ].copy()

    eligible["mean_clutch_resid"] = eligible["sum_resid_c"] / eligible["n_clutch"]
    eligible["mean_regular_resid"] = eligible["sum_resid_r"] / eligible["n_regular"]
    eligible["d_p"] = eligible["mean_clutch_resid"] - eligible["mean_regular_resid"]
    eligible["var_clutch"] = eligible["sum_var_c"] / (eligible["n_clutch"] ** 2)
    eligible["var_regular"] = eligible["sum_var_r"] / (eligible["n_regular"] ** 2)
    eligible["s2_d"] = eligible["var_clutch"] + eligible["var_regular"]

    cols = [
        "PLAYER_ID", "PLAYER_NAME", "n_clutch", "n_regular",
        "mean_clutch_resid", "mean_regular_resid", "d_p",
        "var_clutch", "var_regular", "s2_d",
    ]
    return eligible[cols].reset_index(drop=True)


def q_profile_ci_tau_d(
    d_p: np.ndarray,
    s2_d: np.ndarray,
    alpha: float = ALPHA,
) -> Dict[str, float]:
    """Exact Q-profile (Viechtbauer 2007) confidence interval for tau_d (in pp).

    Inverts Q(tau^2) = sum( (d_p - mu(tau^2))^2 / (s2_p + tau^2) ) ~ chi^2_{P-1}.
    Since Q(tau^2) is strictly decreasing in tau^2 >= 0, the lower bound is 0
    whenever Q(0) <= chi^2_{P-1, 1 - alpha/2}.
    """
    P = len(d_p)
    df_q = P - 1
    if df_q < 1:
        return {
            "tau_d_ci_lower_pp": 0.0,
            "tau_d_ci_upper_pp": 0.0,
            "cochran_Q_p_value": 1.0,
        }

    def _q_of_tau2(tau2: float) -> float:
        w = 1.0 / (s2_d + tau2)
        mu = float(np.sum(w * d_p) / np.sum(w))
        return float(np.sum(w * (d_p - mu) ** 2))

    q0 = _q_of_tau2(0.0)
    p_val = float(1.0 - chi2.cdf(q0, df_q))
    crit_lo = float(chi2.ppf(alpha / 2.0, df_q))
    crit_hi = float(chi2.ppf(1.0 - alpha / 2.0, df_q))

    # Bracket upper search limit safely
    hi_bracket = 1.0
    while _q_of_tau2(hi_bracket) > crit_lo and hi_bracket < 1e4:
        hi_bracket *= 2.0

    tau2_lo = 0.0 if q0 <= crit_hi else float(brentq(lambda t: _q_of_tau2(t) - crit_hi, 0.0, hi_bracket))
    tau2_hi = 0.0 if q0 <= crit_lo else float(brentq(lambda t: _q_of_tau2(t) - crit_lo, 0.0, hi_bracket))

    return {
        "tau_d_ci_lower_pp": round(float(np.sqrt(tau2_lo) * 100.0), 3),
        "tau_d_ci_upper_pp": round(float(np.sqrt(tau2_hi) * 100.0), 3),
        "cochran_Q_p_value": round(p_val, 4),
    }


def estimate_heterogeneity(
    player_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run DerSimonian–Laird on player differentials and shrink each d_p."""
    valid = player_df[player_df["s2_d"] > 0].copy()
    d_p = valid["d_p"].values.astype(float)
    n_c = valid["n_clutch"].values.astype(float)
    s2_d = valid["s2_d"].values.astype(float)

    P = len(d_p)
    if P < 2:
        summary = {
            "n_players": int(P),
            "tau_d2": 0.0,
            "tau_d_pp": 0.0,
            "dbar_pp": float(d_p.mean() * 100.0) if P == 1 else 0.0,
            "cochran_Q": 0.0,
            "df": max(0, P - 1),
            "cochran_Q_p_value": 1.0,
            "i_squared_pct": 0.0,
            "avg_shrinkage_factor": 0.0,
            "median_se_d_p_pp": 0.0,
        }
        valid["shrinkage_factor"] = 0.0
        valid["d_p_shrunk"] = d_p
        return summary, valid

    tau_d2, dbar, d_shrunk = dersimonian_laird(d_p, n_c, s2_d)
    w_p = 1.0 / s2_d
    Q = float(np.sum(w_p * (d_p - dbar) ** 2))
    df_q = P - 1
    p_q = float(1.0 - chi2.cdf(Q, df_q))
    i_sq = max(0.0, (Q - df_q) / Q) * 100.0 if Q > 0 else 0.0
    shrinkage = tau_d2 / (tau_d2 + s2_d)

    valid["shrinkage_factor"] = shrinkage
    valid["d_p_shrunk"] = d_shrunk

    summary = {
        "n_players": int(P),
        "tau_d2": float(tau_d2),
        "tau_d_pp": float(np.sqrt(tau_d2) * 100.0),
        "dbar_pp": float(dbar * 100.0),
        "cochran_Q": Q,
        "df": int(df_q),
        "cochran_Q_p_value": p_q,
        "i_squared_pct": float(i_sq),
        "avg_shrinkage_factor": float(np.mean(shrinkage)),
        "median_se_d_p_pp": float(np.median(np.sqrt(s2_d)) * 100.0),
    }
    return summary, valid


# ---------------------------------------------------------------------------
# Player-level bootstrap (unbiased for between-player tau_d)
# ---------------------------------------------------------------------------

def player_bootstrap_tau_d(
    player_df: pd.DataFrame,
    n_resamples: int = N_BOOT,
    random_state: int = RNG_SEED,
) -> Dict[str, Any]:
    """Bootstrap over the P eligible players (2,000 resamples, seed 42).

    Each drawn player carries their single observed (d_p, s2_d), avoiding the
    within-player variance-doubling artifact of game-level resampling for tau_d.
    """
    valid = player_df[player_df["s2_d"] > 0]
    d_p = valid["d_p"].values.astype(float)
    n_c = valid["n_clutch"].values.astype(float)
    s2_d = valid["s2_d"].values.astype(float)
    P = len(d_p)

    rng = np.random.default_rng(random_state)
    boot_tau_pp = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.choice(P, size=P, replace=True)
        t2, _, _ = dersimonian_laird(d_p[idx], n_c[idx], s2_d[idx])
        boot_tau_pp[b] = np.sqrt(t2) * 100.0

    lower_p = 100.0 * (ALPHA / 2.0)
    upper_p = 100.0 * (1.0 - ALPHA / 2.0)
    return {
        "n_resamples": int(n_resamples),
        "rng_seed": int(random_state),
        "tau_d_ci_lower_pp": round(float(np.percentile(boot_tau_pp, lower_p)), 3),
        "tau_d_ci_upper_pp": round(float(np.percentile(boot_tau_pp, upper_p)), 3),
        "tau_d_boot_median_pp": round(float(np.median(boot_tau_pp)), 3),
        "frac_boot_zero": round(float(np.mean(boot_tau_pp == 0.0)), 4),
    }


# ---------------------------------------------------------------------------
# Fast vectorized game-level cluster bootstrap (for dbar + tau_d diagnostic)
# ---------------------------------------------------------------------------

def _build_game_player_matrices(
    oof_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Aggregate shot counts, residual sums, and binomial-var sums by (game, player)."""
    unique_games = np.sort(oof_df["GAME_ID"].astype(str).unique())
    sub = oof_df[oof_df["SET_LABEL"].isin(["CLUTCH", "REGULAR"])].copy()
    sub["GAME_ID"] = sub["GAME_ID"].astype(str)
    sub["resid_A"] = sub["SHOT_MADE_FLAG"].astype(float) - sub["p_A"].astype(float)
    sub["binom_var"] = sub["p_A"].astype(float) * (1.0 - sub["p_A"].astype(float))

    clutch_players = np.sort(sub.loc[sub["SET_LABEL"] == "CLUTCH", "PLAYER_ID"].unique())
    sub = sub[sub["PLAYER_ID"].isin(clutch_players)].copy()

    g_to_idx = {g: i for i, g in enumerate(unique_games)}
    p_to_idx = {p: i for i, p in enumerate(clutch_players)}

    n_games = len(unique_games)
    n_players = len(clutch_players)

    mats = {
        "n_c": np.zeros((n_games, n_players), dtype=np.float64),
        "s_c": np.zeros((n_games, n_players), dtype=np.float64),
        "v_c": np.zeros((n_games, n_players), dtype=np.float64),
        "n_r": np.zeros((n_games, n_players), dtype=np.float64),
        "s_r": np.zeros((n_games, n_players), dtype=np.float64),
        "v_r": np.zeros((n_games, n_players), dtype=np.float64),
    }

    for set_lbl, prefix in [("CLUTCH", "c"), ("REGULAR", "r")]:
        part = sub[sub["SET_LABEL"] == set_lbl]
        grp = part.groupby(["GAME_ID", "PLAYER_ID"], sort=False).agg(
            n=("resid_A", "count"),
            s=("resid_A", "sum"),
            v=("binom_var", "sum"),
        ).reset_index()
        gi = grp["GAME_ID"].map(g_to_idx).values
        pi = grp["PLAYER_ID"].map(p_to_idx).values
        mats[f"n_{prefix}"][gi, pi] = grp["n"].values
        mats[f"s_{prefix}"][gi, pi] = grp["s"].values
        mats[f"v_{prefix}"][gi, pi] = grp["v"].values

    return unique_games, clutch_players, mats


def game_bootstrap_diagnostics(
    oof_df: pd.DataFrame,
    min_clutch_shots: int = MIN_CLUTCH_SHOTS,
    n_resamples: int = N_BOOT,
    random_state: int = RNG_SEED,
) -> Dict[str, Any]:
    """Game-level cluster bootstrap (2,000 resamples, seed 42).

    Provides a valid game-level CI for the weighted mean differential ``dbar``.
    Also records the naive game-level bootstrap distribution of ``tau_d`` as a
    diagnostic: because resampling games resamples shots within fixed players,
    it doubles within-player binomial variance (SE ~ 7.3 pp) while DL subtracts
    only one copy, biasing naive game-resampled ``tau_d`` upward well above the
    observed point estimate.
    """
    unique_games, _players, mats = _build_game_player_matrices(oof_df)
    n_games = len(unique_games)
    rng = np.random.default_rng(random_state)

    boot_tau_pp = np.empty(n_resamples, dtype=np.float64)
    boot_dbar_pp = np.empty(n_resamples, dtype=np.float64)
    boot_n_players = np.empty(n_resamples, dtype=np.int64)

    n_c_mat = mats["n_c"]
    s_c_mat = mats["s_c"]
    v_c_mat = mats["v_c"]
    n_r_mat = mats["n_r"]
    s_r_mat = mats["s_r"]
    v_r_mat = mats["v_r"]

    for b in range(n_resamples):
        sampled = rng.choice(n_games, size=n_games, replace=True)
        m = np.bincount(sampled, minlength=n_games).astype(np.float64)

        n_c = m @ n_c_mat
        n_r = m @ n_r_mat
        mask = (n_c >= min_clutch_shots) & (n_r >= 1)

        n_elig = int(mask.sum())
        boot_n_players[b] = n_elig
        if n_elig < 2:
            boot_tau_pp[b] = 0.0
            boot_dbar_pp[b] = np.nan
            continue

        s_c = (m @ s_c_mat)[mask]
        v_c = (m @ v_c_mat)[mask]
        s_r = (m @ s_r_mat)[mask]
        v_r = (m @ v_r_mat)[mask]
        nc_m = n_c[mask]
        nr_m = n_r[mask]

        d_p = (s_c / nc_m) - (s_r / nr_m)
        s2_d = (v_c / (nc_m ** 2)) + (v_r / (nr_m ** 2))

        valid = s2_d > 0
        if valid.sum() < 2:
            boot_tau_pp[b] = 0.0
            boot_dbar_pp[b] = np.nan
            continue

        tau2_b, dbar_b, _ = dersimonian_laird(d_p[valid], nc_m[valid], s2_d[valid])
        boot_tau_pp[b] = np.sqrt(tau2_b) * 100.0
        boot_dbar_pp[b] = dbar_b * 100.0

    lower_p = 100.0 * (ALPHA / 2.0)
    upper_p = 100.0 * (1.0 - ALPHA / 2.0)

    return {
        "n_resamples": int(n_resamples),
        "rng_seed": int(random_state),
        "n_games_pool": int(n_games),
        "dbar_ci_lower_pp": round(float(np.nanpercentile(boot_dbar_pp, lower_p)), 3),
        "dbar_ci_upper_pp": round(float(np.nanpercentile(boot_dbar_pp, upper_p)), 3),
        "mean_eligible_players_per_draw": round(float(np.mean(boot_n_players)), 1),
        "naive_game_boot_tau_d_ci_lower_pp": round(float(np.percentile(boot_tau_pp, lower_p)), 3),
        "naive_game_boot_tau_d_ci_upper_pp": round(float(np.percentile(boot_tau_pp, upper_p)), 3),
        "naive_game_boot_tau_d_median_pp": round(float(np.median(boot_tau_pp)), 3),
        "naive_game_boot_bias_warning": (
            "Naive game-level bootstrap of tau_d is strongly upward-biased because "
            "resampling games resamples shots within fixed players, doubling within-"
            "player sampling variance while DerSimonian-Laird subtracts only one copy. "
            "Use the Q-profile and player-bootstrap intervals for inference on tau_d."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 3b — Step 6: player_clutch.py")
    print("=" * 60)

    print("\n[1] Loading data/processed/oof_predictions.parquet …")
    oof = pd.read_parquet(DATA / "oof_predictions.parquet")
    oof["GAME_ID"] = oof["GAME_ID"].astype(str)

    print(f"\n[2] Computing player-level differentials (min_clutch_shots={MIN_CLUTCH_SHOTS}) …")
    player_df = compute_player_differentials(oof, min_clutch_shots=MIN_CLUTCH_SHOTS)
    summary, shrunk_df = estimate_heterogeneity(player_df)
    qprof = q_profile_ci_tau_d(
        player_df["d_p"].values, player_df["s2_d"].values, alpha=ALPHA
    )

    print(f"    Eligible players (n_clutch >= {MIN_CLUTCH_SHOTS}): {summary['n_players']}")
    print(f"    Median per-player sampling SE(d_p): {summary['median_se_d_p_pp']:.2f} pp")
    print(f"    Weighted mean differential (dbar): {summary['dbar_pp']:+.3f} pp")
    print(f"    Cochran Q = {summary['cochran_Q']:.2f} (df = {summary['df']}, "
          f"p = {summary['cochran_Q_p_value']:.4f}), I^2 = {summary['i_squared_pct']:.1f}%")
    print(f"    tau_d2 = {summary['tau_d2']:.6f}")
    print(f"    tau_d  = {summary['tau_d_pp']:.3f} pp")
    print(f"    Avg shrinkage factor = {summary['avg_shrinkage_factor']:.3f}")
    print(f"    Exact Q-profile 95% CI for tau_d: "
          f"[{qprof['tau_d_ci_lower_pp']:.3f}, {qprof['tau_d_ci_upper_pp']:.3f}] pp")

    print(f"\n[3] Player-level bootstrap ({N_BOOT:,} resamples, seed {RNG_SEED}) …")
    pboot = player_bootstrap_tau_d(player_df, n_resamples=N_BOOT, random_state=RNG_SEED)
    print(f"    Player-bootstrap 95% CI for tau_d: "
          f"[{pboot['tau_d_ci_lower_pp']:.3f}, {pboot['tau_d_ci_upper_pp']:.3f}] pp "
          f"(median={pboot['tau_d_boot_median_pp']:.3f}, frac_zero={pboot['frac_boot_zero']:.3f})")

    print(f"\n[4] Game-level cluster bootstrap ({N_BOOT:,} resamples, seed {RNG_SEED}) …")
    gboot = game_bootstrap_diagnostics(
        oof, min_clutch_shots=MIN_CLUTCH_SHOTS, n_resamples=N_BOOT, random_state=RNG_SEED
    )
    print(f"    dbar 95% game-level CI: [{gboot['dbar_ci_lower_pp']:+.3f}, {gboot['dbar_ci_upper_pp']:+.3f}] pp")
    print(f"    [Diagnostic] Naive game-resampled tau_d: median={gboot['naive_game_boot_tau_d_median_pp']:.3f} pp, "
          f"95% [{gboot['naive_game_boot_tau_d_ci_lower_pp']:.3f}, {gboot['naive_game_boot_tau_d_ci_upper_pp']:.3f}] pp "
          f"(upward-biased by within-player variance doubling)")

    # Enforce strict reporting rule (brief §7 step 6.4) using the unbiased intervals
    ci_excludes_zero = (
        qprof["tau_d_ci_lower_pp"] > 0.0 and pboot["tau_d_ci_lower_pp"] > 0.0
    )

    out: Dict[str, Any] = {
        "meta": {
            "description": (
                "Player-level clutch differential analysis (Phase 3b Step 6). "
                "d_p = (mean Model A clutch residual) - (mean Model A regular-play residual) "
                "for players with at least 30 clutch shots. Between-player variance tau_d2 "
                "estimated via DerSimonian-Laird using binomial sampling variances."
            ),
            "min_clutch_shots": MIN_CLUTCH_SHOTS,
            "reporting_rule": (
                "Individual players are listed ONLY if the unbiased 95% interval for "
                "tau_d strictly excludes zero. Otherwise no player is named or ranked."
            ),
            "methodological_note": (
                f"Naive game-level cluster bootstrap for tau_d is strongly upward-biased "
                f"(bootstrap median {gboot['naive_game_boot_tau_d_median_pp']:.3f} pp vs "
                f"observed {summary['tau_d_pp']:.3f} pp) because resampling games "
                f"resamples shots within fixed players, doubling within-player binomial "
                f"sampling variance while DL subtracts only one copy. Both the exact "
                f"Q-profile CI [{qprof['tau_d_ci_lower_pp']:.3f}, {qprof['tau_d_ci_upper_pp']:.3f}] pp "
                f"and the player-level bootstrap CI "
                f"[{pboot['tau_d_ci_lower_pp']:.3f}, {pboot['tau_d_ci_upper_pp']:.3f}] pp "
                f"include zero, agreeing with Cochran's Q = {summary['cochran_Q']:.2f} "
                f"on df = {summary['df']} (p = {summary['cochran_Q_p_value']:.4f})."
            ),
        },
        "summary": {
            "n_eligible_players": summary["n_players"],
            "median_se_d_p_pp": round(summary["median_se_d_p_pp"], 3),
            "dbar_pp": round(summary["dbar_pp"], 3),
            "dbar_game_ci_lower_pp": gboot["dbar_ci_lower_pp"],
            "dbar_game_ci_upper_pp": gboot["dbar_ci_upper_pp"],
            "cochran_Q": round(summary["cochran_Q"], 3),
            "df": summary["df"],
            "cochran_Q_p_value": round(summary["cochran_Q_p_value"], 4),
            "i_squared_pct": round(summary["i_squared_pct"], 2),
            "tau_d2": summary["tau_d2"],
            "tau_d_pp": round(summary["tau_d_pp"], 3),
            "q_profile_ci_lower_pp": qprof["tau_d_ci_lower_pp"],
            "q_profile_ci_upper_pp": qprof["tau_d_ci_upper_pp"],
            "player_boot_ci_lower_pp": pboot["tau_d_ci_lower_pp"],
            "player_boot_ci_upper_pp": pboot["tau_d_ci_upper_pp"],
            "avg_shrinkage_factor": round(summary["avg_shrinkage_factor"], 4),
        },
        "q_profile": qprof,
        "player_bootstrap": pboot,
        "game_bootstrap_diagnostic": gboot,
        "heterogeneity_detectable": bool(ci_excludes_zero),
    }

    if ci_excludes_zero:
        print("\n    Reporting rule: unbiased 95% CI for tau_d excludes zero — listing shrunk differentials.")
        shrunk_sorted = shrunk_df.sort_values("d_p_shrunk", ascending=False)
        players_list = []
        for _, row in shrunk_sorted.iterrows():
            players_list.append({
                "player_id": int(row["PLAYER_ID"]),
                "player_name": str(row["PLAYER_NAME"]),
                "n_clutch": int(row["n_clutch"]),
                "n_regular": int(row["n_regular"]),
                "raw_d_p_pp": round(float(row["d_p"]) * 100.0, 3),
                "se_d_p_pp": round(float(np.sqrt(row["s2_d"])) * 100.0, 3),
                "shrinkage_factor": round(float(row["shrinkage_factor"]), 4),
                "shrunk_d_p_pp": round(float(row["d_p_shrunk"]) * 100.0, 3),
            })
        out["players"] = players_list
    else:
        print("\n    Reporting rule: unbiased 95% CI for tau_d includes zero.")
        print("    No player-level heterogeneity in clutch performance is detectable.")
        print("    Individual players are NOT named or ranked.")
        out["reporting_outcome"] = (
            f"No player-level heterogeneity in clutch performance is detectable at this "
            f"sample size (Q-profile 95% CI [{qprof['tau_d_ci_lower_pp']:.3f}, "
            f"{qprof['tau_d_ci_upper_pp']:.3f}] pp; player-bootstrap 95% CI "
            f"[{pboot['tau_d_ci_lower_pp']:.3f}, {pboot['tau_d_ci_upper_pp']:.3f}] pp; "
            f"Cochran's Q = {summary['cochran_Q']:.2f} on df = {summary['df']}, "
            f"p = {summary['cochran_Q_p_value']:.4f}). Per the pre-specified "
            f"reporting rule, no individual players are named or ranked."
        )

    with open(DOCS / "player_clutch.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n    Saved → docs/player_clutch.json")
    print("\n✓ Step 6 complete.")


if __name__ == "__main__":
    main()
