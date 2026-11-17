"""Phase 3b Step 4 & 5: clutch residual and exploratory cuts.

Computes all pre-specified results from §6 of the brief using the saved
out-of-fold predictions table (oof_predictions.parquet).  Clutch outcomes
never influence Model A or Model B — the OOF table was produced before this
script runs, and this script only reads predictions to compute residuals.

Outputs
-------
docs/clutch_effect.json
    Full results: primary (Model B overall clutch residual), secondary
    pre-specified results, and exploratory cuts under an ``exploratory`` key.
docs/clutch_effect_overview.png
    Forest plot of clutch residuals (overall, 2PT, 3PT) and the
    late-blowout control, for both models.
docs/residual_by_period.png
    REGULAR out-of-fold residual by period under Model A and Model B,
    with 95% bootstrap CIs.

Run from the repo root::

    python src/clutch_effect.py

Hard rules enforced here (Phase 3b brief §3)
--------------------------------------------
- All bootstrap resampling uses game-level clusters, 2,000 resamples,
  ``numpy.random.default_rng(42)``.
- No model re-fits — only pre-computed OOF predictions are used.
- Player offsets (b_player) are treated as fixed (slightly understates
  Model B uncertainty; documented in shooter_offsets.json).
- No causal language in any output key or comment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

N_BOOT = 2_000
RNG_SEED = 42
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _build_game_index(game_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (unique_games, {game_id -> row_indices}) for fast bootstrap."""
    unique_games = np.unique(game_ids)
    df = pd.DataFrame({"game": game_ids, "idx": np.arange(len(game_ids))})
    game_groups: Dict[str, np.ndarray] = (
        df.groupby("game")["idx"].apply(np.array).to_dict()
    )
    return unique_games, game_groups


def bootstrap_residual(
    game_ids: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    n_resamples: int = N_BOOT,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute observed FG%, expected FG%, residual (pp) with 95% game-level CI.

    The residual is ``mean(y - p) * 100`` in percentage points.  The 95%
    percentile interval is from 2,000 game-cluster bootstrap resamples.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    n_shots = int(len(y))
    unique_games, game_groups = _build_game_index(game_ids)
    n_games = int(len(unique_games))

    obs_fg = float(y.mean()) * 100.0
    exp_fg = float(p.mean()) * 100.0
    point_resid = obs_fg - exp_fg

    boot_resids = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        sampled = rng.choice(unique_games, size=n_games, replace=True)
        idx = np.concatenate([game_groups[g] for g in sampled])
        boot_resids[b] = (y[idx].mean() - p[idx].mean()) * 100.0

    lower = float(np.percentile(boot_resids, 100 * ALPHA / 2))
    upper = float(np.percentile(boot_resids, 100 * (1 - ALPHA / 2)))

    return {
        "n_shots": n_shots,
        "n_games": n_games,
        "obs_fg_pct": round(obs_fg, 3),
        "exp_fg_pct": round(exp_fg, 3),
        "residual_pp": round(point_resid, 3),
        "ci_lower_pp": round(lower, 3),
        "ci_upper_pp": round(upper, 3),
    }


def _residual_row(
    label: str,
    df: pd.DataFrame,
    model_col: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Compute a single residual row for a subset of `df`."""
    game_ids = df["GAME_ID"].values.astype(str)
    y = df["SHOT_MADE_FLAG"].values.astype(float)
    p = df[model_col].values.astype(float)

    n_shots = len(y)
    unique_games, game_groups = _build_game_index(game_ids)
    n_games = int(len(unique_games))

    obs_fg = float(y.mean()) * 100.0
    exp_fg = float(p.mean()) * 100.0
    point_resid = obs_fg - exp_fg

    boot_resids = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        sampled = rng.choice(unique_games, size=n_games, replace=True)
        idx = np.concatenate([game_groups[g] for g in sampled])
        boot_resids[b] = (y[idx].mean() - p[idx].mean()) * 100.0

    lower = float(np.percentile(boot_resids, 100 * ALPHA / 2))
    upper = float(np.percentile(boot_resids, 100 * (1 - ALPHA / 2)))

    return {
        "label": label,
        "n_shots": n_shots,
        "n_games": n_games,
        "obs_fg_pct": round(obs_fg, 3),
        "exp_fg_pct": round(exp_fg, 3),
        "residual_pp": round(point_resid, 3),
        "ci_lower_pp": round(lower, 3),
        "ci_upper_pp": round(upper, 3),
    }


def _bootstrap_difference(
    rng: np.random.Generator,
    df_all: pd.DataFrame,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    model_col: str,
    n_resamples: int = N_BOOT,
) -> Dict[str, Any]:
    """Game-level cluster bootstrap CI for difference of residuals (mask_a minus mask_b).

    Resamples whole GAME_IDs with replacement from all games in ``df_all``
    (preserving the overlap of games that contain both CLUTCH and LATE_BLOWOUT
    shots).  Vectorized via per-game residual sums and shot counts.
    """
    game_ids = df_all["GAME_ID"].values.astype(str)
    resid = df_all["SHOT_MADE_FLAG"].values.astype(float) - df_all[model_col].values.astype(float)

    df_game = pd.DataFrame({
        "game": game_ids,
        "sum_a": np.where(mask_a, resid, 0.0),
        "n_a": mask_a.astype(np.int64),
        "sum_b": np.where(mask_b, resid, 0.0),
        "n_b": mask_b.astype(np.int64),
    })
    agg = df_game.groupby("game", sort=True).sum()

    sum_a = agg["sum_a"].values
    n_a = agg["n_a"].values
    sum_b = agg["sum_b"].values
    n_b = agg["n_b"].values
    n_games = len(agg)

    point_a = (sum_a.sum() / n_a.sum()) * 100.0
    point_b = (sum_b.sum() / n_b.sum()) * 100.0
    point_diff = point_a - point_b

    sampled = rng.choice(n_games, size=(n_resamples, n_games), replace=True)
    tot_n_a = n_a[sampled].sum(axis=1)
    tot_n_b = n_b[sampled].sum(axis=1)
    boot_a = (sum_a[sampled].sum(axis=1) / tot_n_a) * 100.0
    boot_b = (sum_b[sampled].sum(axis=1) / tot_n_b) * 100.0
    boot_diffs = boot_a - boot_b

    lower = float(np.percentile(boot_diffs, 100 * ALPHA / 2))
    upper = float(np.percentile(boot_diffs, 100 * (1 - ALPHA / 2)))

    return {
        "n_shots_clutch": int(n_a.sum()),
        "n_shots_late_blowout": int(n_b.sum()),
        "n_games_pool": int(n_games),
        "point_diff_pp": round(float(point_diff), 3),
        "ci_lower_pp": round(lower, 3),
        "ci_upper_pp": round(upper, 3),
    }


# ---------------------------------------------------------------------------
# Descriptive table helper (now includes mean b_player)
# ---------------------------------------------------------------------------

def _desc_row(label: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Return a descriptive row: n_shots, obs_fg_pct, mean_p_A, mean_p_B, mean_b_player."""
    return {
        "label": label,
        "n_shots": int(len(df)),
        "obs_fg_pct": round(float(df["SHOT_MADE_FLAG"].mean()) * 100, 3),
        "mean_p_A": round(float(df["p_A"].mean()) * 100, 3),
        "mean_p_B": round(float(df["p_B"].mean()) * 100, 3),
        "mean_b_player_pp": round(float(df["b_player"].mean()) * 100, 3),
    }


def build_descriptive_table(oof: pd.DataFrame) -> List[Dict[str, Any]]:
    """For each set × SHOT_VALUE combination: n_shots, obs FG%, mean p_A, mean p_B, mean b_player."""
    rows = []
    for sl in ["REGULAR", "LATE_BLOWOUT", "CLUTCH"]:
        for sv in [2, 3]:
            sub = oof[(oof["SET_LABEL"] == sl) & (oof["SHOT_VALUE"] == sv)]
            rows.append(_desc_row(f"{sl} {sv}PT", sub))
    return rows


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def plot_forest(
    rows_A: List[Dict[str, Any]],
    rows_B: List[Dict[str, Any]],
    save_path: Path,
) -> None:
    """Draw a horizontal forest plot of residuals (pp) with 95% CI."""
    labels = [r["label"] for r in rows_A]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.7 * n))
    y_positions = np.arange(n)

    for i, (ra, rb) in enumerate(zip(rows_A, rows_B)):
        y_a = y_positions[i] + 0.15
        y_b = y_positions[i] - 0.15
        ax.plot(ra["residual_pp"], y_a, "o", color="#1f77b4", markersize=7, zorder=4)
        ax.hlines(y_a, ra["ci_lower_pp"], ra["ci_upper_pp"],
                  color="#1f77b4", linewidth=2, zorder=3)
        ax.plot(rb["residual_pp"], y_b, "s", color="#ff7f0e", markersize=7, zorder=4)
        ax.hlines(y_b, rb["ci_lower_pp"], rb["ci_upper_pp"],
                  color="#ff7f0e", linewidth=2, zorder=3)

    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Residual (observed − expected FG%, pp)", fontsize=11)
    ax.set_title(
        "Clutch residual and late-blowout control — Model A vs Model B\n"
        "(positive = observed > expected; baseline fitted on regular-play shots only)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#1f77b4", marker="o", linewidth=2, markersize=7,
               label="Model A (shot quality)"),
        Line2D([0], [0], color="#ff7f0e", marker="s", linewidth=2, markersize=7,
               label="Model B (shot + shooter skill)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Forest plot saved → {save_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Period residual plot (Model A and Model B)
# ---------------------------------------------------------------------------

def plot_residual_by_period(
    period_rows_A: List[Dict[str, Any]],
    period_rows_B: List[Dict[str, Any]],
    save_path: Path,
) -> None:
    """Bar chart of REGULAR OOF residual by period with 95% CI error bars."""
    labels = [r["label"] for r in period_rows_A]
    x = np.arange(len(labels))
    width = 0.35

    residuals_A = np.array([r["residual_pp"] for r in period_rows_A])
    err_low_A = residuals_A - np.array([r["ci_lower_pp"] for r in period_rows_A])
    err_high_A = np.array([r["ci_upper_pp"] for r in period_rows_A]) - residuals_A

    residuals_B = np.array([r["residual_pp"] for r in period_rows_B])
    err_low_B = residuals_B - np.array([r["ci_lower_pp"] for r in period_rows_B])
    err_high_B = np.array([r["ci_upper_pp"] for r in period_rows_B]) - residuals_B

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, residuals_A, width,
           yerr=[err_low_A, err_high_A], capsize=5, color="#1f77b4",
           alpha=0.8, label="Model A", error_kw={"linewidth": 1.5})
    ax.bar(x + width / 2, residuals_B, width,
           yerr=[err_low_B, err_high_B], capsize=5, color="#ff7f0e",
           alpha=0.8, label="Model B", error_kw={"linewidth": 1.5})

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Residual (observed − expected FG%, pp)", fontsize=11)
    ax.set_title(
        "REGULAR out-of-fold residual by period — Model A and Model B\n"
        "(Q4 = clock > 5:00 only; last 5 min of Q4 and OT excluded)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Period residual plot saved → {save_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 60)
    print("Phase 3b — Steps 4 & 5: clutch_effect.py")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load OOF predictions
    # ------------------------------------------------------------------
    print("\n[1] Loading OOF predictions …")
    oof = pd.read_parquet(DATA / "oof_predictions.parquet")
    oof["GAME_ID"] = oof["GAME_ID"].astype(str)
    print(f"    Rows: {len(oof):,}")
    print(f"    SET_LABEL counts: {oof['SET_LABEL'].value_counts().to_dict()}")

    if "p_B" not in oof.columns:
        raise RuntimeError(
            "STOP: oof_predictions.parquet is missing column 'p_B'. "
            "Run src/baseline_model.py (which calls the shooter-offset step) first."
        )

    reg = oof[oof["SET_LABEL"] == "REGULAR"].copy()
    clutch = oof[oof["SET_LABEL"] == "CLUTCH"].copy()
    lb = oof[oof["SET_LABEL"] == "LATE_BLOWOUT"].copy()
    print(f"    REGULAR: {len(reg):,}  CLUTCH: {len(clutch):,}  LATE_BLOWOUT: {len(lb):,}")

    rng = np.random.default_rng(RNG_SEED)

    # ------------------------------------------------------------------
    # Descriptive table (with b_player)
    # ------------------------------------------------------------------
    print("\n[2] Descriptive table (n, obs FG%, mean p_A, mean p_B, mean b_player) …")
    desc_table = build_descriptive_table(oof)
    print(f"\n    {'Label':<22} {'n':>8} {'obs FG%':>9} {'mean p_A':>10} "
          f"{'mean p_B':>10} {'b_player':>10}")
    print("    " + "-" * 72)
    for row in desc_table:
        print(
            f"    {row['label']:<22} {row['n_shots']:>8,} "
            f"{row['obs_fg_pct']:>9.2f} {row['mean_p_A']:>10.2f} "
            f"{row['mean_p_B']:>10.2f} {row['mean_b_player_pp']:>+10.3f}"
        )

    # ------------------------------------------------------------------
    # Primary result: overall clutch residual
    # ------------------------------------------------------------------
    print("\n[3] Primary result: overall clutch residual (Model A and B) …")
    clutch_A_overall = _residual_row("CLUTCH overall (Model A)", clutch, "p_A", rng)
    clutch_B_overall = _residual_row("CLUTCH overall (Model B)", clutch, "p_B", rng)
    print(f"    Model A: {clutch_A_overall['residual_pp']:+.3f} pp  "
          f"95% CI [{clutch_A_overall['ci_lower_pp']:+.3f}, {clutch_A_overall['ci_upper_pp']:+.3f}]  "
          f"n={clutch_A_overall['n_shots']:,}")
    print(f"    Model B: {clutch_B_overall['residual_pp']:+.3f} pp  "
          f"95% CI [{clutch_B_overall['ci_lower_pp']:+.3f}, {clutch_B_overall['ci_upper_pp']:+.3f}]  "
          f"n={clutch_B_overall['n_shots']:,}")

    # ------------------------------------------------------------------
    # By SHOT_VALUE
    # ------------------------------------------------------------------
    print("\n[4] Clutch residual by SHOT_VALUE …")
    clutch_2pt = clutch[clutch["SHOT_VALUE"] == 2]
    clutch_3pt = clutch[clutch["SHOT_VALUE"] == 3]
    clutch_A_2pt = _residual_row("CLUTCH 2PT (Model A)", clutch_2pt, "p_A", rng)
    clutch_B_2pt = _residual_row("CLUTCH 2PT (Model B)", clutch_2pt, "p_B", rng)
    clutch_A_3pt = _residual_row("CLUTCH 3PT (Model A)", clutch_3pt, "p_A", rng)
    clutch_B_3pt = _residual_row("CLUTCH 3PT (Model B)", clutch_3pt, "p_B", rng)
    for row in [clutch_A_2pt, clutch_B_2pt, clutch_A_3pt, clutch_B_3pt]:
        print(f"    {row['label']}: {row['residual_pp']:+.3f} pp  "
              f"95% CI [{row['ci_lower_pp']:+.3f}, {row['ci_upper_pp']:+.3f}]  n={row['n_shots']:,}")

    # ------------------------------------------------------------------
    # Late-blowout control
    # ------------------------------------------------------------------
    print("\n[5] Late-blowout control residual …")
    lb_A_overall = _residual_row("LATE_BLOWOUT overall (Model A)", lb, "p_A", rng)
    lb_B_overall = _residual_row("LATE_BLOWOUT overall (Model B)", lb, "p_B", rng)
    lb_A_2pt = _residual_row("LATE_BLOWOUT 2PT (Model A)", lb[lb["SHOT_VALUE"] == 2], "p_A", rng)
    lb_B_2pt = _residual_row("LATE_BLOWOUT 2PT (Model B)", lb[lb["SHOT_VALUE"] == 2], "p_B", rng)
    lb_A_3pt = _residual_row("LATE_BLOWOUT 3PT (Model A)", lb[lb["SHOT_VALUE"] == 3], "p_A", rng)
    lb_B_3pt = _residual_row("LATE_BLOWOUT 3PT (Model B)", lb[lb["SHOT_VALUE"] == 3], "p_B", rng)
    for row in [lb_A_overall, lb_B_overall, lb_A_2pt, lb_B_2pt, lb_A_3pt, lb_B_3pt]:
        print(f"    {row['label']}: {row['residual_pp']:+.3f} pp  "
              f"95% CI [{row['ci_lower_pp']:+.3f}, {row['ci_upper_pp']:+.3f}]  n={row['n_shots']:,}")

    # ------------------------------------------------------------------
    # CLUTCH minus LATE_BLOWOUT difference (joint game-level cluster bootstrap)
    # ------------------------------------------------------------------
    print("\n[5b] CLUTCH minus LATE_BLOWOUT difference (joint game-level bootstrap, seed 42) …")
    diff_results: Dict[str, Dict[str, Any]] = {}
    diff_rng = np.random.default_rng(RNG_SEED)
    is_clutch = (oof["SET_LABEL"] == "CLUTCH").values
    is_lb = (oof["SET_LABEL"] == "LATE_BLOWOUT").values

    for model_col, model_label in [("p_A", "Model A"), ("p_B", "Model B")]:
        for sv_label, sv_mask in [
            ("overall", np.ones(len(oof), dtype=bool)),
            ("2PT", (oof["SHOT_VALUE"] == 2).values),
            ("3PT", (oof["SHOT_VALUE"] == 3).values),
        ]:
            key = f"{sv_label}_{model_label}"
            d = _bootstrap_difference(
                diff_rng,
                oof,
                is_clutch & sv_mask,
                is_lb & sv_mask,
                model_col,
            )
            diff_results[key] = d
            print(f"    {key}: diff = {d['point_diff_pp']:+.3f} pp  "
                  f"95% CI [{d['ci_lower_pp']:+.3f}, {d['ci_upper_pp']:+.3f}]")

    # ------------------------------------------------------------------
    # REGULAR residual by period (Model A and B)
    # ------------------------------------------------------------------
    print("\n[6] REGULAR OOF residual by period — Model A and Model B …")
    period_rows_A: List[Dict[str, Any]] = []
    period_rows_B: List[Dict[str, Any]] = []
    for p_val in sorted(reg["PERIOD"].unique()):
        sub = reg[reg["PERIOD"] == p_val]
        label = f"Q{p_val}" if p_val <= 4 else f"OT{p_val - 4}"
        row_A = _residual_row(label, sub, "p_A", rng)
        row_B = _residual_row(label, sub, "p_B", rng)
        period_rows_A.append(row_A)
        period_rows_B.append(row_B)
        print(
            f"    {label}: Model A {row_A['residual_pp']:+.3f} pp [{row_A['ci_lower_pp']:+.3f}, {row_A['ci_upper_pp']:+.3f}]  "
            f"Model B {row_B['residual_pp']:+.3f} pp [{row_B['ci_lower_pp']:+.3f}, {row_B['ci_upper_pp']:+.3f}]  "
            f"n={row_A['n_shots']:,}"
        )

    # ------------------------------------------------------------------
    # REGULAR residual by SHOT_VALUE (Model A and B)
    # ------------------------------------------------------------------
    print("\n[7] REGULAR OOF residual by SHOT_VALUE — Model A and Model B …")
    reg_sv_rows_A: List[Dict[str, Any]] = []
    reg_sv_rows_B: List[Dict[str, Any]] = []
    for sv in [2, 3]:
        sub = reg[reg["SHOT_VALUE"] == sv]
        row_A = _residual_row(f"{sv}PT", sub, "p_A", rng)
        row_B = _residual_row(f"{sv}PT", sub, "p_B", rng)
        reg_sv_rows_A.append(row_A)
        reg_sv_rows_B.append(row_B)
        print(
            f"    {sv}PT: Model A {row_A['residual_pp']:+.3f} pp [{row_A['ci_lower_pp']:+.3f}, {row_A['ci_upper_pp']:+.3f}]  "
            f"Model B {row_B['residual_pp']:+.3f} pp [{row_B['ci_lower_pp']:+.3f}, {row_B['ci_upper_pp']:+.3f}]  "
            f"n={row_A['n_shots']:,}"
        )

    # ------------------------------------------------------------------
    # Player-level heterogeneity (SD of clutch differential, Model A)
    # ------------------------------------------------------------------
    print("\n[8] Player-level heterogeneity (SD of clutch differential, Model A) …")
    clutch_player_A = (
        clutch.groupby("PLAYER_ID")
        .agg(
            mean_clutch_resid_A=("SHOT_MADE_FLAG", "mean"),
            mean_p_A=("p_A", "mean"),
            n_clutch=("SHOT_MADE_FLAG", "count"),
        )
        .reset_index()
    )
    clutch_player_A["diff_A"] = (
        clutch_player_A["mean_clutch_resid_A"] - clutch_player_A["mean_p_A"]
    )
    n_arr = clutch_player_A["n_clutch"].values.astype(float)
    d_arr = clutch_player_A["diff_A"].values.astype(float)
    w = n_arr / n_arr.sum()
    wmean = float(np.sum(w * d_arr))
    wsd = float(np.sqrt(np.sum(w * (d_arr - wmean) ** 2)))
    print(f"    n_players: {len(clutch_player_A):,}")
    print(f"    Weighted mean clutch diff (Model A): {wmean * 100:+.3f} pp")
    print(f"    Weighted SD of clutch diff (Model A): {wsd * 100:.3f} pp")

    # Bootstrap SD
    unique_games_c, game_groups_c = _build_game_index(clutch["GAME_ID"].values.astype(str))
    n_games_c = len(unique_games_c)
    boot_sd: List[float] = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique_games_c, size=n_games_c, replace=True)
        idx = np.concatenate([game_groups_c[g] for g in sampled])
        sub_b = clutch.iloc[idx]
        agg = sub_b.groupby("PLAYER_ID").agg(
            mean_resid=("SHOT_MADE_FLAG", "mean"),
            mean_p=("p_A", "mean"),
            n=("SHOT_MADE_FLAG", "count"),
        )
        agg["diff"] = agg["mean_resid"] - agg["mean_p"]
        n_b = agg["n"].values.astype(float)
        d_b = agg["diff"].values.astype(float)
        w_b = n_b / n_b.sum()
        wmean_b = float(np.sum(w_b * d_b))
        boot_sd.append(float(np.sqrt(np.sum(w_b * (d_b - wmean_b) ** 2))) * 100.0)
    sd_ci_lo = float(np.percentile(boot_sd, 100 * ALPHA / 2))
    sd_ci_hi = float(np.percentile(boot_sd, 100 * (1 - ALPHA / 2)))
    print(f"    SD 95% CI: [{sd_ci_lo:.3f}, {sd_ci_hi:.3f}] pp")

    heterogeneity = {
        "n_players": int(len(clutch_player_A)),
        "weighted_mean_diff_pp": round(wmean * 100, 3),
        "weighted_sd_diff_pp": round(wsd * 100, 3),
        "sd_ci_lower_pp": round(sd_ci_lo, 3),
        "sd_ci_upper_pp": round(sd_ci_hi, 3),
        "note": (
            "Full player-level analysis (Step 6) will apply DerSimonian-Laird "
            "shrinkage and report individual players only if tau_d CI excludes zero."
        ),
    }

    # ------------------------------------------------------------------
    # Step 5: Exploratory cuts (must be labelled exploratory)
    # ------------------------------------------------------------------
    print("\n[9] Exploratory: clutch residual by time bucket …")
    all_shots = pd.read_parquet(DATA / "all_shots.parquet")
    all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(str)
    time_cols = all_shots[["GAME_ID", "GAME_EVENT_ID", "TIME_REMAINING_SEC", "IS_HOME"]].copy()
    clutch_ext = clutch.merge(time_cols, on=["GAME_ID", "GAME_EVENT_ID"], how="left")

    time_buckets = [
        ("last_60s", clutch_ext["TIME_REMAINING_SEC"] < 60),
        ("60_180s", (clutch_ext["TIME_REMAINING_SEC"] >= 60) & (clutch_ext["TIME_REMAINING_SEC"] < 180)),
        ("180_300s", (clutch_ext["TIME_REMAINING_SEC"] >= 180) & (clutch_ext["TIME_REMAINING_SEC"] <= 300)),
    ]
    exp_time_A: List[Dict[str, Any]] = []
    exp_time_B: List[Dict[str, Any]] = []
    for label, mask in time_buckets:
        sub = clutch_ext[mask]
        exp_time_A.append(_residual_row(label, sub, "p_A", rng))
        exp_time_B.append(_residual_row(label, sub, "p_B", rng))
        print(
            f"    {label}: Model A {exp_time_A[-1]['residual_pp']:+.3f} pp "
            f"[{exp_time_A[-1]['ci_lower_pp']:+.3f}, {exp_time_A[-1]['ci_upper_pp']:+.3f}]  "
            f"Model B {exp_time_B[-1]['residual_pp']:+.3f} pp "
            f"[{exp_time_B[-1]['ci_lower_pp']:+.3f}, {exp_time_B[-1]['ci_upper_pp']:+.3f}]  "
            f"n={len(sub):,}"
        )

    print("\n[10] Exploratory: clutch residual home vs away …")
    exp_home_A: List[Dict[str, Any]] = []
    exp_home_B: List[Dict[str, Any]] = []
    for label, mask in [
        ("home", clutch_ext["IS_HOME"] == 1),
        ("away", clutch_ext["IS_HOME"] == 0),
    ]:
        sub = clutch_ext[mask]
        exp_home_A.append(_residual_row(label, sub, "p_A", rng))
        exp_home_B.append(_residual_row(label, sub, "p_B", rng))
        print(
            f"    {label}: Model A {exp_home_A[-1]['residual_pp']:+.3f} pp "
            f"[{exp_home_A[-1]['ci_lower_pp']:+.3f}, {exp_home_A[-1]['ci_upper_pp']:+.3f}]  "
            f"Model B {exp_home_B[-1]['residual_pp']:+.3f} pp "
            f"[{exp_home_B[-1]['ci_lower_pp']:+.3f}, {exp_home_B[-1]['ci_upper_pp']:+.3f}]  "
            f"n={len(sub):,}"
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[11] Generating forest plot …")
    forest_labels = [
        "CLUTCH overall",
        "CLUTCH 2PT",
        "CLUTCH 3PT",
        "LATE_BLOWOUT overall",
    ]
    forest_A_data = [clutch_A_overall, clutch_A_2pt, clutch_A_3pt, lb_A_overall]
    forest_B_data = [clutch_B_overall, clutch_B_2pt, clutch_B_3pt, lb_B_overall]
    for d, label in zip(forest_A_data + forest_B_data, forest_labels * 2):
        d["label"] = label
    plot_forest(forest_A_data, forest_B_data, DOCS / "clutch_effect_overview.png")

    print("\n[12] Generating residual-by-period plot …")
    plot_residual_by_period(period_rows_A, period_rows_B, DOCS / "residual_by_period.png")

    # ------------------------------------------------------------------
    # Assemble JSON
    # ------------------------------------------------------------------
    print("\n[13] Writing docs/clutch_effect.json …")

    results: Dict[str, Any] = {
        "meta": {
            "estimand": (
                "Clutch residual: observed FG% minus expected FG% from a shot-quality "
                "model fitted only on regular-play shots, in percentage points. "
                "Positive = observed > expected; negative = observed < expected. "
                "This is a descriptive quantity; it bundles defensive intensity, "
                "fatigue, shot selection within type, shooter composition, and any "
                "other difference between clutch and regular play."
            ),
            "baseline_fitted_on": "REGULAR shots only (196,198 shots)",
            "clutch_outcomes_in_baseline": False,
            "n_bootstrap": N_BOOT,
            "bootstrap_method": "game-level cluster bootstrap (resample GAME_IDs), 95% percentile CI",
            "rng_seed": RNG_SEED,
            "precision_note": (
                "Binomial standard error about 0.6 pp for the overall clutch residual "
                "(7,104 shots), about 0.8 pp for 2PT (4,246 shots), about 0.9 pp for "
                "3PT (2,858 shots). The 95% interval half-width is about 1 pp overall."
            ),
            "model_B_uncertainty_note": (
                "Player offsets (b_player) are treated as fixed in the bootstrap. "
                "This slightly understates Model B uncertainty."
            ),
        },
        "descriptive_table": desc_table,
        "primary_result": {
            "description": "Overall clutch residual under Model B (primary estimand)",
            "model_B": clutch_B_overall,
            "model_A": clutch_A_overall,
        },
        "secondary": {
            "clutch_by_shot_value": {
                "model_A": [clutch_A_2pt, clutch_A_3pt],
                "model_B": [clutch_B_2pt, clutch_B_3pt],
            },
            "late_blowout_control": {
                "description": (
                    "If clutch and late-blowout residuals look alike, the effect is "
                    "more consistent with a 'late in the game' effect than a 'close game' effect."
                ),
                "model_A": {
                    "overall": lb_A_overall,
                    "by_shot_value": [lb_A_2pt, lb_A_3pt],
                },
                "model_B": {
                    "overall": lb_B_overall,
                    "by_shot_value": [lb_B_2pt, lb_B_3pt],
                },
            },
            "clutch_minus_late_blowout_difference": {
                "description": (
                    "Game-level cluster bootstrap CI (2,000 resamples, seed 42, "
                    "resampling whole GAME_IDs over all 1,230 games to preserve "
                    "the 437-game overlap) for CLUTCH residual minus LATE_BLOWOUT "
                    "residual. Negative means CLUTCH residual is more negative "
                    "than LATE_BLOWOUT."
                ),
                "results": diff_results,
            },
            "regular_residual_by_period": {
                "description": (
                    "REGULAR out-of-fold residual by period (Q1–Q4). "
                    "Period 4 contains only the first 7 min of Q4 (clock > 5:00; "
                    "last 5 min of Q4 and OT are excluded). "
                    "The alternating sign pattern (+Q1, −Q2, +Q3, −Q4) is unexplained: "
                    "Model B moves Q1 and Q3 toward zero, but Q2 moves from −0.44 to "
                    "−0.57 pp and Q4 stays at −0.60 pp."
                ),
                "model_A": period_rows_A,
                "model_B": period_rows_B,
            },
            "regular_residual_by_shot_value": {
                "description": (
                    "REGULAR out-of-fold residual by SHOT_VALUE under Model A and Model B."
                ),
                "model_A": reg_sv_rows_A,
                "model_B": reg_sv_rows_B,
            },
            "player_heterogeneity": heterogeneity,
        },
        "exploratory": {
            "warning": (
                "The following results are exploratory and were not pre-specified. "
                "Do not use significance language for these cuts."
            ),
            "clutch_by_time_bucket": {
                "description": "Clutch residual by time remaining in the clutch window (<=300 s).",
                "model_A": exp_time_A,
                "model_B": exp_time_B,
            },
            "clutch_home_vs_away": {
                "description": "Clutch residual split by home vs away shooter.",
                "model_A": exp_home_A,
                "model_B": exp_home_B,
            },
        },
    }

    with open(DOCS / "clutch_effect.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"    Saved → docs/clutch_effect.json")

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STOP POINT 3 — Summary")
    print("=" * 60)
    print(f"\nPRIMARY RESULT (Model B, overall clutch residual):")
    print(f"  {clutch_B_overall['residual_pp']:+.3f} pp  "
          f"95% CI [{clutch_B_overall['ci_lower_pp']:+.3f}, {clutch_B_overall['ci_upper_pp']:+.3f}]  "
          f"(n={clutch_B_overall['n_shots']:,} shots, {clutch_B_overall['n_games']:,} games)")
    print(f"\nModel A overall clutch residual (less adjusted):")
    print(f"  {clutch_A_overall['residual_pp']:+.3f} pp  "
          f"95% CI [{clutch_A_overall['ci_lower_pp']:+.3f}, {clutch_A_overall['ci_upper_pp']:+.3f}]")
    print(f"\nLATE_BLOWOUT control (Model B):")
    print(f"  {lb_B_overall['residual_pp']:+.3f} pp  "
          f"95% CI [{lb_B_overall['ci_lower_pp']:+.3f}, {lb_B_overall['ci_upper_pp']:+.3f}]")
    print("\n✓ Steps 4 & 5 complete.")


if __name__ == "__main__":
    main()
