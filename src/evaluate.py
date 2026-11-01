"""Evaluation utilities for the NBA Clutch Decision Engine.

Provides calibration and discrimination metrics (Brier score, log loss, ROC AUC,
Expected Calibration Error with equal-frequency bins), publication-quality
reliability plots, and game-level clustered bootstrap confidence intervals
(both per-model and paired model differences).
"""

import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # Headless backend for CPU/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of predicted probabilities."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.lr = LogisticRegression(random_state=random_state)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        p_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = logit(p_clipped).reshape(-1, 1)
        self.lr.fit(logits, y)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = logit(p_clipped).reshape(-1, 1)
        return self.lr.predict_proba(logits)[:, 1]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = ROOT / "docs"


def expected_calibration_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_prob: Union[np.ndarray, pd.Series, Sequence[float]],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-frequency (quantile) bins.

    Bins on values (not ranks) using quantiles as bin edges, ensuring that identical
    probabilities (ties, such as from step functions or isotonic calibration) remain
    in the same bin.

    For constant predictors with a single unique value c, ECE evaluates to |c - mean(y)|.
    Formula:
        ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return 0.0

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_p, quantiles))
    if len(edges) < 2:
        return float(abs(np.mean(y_t) - np.mean(y_p)))

    bins = pd.cut(y_p, bins=edges, include_lowest=True)
    df = pd.DataFrame({"y": y_t, "p": y_p, "bin": bins})
    grouped = df.groupby("bin", observed=False)
    counts = grouped.size()
    valid = counts > 0
    counts = counts[valid].values
    acc = grouped["y"].mean()[valid].values
    conf = grouped["p"].mean()[valid].values

    ece = (np.abs(acc - conf) * (counts / n)).sum()
    return float(ece)


def compute_calibration_curve(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_prob: Union[np.ndarray, pd.Series, Sequence[float]],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute observed make rates and mean predicted probabilities per quantile bin.

    Returns:
        (bin_confs, bin_accs, bin_counts) as numpy arrays.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_p, quantiles))
    if len(edges) < 2:
        return np.array([float(np.mean(y_p))]), np.array([float(np.mean(y_t))]), np.array([len(y_t)])

    bins = pd.cut(y_p, bins=edges, include_lowest=True)
    df = pd.DataFrame({"y": y_t, "p": y_p, "bin": bins})
    grouped = df.groupby("bin", observed=False)
    counts = grouped.size()
    valid = counts > 0
    confs = grouped["p"].mean()[valid].values
    accs = grouped["y"].mean()[valid].values
    counts = counts[valid].values
    return confs, accs, counts


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_prob: Union[np.ndarray, pd.Series, Sequence[float]],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute standard classification and calibration metrics.

    Returns:
        dict with keys: 'brier', 'log_loss', 'roc_auc', 'ece'
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    y_p_clipped = np.clip(y_p, 1e-15, 1.0 - 1e-15)

    brier = float(brier_score_loss(y_t, y_p_clipped))
    ll = float(log_loss(y_t, y_p_clipped))

    if len(np.unique(y_t)) > 1:
        auc = float(roc_auc_score(y_t, y_p_clipped))
    else:
        auc = float("nan")

    ece = float(expected_calibration_error(y_t, y_p, n_bins=n_bins))

    return {
        "brier": brier,
        "log_loss": ll,
        "roc_auc": auc,
        "ece": ece,
    }


def bootstrap_metrics(
    game_ids: Union[np.ndarray, pd.Series, Sequence[str]],
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_prob: Union[np.ndarray, pd.Series, Sequence[float]],
    n_resamples: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
    n_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute cluster bootstrap confidence intervals by resampling GAME_IDs.

    Resampling whole games preserves intra-game correlation structure.
    Returns:
        dict mapping metric name -> {'point': float, 'ci_lower': float, 'ci_upper': float, 'std_err': float}
    """
    g_arr = np.asarray(game_ids)
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    point_estimates = compute_metrics(y_t, y_p, n_bins=n_bins)

    # Pre-index shots by game for speed
    unique_games = np.unique(g_arr)
    n_games = len(unique_games)
    game_df = pd.DataFrame({"game": g_arr, "idx": np.arange(len(g_arr))})
    game_groups = game_df.groupby("game")["idx"].apply(np.array).to_dict()

    rng = np.random.default_rng(random_state)
    boot_records = {m: [] for m in point_estimates}

    for _ in range(n_resamples):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        sample_indices = np.concatenate([game_groups[g] for g in sampled_games])

        y_boot = y_t[sample_indices]
        p_boot = y_p[sample_indices]
        metrics_b = compute_metrics(y_boot, p_boot, n_bins=n_bins)
        for m, val in metrics_b.items():
            boot_records[m].append(val)

    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)

    results = {}
    for m in point_estimates:
        arr = np.array(boot_records[m])
        # Filter out NaNs if any fold had single-class target for AUC
        valid_arr = arr[~np.isnan(arr)] if np.isnan(arr).any() else arr
        if m == "ece":
            # Resampled ECE is biased upward due to finite-sample bin noise; report point estimate only
            results[m] = {
                "point": point_estimates[m],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "std_err": np.nan,
            }
        else:
            results[m] = {
                "point": point_estimates[m],
                "ci_lower": float(np.percentile(valid_arr, lower_p)) if len(valid_arr) else np.nan,
                "ci_upper": float(np.percentile(valid_arr, upper_p)) if len(valid_arr) else np.nan,
                "std_err": float(np.std(valid_arr, ddof=1)) if len(valid_arr) > 1 else np.nan,
            }

    return results


def bootstrap_paired_difference(
    game_ids: Union[np.ndarray, pd.Series, Sequence[str]],
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_prob_a: Union[np.ndarray, pd.Series, Sequence[float]],
    y_prob_b: Union[np.ndarray, pd.Series, Sequence[float]],
    n_resamples: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
    n_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute paired game-level bootstrap differences between Model A and Model B.

    Differences are defined as: diff = Model_B - Model_A.
    For Brier and Log Loss, negative differences mean Model_B improves upon Model_A.
    Both models are evaluated on the exact same resampled games in each iteration,
    canceling out game-level variance for maximum statistical power.

    Returns:
        dict mapping metric -> {'point_diff': float, 'ci_lower': float, 'ci_upper': float,
                                'std_err': float, 'p_value': float}
    """
    g_arr = np.asarray(game_ids)
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(y_prob_a, dtype=float)
    p_b = np.asarray(y_prob_b, dtype=float)

    point_a = compute_metrics(y_t, p_a, n_bins=n_bins)
    point_b = compute_metrics(y_t, p_b, n_bins=n_bins)
    point_diffs = {m: point_b[m] - point_a[m] for m in point_a}

    unique_games = np.unique(g_arr)
    n_games = len(unique_games)
    game_df = pd.DataFrame({"game": g_arr, "idx": np.arange(len(g_arr))})
    game_groups = game_df.groupby("game")["idx"].apply(np.array).to_dict()

    rng = np.random.default_rng(random_state)
    boot_diffs = {m: [] for m in point_diffs}

    for _ in range(n_resamples):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        sample_indices = np.concatenate([game_groups[g] for g in sampled_games])

        y_boot = y_t[sample_indices]
        p_a_boot = p_a[sample_indices]
        p_b_boot = p_b[sample_indices]

        metrics_a = compute_metrics(y_boot, p_a_boot, n_bins=n_bins)
        metrics_b = compute_metrics(y_boot, p_b_boot, n_bins=n_bins)

        for m in point_diffs:
            boot_diffs[m].append(metrics_b[m] - metrics_a[m])

    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)

    results = {}
    for m in point_diffs:
        arr = np.array(boot_diffs[m])
        valid_arr = arr[~np.isnan(arr)] if np.isnan(arr).any() else arr

        # Two-sided empirical p-value for H0: difference = 0
        if len(valid_arr) > 0:
            p_pos = np.mean(valid_arr >= 0)
            p_neg = np.mean(valid_arr <= 0)
            two_sided_p = float(min(1.0, 2.0 * min(p_pos, p_neg)))
        else:
            two_sided_p = np.nan

        results[m] = {
            "point_diff": point_diffs[m],
            "ci_lower": float(np.percentile(valid_arr, lower_p)) if len(valid_arr) else np.nan,
            "ci_upper": float(np.percentile(valid_arr, upper_p)) if len(valid_arr) else np.nan,
            "std_err": float(np.std(valid_arr, ddof=1)) if len(valid_arr) > 1 else np.nan,
            "p_value": two_sided_p,
        }

    return results


def plot_reliability(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    models: Mapping[str, Union[np.ndarray, pd.Series, Sequence[float]]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Reliability Diagram (Calibration Curve)",
    n_bins: int = 10,
    figsize: Tuple[int, int] = (9, 8),
) -> plt.Figure:
    """Generate and save a 2-panel calibration plot comparing one or more models.

    Top panel: Observed make rate vs mean predicted probability per quantile bin.
    Bottom panel: Predicted probability distribution per model.
    """
    y_t = np.asarray(y_true, dtype=int)

    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Reference line for perfect calibration
    ax_curve.plot(
        [0.0, 1.0], [0.0, 1.0],
        linestyle="--", color="gray", linewidth=1.5,
        label="Perfect Calibration (y = x)", zorder=1
    )

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (name, probs) in enumerate(models.items()):
        p_arr = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
        color = color_palette[idx % len(color_palette)]

        # Compute quantile curve and ECE
        confs, accs, _ = compute_calibration_curve(y_t, p_arr, n_bins=n_bins)
        ece = expected_calibration_error(y_t, p_arr, n_bins=n_bins)

        # Plot calibration points and connected line
        ax_curve.plot(
            confs, accs,
            marker="o", linewidth=2.0, markersize=6,
            color=color, label=f"{name} (ECE={ece:.3f})", zorder=3
        )

        # Plot marginal probability distribution (omit constant baseline to prevent y-axis distortion)
        if "constant" not in name.lower():
            ax_hist.hist(
                p_arr, bins=np.linspace(0, 1, 31),
                alpha=0.35, color=color, density=True, label=name
            )

    ax_curve.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax_curve.set_ylabel("Observed FG% (Make Rate)", fontsize=11)
    ax_curve.set_xlim(-0.02, 1.02)
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.grid(True, linestyle=":", alpha=0.5)
    ax_curve.legend(loc="upper left", frameon=True, fontsize=10)

    ax_hist.set_xlabel("Predicted Probability P(make | X)", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")

    return fig


def format_metrics_table(results_dict: Mapping[str, Dict[str, Dict[str, float]]]) -> str:
    """Format bootstrap metrics into a markdown table.

    Expects {model_name: {metric_name: {'point': ..., 'ci_lower': ..., 'ci_upper': ...}}}.
    """
    rows = []
    rows.append("| Model | Brier Score (95% CI) | Log Loss (95% CI) | ROC AUC (95% CI) | ECE |")
    rows.append("| :--- | :--- | :--- | :--- | :--- |")

    for model, metrics in results_dict.items():
        b = metrics["brier"]
        b_str = f"{b['point']:.4f} ({b['ci_lower']:.4f}, {b['ci_upper']:.4f})"

        ll = metrics["log_loss"]
        ll_str = f"{ll['point']:.4f} ({ll['ci_lower']:.4f}, {ll['ci_upper']:.4f})"

        auc = metrics["roc_auc"]
        auc_str = f"{auc['point']:.3f} ({auc['ci_lower']:.3f}, {auc['ci_upper']:.3f})"

        ece = metrics["ece"]
        ece_str = f"{ece['point']:.4f}"

        rows.append(f"| {model} | {b_str} | {ll_str} | {auc_str} | {ece_str} |")

    return "\n".join(rows)


def format_paired_diff_table(
    diff_results: Dict[str, Dict[str, float]],
    model_a_name: str,
    model_b_name: str,
    include_metrics: Optional[Sequence[str]] = None,
) -> str:
    """Format paired bootstrap differences into a markdown table.

    By default, excludes ECE because percentile bootstrap intervals for ECE
    are upward-biased under resampling.
    """
    rows = []
    rows.append(f"| Metric | Point Diff ({model_b_name} - {model_a_name}) | 95% Bootstrap CI | Std Err | p-value |")
    rows.append("| :--- | :--- | :--- | :--- | :--- |")

    target_metrics = include_metrics or [m for m in ("brier", "log_loss", "roc_auc") if m in diff_results]
    for metric_name in target_metrics:
        stats = diff_results[metric_name]
        diff_str = f"{stats['point_diff']:+.4f}"
        ci_str = f"[{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]"
        se_str = f"{stats['std_err']:.4f}"
        p_val = stats["p_value"]
        p_str = "< 0.002" if p_val < 0.002 else f"{p_val:.3f}"
        rows.append(f"| {metric_name} | {diff_str} | {ci_str} | {se_str} | {p_str} |")

    return "\n".join(rows)


if __name__ == "__main__":
    from src.features import SRC, TARGET, build_features, split

    print("Running evaluate.py verification...")
    raw_df = pd.read_parquet(SRC)
    feats = build_features(raw_df)
    train_df, test_df = split(feats)

    # Constant baseline: predict training make rate on test set
    train_make_rate = train_df[TARGET].mean()
    y_test = test_df[TARGET].values
    game_ids = test_df["GAME_ID"].values
    p_const = np.full_like(y_test, fill_value=train_make_rate, dtype=float)

    # Synthetic baseline slightly better than constant to verify paired bootstrap
    rng = np.random.default_rng(42)
    p_synth = np.clip(p_const + 0.15 * (y_test - 0.5) + rng.normal(0, 0.05, size=len(y_test)), 0.05, 0.95)

    print(f"\nTest set size: {len(test_df):,} shots across {len(np.unique(game_ids))} games.")
    print(f"Training make rate: {train_make_rate:.4f} | Test make rate: {y_test.mean():.4f}")

    # Bootstrap metrics
    const_boot = bootstrap_metrics(game_ids, y_test, p_const, n_resamples=500)
    synth_boot = bootstrap_metrics(game_ids, y_test, p_synth, n_resamples=500)

    print("\n--- Individual Model Bootstrap Metrics (95% CI over games) ---")
    table = format_metrics_table({"Constant Baseline": const_boot, "Synthetic Model": synth_boot})
    print(table)

    # Paired bootstrap difference
    paired_diff = bootstrap_paired_difference(
        game_ids, y_test, y_prob_a=p_const, y_prob_b=p_synth, n_resamples=500
    )
    print("\n--- Paired Bootstrap Difference (Synthetic Model - Constant Baseline) ---")
    paired_table = format_paired_diff_table(paired_diff, "Constant Baseline", "Synthetic Model")
    print(paired_table)

    # Test calibration plot
    plot_path = DEFAULT_DOCS_DIR / "test_calibration.png"
    plot_reliability(
        y_test,
        {"Constant Baseline": p_const, "Synthetic Model": p_synth},
        save_path=plot_path,
        title="Sanity Check Calibration Plot",
    )
    print(f"\nCalibration plot saved successfully to: {plot_path}")
    if plot_path.exists():
        plot_path.unlink()  # Clean up test artifact
