from pathlib import Path

import numpy as np
import pytest

from src.evaluate import (
    bootstrap_metrics,
    bootstrap_paired_difference,
    compute_calibration_curve,
    compute_metrics,
    expected_calibration_error,
    format_metrics_table,
    format_paired_diff_table,
    plot_reliability,
)


def test_ece_constant_predictor():
    """(a) Constant predictor c must give ECE == |c - mean(y)|."""
    y = np.array([1, 1, 0, 0, 0])  # mean(y) = 0.40
    c = 0.4377
    p = np.full_like(y, fill_value=c, dtype=float)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(abs(c - 0.40))


def test_ece_step_function_heavy_ties():
    """(b) Step-function predictor with heavy ties does not crash and keeps ties together."""
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    # 3 distinct plateaus with 5 samples each (simulating isotonic regression plateaus)
    p = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8])
    ece = expected_calibration_error(y, p, n_bins=10)
    assert not np.isnan(ece)
    assert 0.0 <= ece <= 1.0

    confs, accs, counts = compute_calibration_curve(y, p, n_bins=10)
    # Bins should preserve values without splitting ties
    assert sum(counts) == len(y)
    for conf, acc, cnt in zip(confs, accs, counts):
        assert 0.0 <= conf <= 1.0
        assert 0.0 <= acc <= 1.0
        assert cnt > 0


def test_ece_perfectly_calibrated_large_n():
    """(c) Perfectly calibrated synthetic predictor gives ECE close to zero for large n."""
    rng = np.random.default_rng(42)
    n = 100_000
    p = rng.uniform(0.1, 0.9, size=n)
    y = rng.binomial(1, p)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.005


def test_compute_calibration_curve():
    """Quantile calibration curve should return valid bins and counts."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    p = np.linspace(0.1, 0.9, 10)
    confs, accs, counts = compute_calibration_curve(y, p, n_bins=5)
    assert len(confs) == len(accs) == len(counts)
    assert sum(counts) == 10
    assert np.all(confs >= 0.0) and np.all(confs <= 1.0)
    assert np.all(accs >= 0.0) and np.all(accs <= 1.0)


def test_compute_metrics():
    """compute_metrics returns all required metrics with valid mathematical bounds."""
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    p = np.array([0.7, 0.2, 0.8, 0.3, 0.6, 0.9, 0.1, 0.4, 0.85, 0.15])
    metrics = compute_metrics(y, p, n_bins=3)

    assert set(metrics.keys()) == {"brier", "log_loss", "roc_auc", "ece"}
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_bootstrap_metrics():
    """Clustered bootstrap over games produces valid confidence intervals."""
    game_ids = np.array(["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"] * 5)
    y = np.array([1, 0, 1, 0, 0, 1, 1, 0] * 5)
    p = np.array([0.6, 0.4, 0.7, 0.3, 0.2, 0.8, 0.9, 0.1] * 5)

    res = bootstrap_metrics(game_ids, y, p, n_resamples=50, alpha=0.05, random_state=42)

    for m in ["brier", "log_loss", "roc_auc"]:
        assert m in res
        ci_lower = res[m]["ci_lower"]
        ci_upper = res[m]["ci_upper"]
        std_err = res[m]["std_err"]
        assert ci_lower <= ci_upper
        assert std_err >= 0.0

    # ECE bootstrap CI is invalid due to upward bias; must be omitted (NaN)
    assert "ece" in res
    assert np.isnan(res["ece"]["ci_lower"])
    assert np.isnan(res["ece"]["ci_upper"])
    assert res["ece"]["point"] >= 0.0


def test_bootstrap_paired_difference():
    """Paired bootstrap detects difference between superior and inferior models."""
    game_ids = np.array(["G1", "G1", "G2", "G2", "G3", "G3"] * 10)
    y = np.array([1, 0, 1, 0, 1, 0] * 10)

    # Model A: terrible predictions
    p_a = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9] * 10)
    # Model B: perfect predictions
    p_b = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1] * 10)

    res = bootstrap_paired_difference(
        game_ids, y, y_prob_a=p_a, y_prob_b=p_b, n_resamples=50, random_state=42
    )

    # Model B has vastly lower Brier and Log Loss, so diff (B - A) must be negative
    assert res["brier"]["point_diff"] < -0.4
    assert res["brier"]["ci_upper"] < 0.0
    assert res["brier"]["p_value"] < 0.05

    assert res["log_loss"]["point_diff"] < 0.0
    assert res["log_loss"]["ci_upper"] < 0.0


def test_bootstrap_paired_difference_identical_models():
    """Identical models should have zero difference and p-value 1.0."""
    game_ids = np.array(["G1", "G1", "G2", "G2"] * 5)
    y = np.array([1, 0, 1, 0] * 5)
    p = np.array([0.5, 0.5, 0.5, 0.5] * 5)

    res = bootstrap_paired_difference(
        game_ids, y, y_prob_a=p, y_prob_b=p, n_resamples=20, random_state=42
    )
    assert res["brier"]["point_diff"] == pytest.approx(0.0)
    assert res["brier"]["ci_lower"] == pytest.approx(0.0)
    assert res["brier"]["ci_upper"] == pytest.approx(0.0)
    assert res["brier"]["p_value"] == pytest.approx(1.0)


def test_plot_reliability_and_formatting(tmp_path: Path):
    """Reliability plot saves cleanly and formatting tables generate markdown."""
    y = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    models = {
        "Baseline": np.full(10, 0.5),
        "TestModel": np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.7, 0.2, 0.4]),
    }

    save_file = tmp_path / "test_calibration.png"
    fig = plot_reliability(y, models, save_path=save_file, n_bins=3)
    assert save_file.exists()
    assert save_file.stat().st_size > 0

    # Test table formatting
    metrics_summary = {
        "Baseline": {
            "brier": {"point": 0.25, "ci_lower": 0.24, "ci_upper": 0.26},
            "log_loss": {"point": 0.69, "ci_lower": 0.68, "ci_upper": 0.70},
            "roc_auc": {"point": 0.50, "ci_lower": 0.50, "ci_upper": 0.50},
            "ece": {"point": 0.00, "ci_lower": 0.00, "ci_upper": 0.00},
        }
    }
    table_str = format_metrics_table(metrics_summary)
    assert "Baseline" in table_str
    assert "Brier Score" in table_str

    diff_summary = {
        "brier": {"point_diff": -0.02, "ci_lower": -0.03, "ci_upper": -0.01, "std_err": 0.005, "p_value": 0.002}
    }
    diff_table = format_paired_diff_table(diff_summary, "ModelA", "ModelB")
    assert "Point Diff (ModelB - ModelA)" in diff_table
    assert "-0.0200" in diff_table
