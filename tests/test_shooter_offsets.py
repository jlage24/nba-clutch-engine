"""Unit tests for shooter_offsets.py (Phase 3b Step 3).

Tests:
- DerSimonian-Laird recovers known tau2 on simulated data
- DerSimonian-Laird returns 0 when there are no true player differences
- Shrinkage moves estimates toward rbar and more so for small n_p
- Offsets for unseen players default to 0
- p_B clipping bounds [0.01, 0.99]
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.shooter_offsets import (
    dersimonian_laird,
    compute_player_offsets,
)


def test_dersimonian_laird_zero_variance():
    """When true player skill differences are zero, DL estimator returns tau2 = 0."""
    rng = np.random.default_rng(42)
    P = 100
    n_shots = 200
    # True skill is identical (p = 0.45) for all players
    p_true = 0.45
    s2_p = np.full(P, p_true * (1 - p_true) / n_shots)
    # Residuals r_p are just sampling noise N(0, s2_p)
    r_p = rng.normal(0, np.sqrt(s2_p))

    tau2, rbar, b_p = dersimonian_laird(r_p, np.full(P, n_shots), s2_p)
    assert tau2 == 0.0, f"Expected tau2 = 0.0 under null hypothesis, got {tau2}"
    # When tau2 == 0, all shrunk estimates equal rbar
    np.testing.assert_allclose(b_p, rbar, atol=1e-10)


def test_dersimonian_laird_recovers_known_tau2():
    """DL estimator recovers true between-player variance within expected sampling noise."""
    rng = np.random.default_rng(123)
    P = 500  # large number of players to test asymptotic recovery
    n_shots = 300
    p_baseline = 0.45
    s2_p = np.full(P, p_baseline * (1 - p_baseline) / n_shots)

    # True player differences ~ N(0, tau_true^2) with tau_true = 0.04 (4 percentage points)
    tau_true = 0.04
    tau2_true = tau_true ** 2  # 0.0016
    theta_p = rng.normal(0, tau_true, size=P)
    # Observed residuals r_p = theta_p + sampling error
    r_p = theta_p + rng.normal(0, np.sqrt(s2_p))

    tau2_est, rbar, b_p = dersimonian_laird(r_p, np.full(P, n_shots), s2_p)
    tau_est = np.sqrt(tau2_est)
    # Should recover 0.04 within +-0.008 (standard error ~ tau / sqrt(2*P))
    assert abs(tau_est - tau_true) < 0.008, (
        f"DL failed to recover known tau: true={tau_true:.4f}, estimated={tau_est:.4f}"
    )


def test_shrinkage_direction_and_sample_size():
    """Shrinkage moves estimates toward rbar, and more so for smaller n_p."""
    # Two players with identical observed residual r_p = +0.10, but player A has
    # n=10 shots and player B has n=500 shots.
    r_p = np.array([0.10, 0.10])
    n_p = np.array([10, 500])
    p_base = 0.45
    s2_p = np.array([p_base * (1 - p_base) / 10, p_base * (1 - p_base) / 500])

    # Fix tau2 = 0.0016 (tau = 0.04)
    tau2 = 0.0016
    rbar = 0.0
    shrinkage = tau2 / (tau2 + s2_p)
    b_p = rbar + (r_p - rbar) * shrinkage

    # Both must shrink toward rbar=0.0 (so 0 < b_p < 0.10)
    assert 0.0 < b_p[0] < 0.10
    assert 0.0 < b_p[1] < 0.10

    # Small sample player (n=10) must undergo MORE shrinkage (b_p closer to 0)
    assert b_p[0] < b_p[1], (
        f"Expected n=10 player to be shrunk more than n=500 player: b_p[10]={b_p[0]}, b_p[500]={b_p[1]}"
    )
    assert shrinkage[0] < shrinkage[1]


def test_compute_player_offsets_unseen_player():
    """compute_player_offsets assigns 0.0 offset to players with 0 shots or unseen IDs."""
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1, 2, 2, 2],
        "SHOT_MADE_FLAG": [1, 0, 1, 1, 0],
        "p_A": [0.4, 0.4, 0.5, 0.5, 0.5],
    })
    offsets, tau2, rbar, _ = compute_player_offsets(df)
    assert "999" not in offsets or offsets.get("999", 0.0) == 0.0


def test_p_b_clipping_bounds():
    """p_B must remain strictly between [0.01, 0.99] regardless of extreme offset."""
    p_A = np.array([0.02, 0.98])
    b_p = np.array([-0.05, +0.05])
    p_B = np.clip(p_A + b_p, 0.01, 0.99)
    assert p_B[0] == 0.01
    assert p_B[1] == 0.99
