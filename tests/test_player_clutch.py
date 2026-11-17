"""Unit tests for src/player_clutch.py (Phase 3b Step 6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS = ROOT / "docs"

from src.player_clutch import (
    compute_player_differentials,
    estimate_heterogeneity,
    player_bootstrap_tau_d,
    q_profile_ci_tau_d,
)


def _simulate_oof(
    n_players: int = 50,
    n_clutch_per_player: int = 40,
    n_reg_per_player: int = 300,
    true_tau_d: float = 0.0,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate an OOF DataFrame with known true between-player clutch SD."""
    rng = np.random.default_rng(seed)
    rows = []
    player_shifts = rng.normal(-0.03, true_tau_d, size=n_players) if true_tau_d > 0 else np.full(n_players, -0.03)
    for pid in range(n_players):
        p_base = 0.45
        p_c = float(np.clip(p_base + player_shifts[pid], 0.05, 0.95))
        y_c = rng.binomial(1, p_c, size=n_clutch_per_player)
        y_r = rng.binomial(1, p_base, size=n_reg_per_player)
        g_c = [f"G{i % 30:03d}" for i in range(n_clutch_per_player)]
        g_r = [f"G{i % 30:03d}" for i in range(n_reg_per_player)]
        for i in range(n_clutch_per_player):
            rows.append((g_c[i], pid, f"P{pid}", "CLUTCH", int(y_c[i]), p_base))
        for i in range(n_reg_per_player):
            rows.append((g_r[i], pid, f"P{pid}", "REGULAR", int(y_r[i]), p_base))
    return pd.DataFrame(
        rows,
        columns=["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "SET_LABEL", "SHOT_MADE_FLAG", "p_A"],
    )


class TestComputePlayerDifferentials:
    def test_min_clutch_filter(self) -> None:
        df = _simulate_oof(n_players=10, n_clutch_per_player=25, n_reg_per_player=100, seed=1)
        out_30 = compute_player_differentials(df, min_clutch_shots=30)
        out_20 = compute_player_differentials(df, min_clutch_shots=20)
        assert len(out_30) == 0
        assert len(out_20) == 10

    def test_variance_formula(self) -> None:
        df = pd.DataFrame({
            "GAME_ID": ["G1"] * 4,
            "PLAYER_ID": [1] * 4,
            "PLAYER_NAME": ["A"] * 4,
            "SET_LABEL": ["CLUTCH", "CLUTCH", "REGULAR", "REGULAR"],
            "SHOT_MADE_FLAG": [1, 0, 1, 1],
            "p_A": [0.5, 0.5, 0.5, 0.5],
        })
        out = compute_player_differentials(df, min_clutch_shots=2)
        assert len(out) == 1
        # mean_clutch_resid = 0.0, mean_regular_resid = 0.5 -> d_p = -0.5
        assert np.isclose(out["d_p"].iloc[0], -0.5)
        # var_clutch = (0.25 + 0.25) / 4 = 0.125; var_regular = 0.125; sum = 0.25
        assert np.isclose(out["s2_d"].iloc[0], 0.25)


class TestHeterogeneityAndIntervals:
    def test_null_simulation_includes_zero(self) -> None:
        df = _simulate_oof(n_players=60, n_clutch_per_player=40, n_reg_per_player=250, true_tau_d=0.0, seed=7)
        pdf = compute_player_differentials(df, min_clutch_shots=30)
        qprof = q_profile_ci_tau_d(pdf["d_p"].values, pdf["s2_d"].values)
        pboot = player_bootstrap_tau_d(pdf, n_resamples=300, random_state=42)
        assert qprof["tau_d_ci_lower_pp"] == 0.0
        assert pboot["tau_d_ci_lower_pp"] == 0.0

    def test_large_true_heterogeneity_detected(self) -> None:
        df = _simulate_oof(n_players=60, n_clutch_per_player=60, n_reg_per_player=300, true_tau_d=0.12, seed=9)
        pdf = compute_player_differentials(df, min_clutch_shots=30)
        summary, _ = estimate_heterogeneity(pdf)
        qprof = q_profile_ci_tau_d(pdf["d_p"].values, pdf["s2_d"].values)
        assert summary["tau_d_pp"] > 6.0
        assert qprof["tau_d_ci_lower_pp"] > 0.0


@pytest.mark.skipif(
    not (DOCS / "player_clutch.json").exists(),
    reason="docs/player_clutch.json not generated; run src/player_clutch.py first.",
)
class TestPlayerClutchJSON:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        with open(DOCS / "player_clutch.json") as f:
            return json.load(f)

    def test_eligible_player_count(self, data) -> None:
        assert data["summary"]["n_eligible_players"] == 76

    def test_reporting_rule_enforced(self, data) -> None:
        """If heterogeneity is not detectable, no individual players may be listed."""
        if not data["heterogeneity_detectable"]:
            assert "players" not in data
            assert "reporting_outcome" in data
        else:
            assert "players" in data
