"""Unit tests for src/clutch_effect.py (Phase 3b Step 4 & 5).

Tests cover:
- bootstrap_residual: point estimate, CI coverage, seed reproducibility.
- _residual_row: label propagation and correct output shape.
- build_descriptive_table: row count and column semantics.
- JSON artifact: required keys present and data types are correct.
- Figure files: written to docs/ when the OOF artifact is present.

Skips that require the generated OOF artifact clearly state the reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------

def _make_mini_oof(n_shots: int = 200, n_games: int = 10, seed: int = 0) -> pd.DataFrame:
    """Create a minimal OOF-like DataFrame for unit testing.

    Has columns: GAME_ID, GAME_EVENT_ID, PLAYER_ID, PLAYER_NAME,
    SET_LABEL, SHOT_VALUE, PERIOD, fold, SHOT_MADE_FLAG, p_A, b_player, p_B.
    """
    rng = np.random.default_rng(seed)
    game_ids = [f"00225{i:04d}" for i in range(n_games)]
    gids = rng.choice(game_ids, size=n_shots)
    player_ids = rng.integers(100, 200, size=n_shots)
    make_flag = rng.integers(0, 2, size=n_shots)
    p_a = np.clip(rng.uniform(0.3, 0.7, size=n_shots), 0.01, 0.99)
    b_player = rng.uniform(-0.05, 0.05, size=n_shots)
    p_b = np.clip(p_a + b_player, 0.01, 0.99)
    sets = rng.choice(["REGULAR", "CLUTCH", "LATE_BLOWOUT"], size=n_shots, p=[0.9, 0.05, 0.05])
    shot_values = rng.choice([2, 3], size=n_shots)
    periods = rng.choice([1, 2, 3, 4], size=n_shots)

    return pd.DataFrame({
        "GAME_ID": gids,
        "GAME_EVENT_ID": np.arange(n_shots),
        "PLAYER_ID": player_ids,
        "PLAYER_NAME": [f"Player{p}" for p in player_ids],
        "SET_LABEL": sets,
        "SHOT_VALUE": shot_values,
        "PERIOD": periods,
        "fold": rng.integers(0, 5, size=n_shots),
        "SHOT_MADE_FLAG": make_flag,
        "p_A": p_a,
        "b_player": b_player,
        "p_B": p_b,
    })


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clutch_effect import (
    _build_game_index,
    _residual_row,
    _residual_row_fast,
    bootstrap_residual,
    build_descriptive_table,
    compute_time_bucket_brier_gain,
    N_BOOT,
)


# ---------------------------------------------------------------------------
# _build_game_index
# ---------------------------------------------------------------------------

class TestBuildGameIndex:
    def test_unique_games(self) -> None:
        gids = np.array(["A", "B", "A", "C", "B"])
        unique, groups = _build_game_index(gids)
        assert set(unique) == {"A", "B", "C"}

    def test_indices_partition(self) -> None:
        gids = np.array(["G1", "G2", "G1", "G2", "G1"])
        _, groups = _build_game_index(gids)
        # Every index appears in exactly one group
        all_idx = sorted(np.concatenate(list(groups.values())))
        assert all_idx == list(range(5))

    def test_group_membership(self) -> None:
        gids = np.array(["A", "B", "A"])
        _, groups = _build_game_index(gids)
        assert set(groups["A"]) == {0, 2}
        assert set(groups["B"]) == {1}


# ---------------------------------------------------------------------------
# bootstrap_residual
# ---------------------------------------------------------------------------

class TestBootstrapResidual:
    def test_point_estimate(self) -> None:
        """Point estimate equals (obs_fg - exp_fg) * 100 in pp."""
        y = np.array([1, 0, 1, 1, 0], dtype=float)
        p = np.array([0.6, 0.4, 0.5, 0.7, 0.3], dtype=float)
        gids = np.array(["G1", "G1", "G2", "G2", "G3"])
        result = bootstrap_residual(gids, y, p, n_resamples=50)
        expected = (y.mean() - p.mean()) * 100
        assert abs(result["residual_pp"] - round(expected, 3)) < 0.001

    def test_ci_contains_point(self) -> None:
        """95% CI should bracket the point estimate."""
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, size=n).astype(float)
        p = np.clip(y + rng.normal(0, 0.2, size=n), 0.01, 0.99)
        gids = np.array([f"G{i % 20}" for i in range(n)])
        result = bootstrap_residual(gids, y, p, n_resamples=200)
        assert result["ci_lower_pp"] <= result["residual_pp"] <= result["ci_upper_pp"]

    def test_seed_reproducibility(self) -> None:
        """Same seed → identical CI."""
        rng_in = np.random.default_rng(99)
        y = rng_in.integers(0, 2, size=100).astype(float)
        p = np.clip(rng_in.uniform(0.3, 0.7, size=100), 0.01, 0.99)
        gids = np.array([f"G{i % 10}" for i in range(100)])
        r1 = bootstrap_residual(gids, y, p, n_resamples=100,
                                rng=np.random.default_rng(42))
        r2 = bootstrap_residual(gids, y, p, n_resamples=100,
                                rng=np.random.default_rng(42))
        assert r1["ci_lower_pp"] == r2["ci_lower_pp"]
        assert r1["ci_upper_pp"] == r2["ci_upper_pp"]


# ---------------------------------------------------------------------------
# _residual_row
# ---------------------------------------------------------------------------

class TestResidualRow:
    def test_output_keys(self) -> None:
        df = _make_mini_oof(n_shots=100, n_games=5)
        rng = np.random.default_rng(0)
        row = _residual_row("test_label", df, "p_A", rng)
        required_keys = {
            "label", "n_shots", "n_games",
            "obs_fg_pct", "exp_fg_pct", "residual_pp",
            "ci_lower_pp", "ci_upper_pp",
        }
        assert required_keys.issubset(set(row.keys()))

    def test_label_propagated(self) -> None:
        df = _make_mini_oof(n_shots=50, n_games=5)
        rng = np.random.default_rng(0)
        row = _residual_row("MyLabel", df, "p_A", rng)
        assert row["label"] == "MyLabel"

    def test_n_shots_correct(self) -> None:
        df = _make_mini_oof(n_shots=80, n_games=4)
        rng = np.random.default_rng(0)
        row = _residual_row("lbl", df, "p_A", rng)
        assert row["n_shots"] == 80

    def test_residual_direction(self) -> None:
        """When y=1 always and p=0.5, residual should be +50 pp."""
        df = _make_mini_oof(n_shots=20, n_games=4)
        df["SHOT_MADE_FLAG"] = 1
        df["p_A"] = 0.5
        rng = np.random.default_rng(0)
        row = _residual_row("ones", df, "p_A", rng)
        assert abs(row["residual_pp"] - 50.0) < 0.01

    def test_ci_ordering(self) -> None:
        df = _make_mini_oof(n_shots=100, n_games=10)
        rng = np.random.default_rng(0)
        row = _residual_row("lbl", df, "p_A", rng)
        assert row["ci_lower_pp"] <= row["residual_pp"] <= row["ci_upper_pp"]


# ---------------------------------------------------------------------------
# build_descriptive_table
# ---------------------------------------------------------------------------

class TestBuildDescriptiveTable:
    def test_row_count(self) -> None:
        """Should have 6 rows: 3 set labels × 2 SHOT_VALUEs."""
        oof = _make_mini_oof(n_shots=500, n_games=20)
        rows = build_descriptive_table(oof)
        assert len(rows) == 6

    def test_set_shot_value_coverage(self) -> None:
        oof = _make_mini_oof(n_shots=500, n_games=20)
        rows = build_descriptive_table(oof)
        labels = {r["label"] for r in rows}
        expected = {
            "REGULAR 2PT", "REGULAR 3PT",
            "LATE_BLOWOUT 2PT", "LATE_BLOWOUT 3PT",
            "CLUTCH 2PT", "CLUTCH 3PT",
        }
        assert labels == expected

    def test_fg_pct_range(self) -> None:
        """Observed FG% should be in [0, 100]."""
        oof = _make_mini_oof(n_shots=500, n_games=20)
        rows = build_descriptive_table(oof)
        for row in rows:
            assert 0 <= row["obs_fg_pct"] <= 100, f"obs_fg_pct out of range: {row}"
            assert 0 <= row["mean_p_A"] <= 100, f"mean_p_A out of range: {row}"
            assert 0 <= row["mean_p_B"] <= 100, f"mean_p_B out of range: {row}"


# ---------------------------------------------------------------------------
# JSON artifact (skips if not present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DOCS / "clutch_effect.json").exists(),
    reason="docs/clutch_effect.json not yet generated; run src/clutch_effect.py first.",
)
class TestClutchEffectJSON:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        with open(DOCS / "clutch_effect.json") as f:
            return json.load(f)

    def test_top_level_keys(self, data) -> None:
        required = {"meta", "descriptive_table", "primary_result", "secondary", "exploratory"}
        assert required.issubset(set(data.keys()))

    def test_primary_result_model_b(self, data) -> None:
        pr = data["primary_result"]["model_B"]
        assert "residual_pp" in pr
        assert "ci_lower_pp" in pr
        assert "ci_upper_pp" in pr
        assert "n_shots" in pr
        assert pr["n_shots"] == 7104

    def test_meta_no_leakage_flag(self, data) -> None:
        assert data["meta"]["clutch_outcomes_in_baseline"] is False

    def test_secondary_has_period_rows(self, data) -> None:
        period_data = data["secondary"]["regular_residual_by_period"]
        assert "model_A" in period_data and "model_B" in period_data
        assert len(period_data["model_A"]) == 4  # Q1–Q4

    def test_exploratory_key_present(self, data) -> None:
        exp = data["exploratory"]
        for key in (
            "clutch_by_time_bucket",
            "clutch_home_vs_away",
            "late_blowout_by_time_bucket",
            "regular_home_vs_away",
            "late_blowout_home_vs_away",
            "clutch_excluding_last_60s",
            "phase3_time_ablation_consistency",
        ):
            assert key in exp

    def test_brier_gain_helper(self) -> None:
        rows = [
            {"n_shots": 100, "residual_pp": -6.0},
            {"n_shots": 300, "residual_pp": -2.0},
        ]
        res = compute_time_bucket_brier_gain(rows)
        # mean = -3.0 pp; var = 0.25*(0.03)^2 + 0.75*(0.01)^2 = 0.000225 + 0.000075 = 0.0003
        assert abs(res["overall_mean_resid_pp"] - (-3.0)) < 1e-6
        assert abs(res["variance_explained_brier_gain"] - 0.0003) < 1e-6
        assert abs(res["oracle_delta_brier"] - (-0.0003)) < 1e-6

    def test_residual_row_fast_matches_point_estimate(self) -> None:
        df = _make_mini_oof(n_shots=120, n_games=8, seed=3)
        r_std = _residual_row("x", df, "p_A", np.random.default_rng(7))
        r_fast = _residual_row_fast("x", df, "p_A", np.random.default_rng(7))
        assert r_std["residual_pp"] == r_fast["residual_pp"]
        assert r_std["obs_fg_pct"] == r_fast["obs_fg_pct"]
        assert r_std["exp_fg_pct"] == r_fast["exp_fg_pct"]
        assert r_fast["ci_lower_pp"] <= r_fast["residual_pp"] <= r_fast["ci_upper_pp"]

    def test_descriptive_table_rows(self, data) -> None:
        assert len(data["descriptive_table"]) == 6
        for row in data["descriptive_table"]:
            assert "mean_b_player_pp" in row

    def test_clutch_minus_late_blowout_present(self, data) -> None:
        diff_sec = data["secondary"]["clutch_minus_late_blowout_difference"]["results"]
        for key in (
            "overall_Model A", "2PT_Model A", "3PT_Model A",
            "overall_Model B", "2PT_Model B", "3PT_Model B",
        ):
            assert key in diff_sec
            assert diff_sec[key]["ci_lower_pp"] <= diff_sec[key]["point_diff_pp"] <= diff_sec[key]["ci_upper_pp"]

    def test_residual_in_range(self, data) -> None:
        """Primary residuals should be within ±50 pp (sanity bound)."""
        for model_key in ["model_A", "model_B"]:
            resid = data["primary_result"][model_key]["residual_pp"]
            assert -50 <= resid <= 50, f"{model_key} residual {resid} out of sanity range"


# ---------------------------------------------------------------------------
# Figure files (skips if not present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DOCS / "clutch_effect_overview.png").exists(),
    reason="docs/clutch_effect_overview.png not generated; run src/clutch_effect.py first.",
)
def test_forest_plot_exists() -> None:
    path = DOCS / "clutch_effect_overview.png"
    assert path.stat().st_size > 10_000, "Forest plot seems too small — may be empty."


@pytest.mark.skipif(
    not (DOCS / "residual_by_period.png").exists(),
    reason="docs/residual_by_period.png not generated; run src/clutch_effect.py first.",
)
def test_period_plot_exists() -> None:
    path = DOCS / "residual_by_period.png"
    assert path.stat().st_size > 10_000, "Period plot seems too small — may be empty."
