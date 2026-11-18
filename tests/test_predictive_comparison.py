"""Unit tests for src/predictive_comparison.py (Phase 3b Step 7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS = ROOT / "docs"

from src.predictive_comparison import (
    compute_oof_shifted_model_b,
    fit_logit_shift,
)


class TestLogitShift:
    def test_exact_mean_calibration(self) -> None:
        rng = np.random.default_rng(42)
        p = rng.uniform(0.2, 0.8, size=500)
        # Shift downward by -0.25 on logit scale
        true_p = expit(logit(p) - 0.25)
        y = rng.binomial(1, true_p)
        delta = fit_logit_shift(p, y)
        p_shifted = expit(logit(p) + delta)
        assert np.isclose(p_shifted.mean(), y.mean(), atol=1e-6)
        assert delta < 0.0

    def test_zero_shift_when_calibrated(self) -> None:
        p = np.full(100, 0.4)
        y = np.array([1] * 40 + [0] * 60, dtype=float)
        delta = fit_logit_shift(p, y)
        assert abs(delta) < 1e-6

    def test_oof_shift_rejects_game_overlap(self) -> None:
        df = pd.DataFrame({
            "GAME_ID": ["G1", "G1", "G2", "G3", "G4", "G5"],
            "fold": [0, 1, 2, 3, 4, 0],
            "p_B": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "SHOT_MADE_FLAG": [1, 0, 1, 0, 1, 0],
        })
        with pytest.raises(ValueError, match="game overlap"):
            compute_oof_shifted_model_b(df)


@pytest.mark.skipif(
    not (DOCS / "predictive_comparison.json").exists(),
    reason="docs/predictive_comparison.json not generated; run src/predictive_comparison.py first.",
)
class TestPredictiveComparisonJSON:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        with open(DOCS / "predictive_comparison.json") as f:
            return json.load(f)

    def test_four_models_present(self, data) -> None:
        expected = {
            "Model A (regular shot-only)",
            "Model B (regular + shooter)",
            "Model B + clutch shift",
            "Clutch-only XGBoost (Phase 3)",
        }
        assert set(data["individual_metrics"].keys()) == expected

    def test_counts_and_exception_documented(self, data) -> None:
        assert data["meta"]["n_clutch_shots"] == 7104
        assert data["meta"]["n_clutch_games"] == 543
        assert data["meta"]["phase3_artifacts_loaded"] is False
        assert "step_7_exception" in data["meta"]

    def test_paired_differences_populated(self, data) -> None:
        assert len(data["paired_differences"]) >= 4
        for pair_vals in data["paired_differences"].values():
            for metric in ("brier", "log_loss", "roc_auc"):
                assert metric in pair_vals
                assert pair_vals[metric]["ci_lower"] <= pair_vals[metric]["ci_upper"]
