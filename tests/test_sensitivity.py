"""Unit tests for src/sensitivity.py (Phase 3b sensitivity checks S1 & S2)."""

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

from src.sensitivity import build_lr_features


class TestBuildLRFeatures:
    def test_no_game_state_columns(self) -> None:
        """Logistic regression sensitivity features must exclude game-state columns."""
        df = pd.DataFrame({
            "SHOT_DISTANCE_FT": [5.0, 24.0],
            "SHOT_ANGLE": [-0.5, 0.8],
            "SHOT_VALUE": [2, 3],
            "IS_PUTBACK": [0, 0],
            "ACTION_GROUP": ["layup", "jump_shot"],
            "TIME_REMAINING_SEC": [10, 200],
            "SCORE_MARGIN": [-2, 3],
            "IS_HOME": [1, 0],
            "IS_OT": [0, 1],
        })
        X = build_lr_features(df)
        forbidden = {"time", "margin", "home", "is_ot", "overtime", "period"}
        for col in X.columns:
            assert not any(tok in col.lower() for tok in forbidden), f"Unexpected game-state col: {col}"
        assert np.allclose(X["dist_sq"].values, [25.0, 576.0])
        assert np.allclose(X["dist_cube"].values, [125.0, 13824.0])
        assert np.allclose(X["abs_angle"].values, [0.5, 0.8])


@pytest.mark.skipif(
    not (DOCS / "sensitivity.json").exists(),
    reason="docs/sensitivity.json not generated; run src/sensitivity.py first.",
)
class TestSensitivityJSON:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        with open(DOCS / "sensitivity.json") as f:
            return json.load(f)

    def test_required_sensitivity_models(self, data) -> None:
        assert data["status"] == "sensitivity, not pre-specified"
        for key in ("S1a_beyond_edges", "S1b_5th_best", "S2_logistic_regression"):
            assert key in data
            assert "clutch_residual_A" in data[key]
            assert "clutch_residual_B" in data[key]
            assert "diff_vs_reference_pp" in data[key]
