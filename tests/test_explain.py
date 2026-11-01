"""Unit tests for model explainability (src/explain.py)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explain import (
    FEATURE_GROUPS,
    audit_domain_consistency,
    compute_grouped_importance,
    compute_ranked_importance,
    save_shap_plots,
)


def test_compute_ranked_importance():
    """Ranked importances should be sorted descending and sum to 100%."""
    # Synthetic SHAP matrix: 100 samples, 4 features
    shap_matrix = np.array([
        [0.4, 0.1, -0.2, 0.0],
        [-0.4, 0.2, 0.3, 0.0],
        [0.4, -0.3, -0.1, 0.0],
    ])
    features = ["feat_a", "feat_b", "feat_c", "feat_d"]

    df = compute_ranked_importance(shap_matrix, features)

    assert len(df) == 4
    assert list(df["feature"]) == ["feat_a", "feat_b", "feat_c", "feat_d"]
    assert df["mean_abs_shap"].iloc[0] >= df["mean_abs_shap"].iloc[1]
    assert df["mean_abs_shap"].iloc[1] >= df["mean_abs_shap"].iloc[2]
    assert df["mean_abs_shap"].iloc[2] >= df["mean_abs_shap"].iloc[3]
    assert np.isclose(df["pct_importance"].sum(), 100.0)
    assert df[df["feature"] == "feat_d"]["mean_abs_shap"].iloc[0] == 0.0


def test_compute_grouped_importance():
    """Grouped importances should correctly aggregate features and sum to 100%."""
    ranked_df = pd.DataFrame({
        "feature": ["LOC_X", "LOC_Y", "SHOT_DISTANCE_FT", "SHOT_ANGLE", "ACTION_GROUP", "IS_HOME"],
        "mean_abs_shap": [0.03, 0.05, 0.35, 0.02, 0.15, 0.02],
        "pct_importance": [4.84, 8.06, 56.45, 3.23, 24.19, 3.23],
    })

    groups = {
        "location": ["LOC_X", "LOC_Y", "SHOT_DISTANCE_FT", "SHOT_ANGLE"],
        "shot_type": ["ACTION_GROUP"],
        "game_context": ["IS_HOME"],
    }

    gdf = compute_grouped_importance(ranked_df, groups)

    assert len(gdf) == 3
    assert set(gdf["group"]) == {"location", "shot_type", "game_context"}

    loc_row = gdf[gdf["group"] == "location"].iloc[0]
    assert np.isclose(loc_row["mean_abs_shap"], 0.45)
    assert np.isclose(gdf["pct_importance"].sum(), 100.0)


def test_audit_domain_consistency():
    """Domain consistency audit returns expected structure and valid verdicts."""
    train_df = pd.DataFrame({
        "IS_HOME": [1, 0, 1, 0] * 20,
        "LOC_X": [0, 5, 0, -5] * 20,
        "LOC_Y": [10, 15, 20, 25] * 20,
        "SCORE_MARGIN": [-2, 0, 3, -1] * 20,
        "TIME_REMAINING_SEC": [15, 120, 250, 45] * 20,
        "SHOT_MADE_FLAG": [1, 0, 1, 0] * 20,
    })
    test_df = train_df.copy()

    ranked_df = pd.DataFrame({
        "feature": ["SHOT_DISTANCE_FT", "TIME_REMAINING_SEC", "LOC_Y", "LOC_X", "IS_HOME", "SCORE_MARGIN"],
        "mean_abs_shap": [0.35, 0.06, 0.05, 0.03, 0.02, 0.015],
        "pct_importance": [67.3, 11.5, 9.6, 5.8, 3.8, 2.9],
    })
    grouped_df = compute_grouped_importance(ranked_df, FEATURE_GROUPS)

    audits = audit_domain_consistency(train_df, test_df, ranked_df, grouped_df)

    assert "is_home" in audits
    assert "location_collinearity" in audits
    assert "score_margin" in audits
    assert "time_remaining" in audits
    for name, audit in audits.items():
        assert len(audit["verdict"]) > 0
