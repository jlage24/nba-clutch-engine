"""Training pipeline for the NBA Clutch Decision Engine.

Trains and evaluates:
1. Constant Baseline (training set make rate)
2. Logistic Regression Baseline (standardized core features)
3. XGBoost (tuned with regularized search over 5-fold GroupKFold)
4. Calibrated XGBoost (Platt scaling vs Isotonic regression on OOF predictions)

Generates out-of-fold (OOF) headline comparisons across the whole training set
and evaluates once on the held-out test set as confirmation.
Saves models to models/ and calibration plot / metrics to docs/.
"""

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.evaluate import (
    bootstrap_metrics,
    bootstrap_paired_difference,
    compute_metrics,
    format_metrics_table,
    format_paired_diff_table,
    plot_reliability,
)
from src.features import (
    FEATURES,
    ID_COLS,
    NUMERIC,
    SRC,
    TARGET,
    build_features,
    split,
)

MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
LOGISTIC_FEATURES = [
    "SHOT_DISTANCE_FT",
    "SHOT_ANGLE",
    "SHOT_VALUE",
    "TIME_REMAINING_SEC",
    "SCORE_MARGIN",
]


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


def get_cv_splits(train_df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate fixed GroupKFold splits on GAME_ID to pair all model comparisons."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(train_df, groups=train_df["GAME_ID"]))


def run_constant_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate constant predictor using training make rate."""
    y_train = train_df[TARGET].values
    oof_preds = np.zeros(len(train_df))

    # OOF: predict fold-train make rate on validation fold
    for tr_idx, val_idx in cv_splits:
        oof_preds[val_idx] = y_train[tr_idx].mean()

    # Test set: predict full training set make rate
    full_train_rate = y_train.mean()
    test_preds = np.full(len(test_df), fill_value=full_train_rate, dtype=float)

    return oof_preds, test_preds


def run_logistic_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression, StandardScaler, List[str]]:
    """Evaluate logistic regression baseline on standardized features.

    Automatically excludes zero-variance / constant columns on the training fold
    to prevent divide-by-zero during standardization.
    """
    # Filter features to those with non-zero variance
    valid_features = [
        col for col in LOGISTIC_FEATURES
        if col in train_df.columns and train_df[col].nunique() > 1
    ]

    oof_preds = np.zeros(len(train_df))
    for tr_idx, val_idx in cv_splits:
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[valid_features])
        X_val = scaler.transform(val[valid_features])

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_tr, tr[TARGET])
        oof_preds[val_idx] = lr.predict_proba(X_val)[:, 1]

    # Full train fit and test evaluation
    final_scaler = StandardScaler()
    X_train_full = final_scaler.fit_transform(train_df[valid_features])
    X_test_full = final_scaler.transform(test_df[valid_features])

    final_lr = LogisticRegression(random_state=42, max_iter=1000)
    final_lr.fit(X_train_full, train_df[TARGET])
    test_preds = final_lr.predict_proba(X_test_full)[:, 1]

    return oof_preds, test_preds, final_lr, final_scaler, valid_features


def tune_xgboost(
    train_df: pd.DataFrame,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Bounded hyperparameter search over regularized XGBoost configurations.

    Scores configurations by 5-fold grouped-CV log loss.
    """
    grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05],
        "min_child_weight": [10, 25, 50],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0, 5.0],
    }
    keys, values = zip(*grid.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_cfg = None
    best_log_loss = float("inf")
    best_oof = None

    y_train = train_df[TARGET].values

    for cfg in configurations:
        oof = np.zeros(len(train_df))
        for tr_idx, val_idx in cv_splits:
            tr = train_df.iloc[tr_idx]
            val = train_df.iloc[val_idx]

            m = xgb.XGBClassifier(
                n_estimators=150,
                enable_categorical=True,
                tree_method="hist",
                random_state=42,
                eval_metric="logloss",
                **cfg,
            )
            m.fit(tr[FEATURES], tr[TARGET])
            oof[val_idx] = m.predict_proba(val[FEATURES])[:, 1]

        score = log_loss(y_train, np.clip(oof, 1e-15, 1.0 - 1e-15))
        if score < best_log_loss:
            best_log_loss = score
            best_cfg = cfg
            best_oof = oof

    return best_cfg, best_oof


def select_calibrator(
    y_train: np.ndarray,
    oof_probs: np.ndarray,
) -> Tuple[Any, str, Dict[str, Dict[str, float]]]:
    """Compare Platt scaling vs Isotonic regression on OOF predictions.

    Prefers Platt scaling unless isotonic is clearly superior, as isotonic
    often overfits small sample tails and produces step-function plateaus.
    """
    # 1. Platt scaling
    platt = PlattCalibrator(random_state=42).fit(oof_probs, y_train)
    p_platt = platt.predict_proba(oof_probs)
    platt_metrics = compute_metrics(y_train, p_platt)

    # 2. Isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_probs, y_train)
    p_iso = iso.predict(oof_probs)
    iso_metrics = compute_metrics(y_train, p_iso)

    comparison = {
        "Platt": platt_metrics,
        "Isotonic": iso_metrics,
    }

    # Prefer Platt unless Isotonic has lower Brier by > 0.005
    if iso_metrics["brier"] < platt_metrics["brier"] - 0.005:
        chosen_calibrator = iso
        chosen_name = "Isotonic"
    else:
        chosen_calibrator = platt
        chosen_name = "Platt"

    return chosen_calibrator, chosen_name, comparison


def main():
    print("=" * 70)
    print("NBA Clutch Decision Engine: Model Training & Evaluation")
    print("=" * 70)

    # 1. Load and split data
    print("\n[1/6] Loading data and generating features...")
    raw_df = pd.read_parquet(SRC)
    feats = build_features(raw_df)
    train_df, test_df = split(feats)

    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values
    games_train = train_df["GAME_ID"].values
    games_test = test_df["GAME_ID"].values

    print(f"Total shots: {len(feats):,} across {feats['GAME_ID'].nunique()} games.")
    print(f"Train split: {len(train_df):,} shots across {train_df['GAME_ID'].nunique()} games.")
    print(f"Test split:  {len(test_df):,} shots across {test_df['GAME_ID'].nunique()} games.")
    print(f"Empirical Train FG%: {y_train.mean():.4f} | Test FG%: {y_test.mean():.4f}")

    # 2. CV Splits (Identical for paired comparison)
    print("\n[2/6] Generating 5-fold GroupKFold splits on GAME_ID...")
    cv_splits = get_cv_splits(train_df, n_splits=5)

    # 3. Baselines
    print("\n[3/6] Fitting Constant and Logistic Regression baselines...")
    oof_const, test_const = run_constant_baseline(train_df, test_df, cv_splits)
    oof_lr, test_lr, final_lr, lr_scaler, lr_features = run_logistic_baseline(
        train_df, test_df, cv_splits
    )

    # 4. XGBoost Tuning & OOF
    print("\n[4/6] Tuning regularized XGBoost across 36 hyperparameter configs...")
    best_cfg, oof_xgb_raw = tune_xgboost(train_df, cv_splits)
    print(f"Optimal XGBoost Hyperparameters: {best_cfg}")

    # Train final XGBoost on full training set
    final_xgb = xgb.XGBClassifier(
        n_estimators=150,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        **best_cfg,
    )
    final_xgb.fit(train_df[FEATURES], y_train)
    test_xgb_raw = final_xgb.predict_proba(test_df[FEATURES])[:, 1]

    # 5. Calibration Selection
    print("\n[5/6] Fitting and comparing calibrators on OOF predictions...")
    calibrator, cal_name, cal_comparison = select_calibrator(y_train, oof_xgb_raw)
    print(f"Chosen calibrator: {cal_name}")
    print(f"  Platt OOF Brier:    {cal_comparison['Platt']['brier']:.4f} | LogLoss: {cal_comparison['Platt']['log_loss']:.4f} | ECE: {cal_comparison['Platt']['ece']:.4f}")
    print(f"  Isotonic OOF Brier: {cal_comparison['Isotonic']['brier']:.4f} | LogLoss: {cal_comparison['Isotonic']['log_loss']:.4f} | ECE: {cal_comparison['Isotonic']['ece']:.4f}")

    if cal_name == "Platt":
        oof_xgb_cal = calibrator.predict_proba(oof_xgb_raw)
        test_xgb_cal = calibrator.predict_proba(test_xgb_raw)
    else:
        oof_xgb_cal = calibrator.predict(oof_xgb_raw)
        test_xgb_cal = calibrator.predict(test_xgb_raw)

    # 6. Evaluation & Bootstrap Analysis
    print("\n[6/6] Computing clustered bootstrap metrics (500 resamples)...")
    models_oof = {
        "Constant Baseline": oof_const,
        "Logistic Regression": oof_lr,
        "XGBoost (Uncalibrated)": oof_xgb_raw,
        "XGBoost (Calibrated)": oof_xgb_cal,
    }
    models_test = {
        "Constant Baseline": test_const,
        "Logistic Regression": test_lr,
        "XGBoost (Uncalibrated)": test_xgb_raw,
        "XGBoost (Calibrated)": test_xgb_cal,
    }

    # Headline OOF metrics over full training set
    oof_results = {
        name: bootstrap_metrics(games_train, y_train, probs, n_resamples=500)
        for name, probs in models_oof.items()
    }

    # Test set confirmation metrics
    test_results = {
        name: bootstrap_metrics(games_test, y_test, probs, n_resamples=500)
        for name, probs in models_test.items()
    }

    # Paired bootstrap differences on Test set
    paired_vs_const = bootstrap_paired_difference(
        games_test, y_test, y_prob_a=test_const, y_prob_b=test_xgb_cal, n_resamples=500
    )
    paired_vs_lr = bootstrap_paired_difference(
        games_test, y_test, y_prob_a=test_lr, y_prob_b=test_xgb_cal, n_resamples=500
    )

    print("\n" + "=" * 70)
    print("HEADLINE RESULTS: Out-of-Fold (OOF) on Training Set (5,550 shots, 434 games)")
    print("=" * 70)
    print(format_metrics_table(oof_results))

    print("\n" + "=" * 70)
    print("CONFIRMATION RESULTS: Held-out Test Set (1,554 shots, 109 games)")
    print("=" * 70)
    print(format_metrics_table(test_results))

    print("\n--- Paired Bootstrap Difference on Test Set: XGBoost (Calibrated) vs Constant Baseline ---")
    print(format_paired_diff_table(paired_vs_const, "Constant Baseline", "XGBoost (Calibrated)"))

    print("\n--- Paired Bootstrap Difference on Test Set: XGBoost (Calibrated) vs Logistic Regression ---")
    print(format_paired_diff_table(paired_vs_lr, "Logistic Regression", "XGBoost (Calibrated)"))

    # Note on test-set ECE noise
    print("\nNote on Calibration Metrics:")
    print("With ~155 shots per quantile bin on the 1,554-shot test set, even a theoretically")
    print("perfect model has an expected ECE of ~0.03 from finite-sample bin noise alone.")
    print("Thus, the reliability diagram and OOF calibration metrics are the primary indicators.")

    # Save Reliability Plot
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = DOCS_DIR / "calibration.png"
    plot_reliability(
        y_test,
        models_test,
        save_path=plot_path,
        title="Clutch Shot Probability Calibration (Test Set)",
    )
    print(f"\nReliability plot saved to: {plot_path}")

    # Save Models and Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_path = MODELS_DIR / "xgb_model.joblib"
    cal_path = MODELS_DIR / "calibrator.joblib"
    feat_path = MODELS_DIR / "features.joblib"

    joblib.dump(final_xgb, xgb_path)
    joblib.dump(calibrator, cal_path)
    joblib.dump(FEATURES, feat_path)

    # Check file sizes
    xgb_size_kb = xgb_path.stat().st_size / 1024
    cal_size_kb = cal_path.stat().st_size / 1024
    feat_size_kb = feat_path.stat().st_size / 1024
    print(f"\nSaved Model Artifacts to {MODELS_DIR}:")
    print(f"  - {xgb_path.name}: {xgb_size_kb:.1f} KB")
    print(f"  - {cal_path.name}: {cal_size_kb:.1f} KB")
    print(f"  - {feat_path.name}: {feat_size_kb:.1f} KB")

    # Save metrics JSON
    metrics_summary = {
        "train_make_rate": float(y_train.mean()),
        "test_make_rate": float(y_test.mean()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "best_hyperparameters": best_cfg,
        "calibrator": cal_name,
        "oof_metrics": oof_results,
        "test_metrics": test_results,
        "paired_diff_vs_constant": paired_vs_const,
        "paired_diff_vs_logistic": paired_vs_lr,
    }
    metrics_json_path = DOCS_DIR / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")


if __name__ == "__main__":
    main()
