# Portable Simulator Model Export (Phase 4a)

This document describes the portable make-probability export under `export/` and the standalone loader in `src/simulator_model.py`. The export allows the simulator to run without `data/`, `models/`, or `pickle` files.

## 1. Contents of `export/`

| File | Format | Size (bytes) | Purpose |
| :--- | :--- | ---: | :--- |
| `export/model_a.json` | XGBoost JSON | 819,320 | Final Model A fitted on all `196,198` normal-play (`REGULAR`) shots (`n_estimators=168`) |
| `export/player_offsets.csv` | CSV | 27,040 | Shrunk per-player offsets (`PLAYER_ID`, `PLAYER_NAME`, `n_regular_shots`, `offset`) for `572` players |
| `export/clutch_shift.json` | JSON | 2,915 | Constant logit-scale clutch shift estimates with 95% game-level bootstrap intervals |
| `export/manifest.json` | JSON | 5,031 | Exact feature order, `ACTION_GROUP` categories, inference guardrails, versions, and limitations |
| **Total** | | **854,306** | **834.3 KB** |

## 2. How the Export Is Built and Rebuilt

To regenerate all files in `export/` from the Phase 3b processed artifacts (`data/processed/all_shots.parquet`, `data/processed/oof_predictions.parquet`, `docs/baseline_quality.json`, and `docs/predictive_comparison.json`), run from the repository root:

```bash
python src/export_model.py
```

Key training and estimation separation rules:

- **Baseline fitted on normal play only:** Model A and the per-player offsets are fitted exclusively on the `196,198` `REGULAR` shots (`1,230` games, `2025-26 regular season`). No `CLUTCH` or `LATE_BLOWOUT` shot enters baseline training.
- **Model A hyperparameters:** Reuses the Phase 3b tuned configuration from `docs/baseline_quality.json` without retuning (`max_depth=6`, `learning_rate=0.03`, `min_child_weight=50`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=3.0`, `tree_method="hist"`, `random_state=42`), with `n_estimators=168` (the median of Phase 3b per-fold tree counts `[140, 170, 170, 168, 167]`). Double-fit determinism check on `219,160` shots: `exact_match=True` (`max_abs_diff=0.0e+00`).
- **Player offsets:** Estimated with the DerSimonian-Laird empirical-Bayes estimator (`src/shooter_offsets.py`) on all `196,198` `REGULAR` out-of-fold residuals from Model A (`tau2=0.001072`, `tau=3.275` pp, `rbar=-0.000436`, average shrinkage `0.490`). Of the `582` players in the dataset, `572` have at least one `REGULAR` shot and appear in `export/player_offsets.csv`; the remaining `10` players with zero `REGULAR` shots are absent and receive offset `0.0`.
- **Clutch outcomes used only for the shift:** The `7,104` `CLUTCH` shots (`543` games) are used solely to estimate the descriptive constant logit-scale clutch shift from out-of-fold `p_A` and `p_B` predictions.

## 3. Prediction Formula and Inference Rules

Given shot features `X`, optional `player_id`, and `clutch_shift` toggle (`True` or `False`):

1. **Model A baseline (`p_A`):** Evaluate `export/model_a.json` on the ordered baseline features: `LOC_X`, `LOC_Y`, `SHOT_DISTANCE_FT`, `SHOT_ANGLE`, `SHOT_VALUE`, `ACTION_GROUP`, `IS_PUTBACK`.
   - `ACTION_GROUP` is constructed as a `pandas.Categorical` with exact ordered categories from `manifest.json`: `dunk`, `layup`, `hook`, `floater`, `stepback`, `pullup`, `fadeaway`, `jump_shot`, `other`. Unknown or missing `ACTION_GROUP` values raise `ValueError`.
   - **1 ft distance clamp:** If `IS_PUTBACK == 0` and `SHOT_DISTANCE_FT < 1.0`, `SHOT_DISTANCE_FT` is clamped to `1.0` ft (matching `src/predict.py`, since `(0, 0)` coordinates in training data occur almost exclusively on tips and putbacks).
2. **Base probability (`p_base`):**
   - If no shooter is chosen (`player_id=None`): `p_base = p_A`, using the Model A clutch shift when enabled.
   - If a shooter is chosen (`player_id` provided): look up `b_player` in `export/player_offsets.csv` (`0.0` if absent or unknown). When `b_player == 0`, `p_B = p_A`; otherwise `p_B = clip(p_A + b_player, 0.01, 0.99)`. Uses the Model B clutch shift when enabled.
3. **Final probability (`p_final`):**

```text
p_final = sigmoid(logit(clip(p_base, 1e-06, 1 - 1e-06)) + shift)
```

where `shift = 0` when `clutch_shift=False`.

Example using `src/simulator_model.py` (depends only on `export/`, `numpy`, `pandas`, and `xgboost`):

```python
from src.simulator_model import load_export, predict_make_probability

model = load_export()
probs = predict_make_probability(
    model,
    [{"LOC_X": 0.0, "LOC_Y": 250.0, "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "3PT Field Goal"}],
    player_id=None,
    clutch_shift=True,
)
```

## 4. Clutch Shift Estimates

The clutch shift is a **descriptive** quantity (the intercept of a logistic regression of `SHOT_MADE_FLAG` on offset `logit(clip(p_base, 1e-6, 1 - 1e-6))` over the `7,104` `CLUTCH` shots in `543` games). It is a clutch gap that bundles defensive intensity, fatigue, shot selection within a shot type, and shooter composition — descriptive only, not a causal effect.

Intervals below are 95% game-level cluster bootstrap intervals (`2,000` resamples, `numpy.random.default_rng(42)`).

| Estimand | `n_shots` | `n_games` | Shift (logit) | 95% Bootstrap CI |
| :--- | ---: | ---: | ---: | :---: |
| **Model A (overall)** | 7,104 | 543 | `-0.127007` | `[-0.174140, -0.081083]` |
| **Model B (overall)** | 7,104 | 543 | `-0.150672` | `[-0.196396, -0.105809]` |
| *Exploratory* Model A — 2PT | 4,246 | 519 | `-0.082224` | `[-0.139883, -0.018925]` |
| *Exploratory* Model A — 3PT | 2,858 | 501 | `-0.195819` | `[-0.272682, -0.115447]` |
| *Exploratory* Model B — 2PT | 4,246 | 519 | `-0.104636` | `[-0.159403, -0.040883]` |
| *Exploratory* Model B — 3PT | 2,858 | 501 | `-0.221561` | `[-0.299050, -0.141030]` |
| *Exploratory* Model B — 3PT minus 2PT | 4,246 / 2,858 | 543 | `-0.116925` | `[-0.217228, -0.017297]` |

- **Comparison with Phase 3b per-fold shifts (`docs/predictive_comparison.json`):** The `5` cross-fitted Model B clutch shifts in Phase 3b were `-0.15196`, `-0.15879`, `-0.14590`, `-0.14875`, `-0.14794` (range `-0.15879` to `-0.14590`, mean `-0.150668`). The exported Model B overall shift (`-0.150672`) falls inside this per-fold range.
- **2-point vs 3-point policy:** Exploratory, not pre-specified: in this cut the 3-point clutch shift is more negative than the 2-point shift (Model B, 3PT minus 2PT = -0.117 logit, 95% CI -0.217 to -0.017; one of several exploratory cuts, with the interval close to zero). The simulator uses one overall shift by default; the per-shot-type shifts are exported so the app can offer an optional switch. Across the `2,000` bootstrap resamples, the correlation between the Model B 2-point and 3-point shifts is `-0.062702` (`std_err_logit` of difference = `0.050990`).
- **Interval comparison note:** The 2-point and 3-point intervals each include the overall shift, but this does not test whether the two shifts differ, and the paired difference above is the relevant quantity.

## 5. Limitations

- **Single season and observational comparison:** Data covers only the `2025-26 regular season` (`219,160` total shots across `1,230` games).
- **No defender or shot-clock tracking:** No closest-defender distance, contest level, or shot-clock data is available.
- **Descriptive clutch gap:** The clutch shift bundles defensive intensity, fatigue, shot selection within shot type, and shooter composition; it is descriptive only, not a causal effect.
- **Out-of-fold vs full-data mismatch:** The clutch shift was estimated with out-of-fold `p_A`/`p_B` (from models trained on 80% of games), whereas the exported Model A is fitted on 100% of regular-play shots (a second-order difference).
- **All-regular-play player offsets:** The offsets in the export use all `196,198` regular-play shots (out-of-fold residuals), so players with few shots are shrunk more toward the grand mean, and the `10` players with no regular-play shots are absent from `export/player_offsets.csv` and receive offset `0.0`.
- **Fixed-offset uncertainty:** Treating the player offsets as fixed when bootstrapping the clutch shift slightly understates the uncertainty of the Model B shift.

*(Built with Python 3.14.4, XGBoost 3.4.1, pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.1.)*
