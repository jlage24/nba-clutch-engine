# Phase 3 brief: ML engine for the NBA Clutch Decision Engine

You are working in an existing Python repository. Phases 1 and 2 (data pipeline) are finished and validated. Your job is Phase 3: feature engineering, a calibrated shot-make model, evaluation, and explanations. Read this whole document before touching any file, then work through the steps **in order**, stopping after each step to summarize what you did and what you found.

## 1. Project in one paragraph

The project models the probability that a shot goes in during NBA **clutch time** and converts it to **expected points**: `xP = P(make | X) × shot value (2 or 3)`. Clutch means the last 5 minutes of the 4th quarter or overtime with a score margin within 5 points, measured **before** the shot from the shooter's perspective. The final product (Phase 4) is a Streamlit simulator that compares late-game options, so the model must output **well-calibrated probabilities**, not just a good ranking.

## 2. Current repository state

```text
.
├── src/
│   ├── fetch_clutch_shots.py     # shots per team -> data/ (cached)
│   ├── fetch_pbp.py              # PlayByPlayV3 per game -> data/raw/pbp/
│   ├── build_clutch_dataset.py   # join + pre-shot margin + clutch filter
│   ├── check_leakage.py
│   ├── diagnostics.py
│   ├── court.py                  # draw_court(ax), LOC_X/LOC_Y units
│   ├── plot_shots.py
│   └── features.py               # drafted, may be untracked; see step 1
├── data/                         # gitignored
│   ├── late_game_shots_*.parquet
│   └── processed/clutch_shots.parquet
├── docs/
├── requirements.txt
└── README.md
```

- Use the existing virtualenv in `venv/`. Python 3.10+.
- Input for modelling: `data/processed/clutch_shots.parquet` (about 7,100 rows, one season so far: 2025-26 regular season, 543 games, 413 distinct players, median 11 clutch shots per player).
- Key columns available: `GAME_ID` (string, keep as string), `GAME_EVENT_ID`, `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ID`, `PERIOD`, `MINUTES_REMAINING`, `SECONDS_REMAINING`, `ACTION_TYPE`, `SHOT_TYPE`, `SHOT_ZONE_BASIC`, `SHOT_DISTANCE`, `LOC_X`, `LOC_Y` (tenths of a foot, hoop at origin), `SHOT_MADE_FLAG` (target), `SCORE_MARGIN` (pre-shot, shooter's perspective), `location` ('h' or 'v', the shooter's side).
- Observed sanity numbers to compare against: overall FG% about 0.45; 2PT about 0.52; 3PT about 0.32; restricted area about 0.65; mid-range about 0.38; above-the-break 3 about 0.30.

## 3. Hard rules (do not break these)

1. **No data leakage.** Features must only use information that exists *before* the shot. Never use the shot's result, the post-shot score, or assist information (assists exist only on made shots, so they leak the target). Do not use `pointsTotal`, `shotResult`, or anything derived from the play-by-play row of the shot itself except the pre-shot margin already computed.
2. **No player identity as a model feature** in the main model. `PLAYER_ID` stays in the table only to compute player-level residuals afterwards.
3. **Never split randomly by shot.** Shots from the same game must never appear on both sides of any split. If more than one season exists in the data, train on earlier seasons and test on the latest. If only one season exists, hold out about 20% of **games** (grouped split on `GAME_ID`, `random_state=42`) as a final test set and never look at it while tuning. Use `GroupKFold` on `GAME_ID` for cross-validation inside the training set.
4. **Do not modify the existing validated pipeline** (`fetch_*`, `build_clutch_dataset.py`, `check_leakage.py`) unless you find a real bug. If you think you found one, stop and explain before changing anything.
5. **No network calls, no re-downloading data.** Everything needed is on disk.
6. **CPU only, no deep learning.** Gradient-boosted trees (XGBoost) and scikit-learn only. Target: full training and evaluation in well under a few minutes on a laptop CPU.
7. **Do not invent results.** Every number written to the README or to docs must come from an actual run you executed. If something fails or looks off, report it plainly.
8. **Reproducibility.** Fix random seeds, keep `GAME_ID` as a string everywhere, and make every script runnable from the repo root as `python src/<script>.py`.
9. **Code style.** English for code, comments, and docs. Small functions, type hints, docstrings that explain *why* when it is not obvious. Use `pathlib` and paths relative to the repo root (`ROOT = Path(__file__).resolve().parents[1]`).

## 4. Git workflow

- Work on a new branch from an up-to-date `main`: `feat/feature-engineering` for step 1, then `feat/model-training` for steps 2 to 6.
- Make small commits, one per logical step, using conventional commits (`feat:`, `fix:`, `docs:`, `test:`). Never commit anything under `data/`, `venv/`, or large model binaries.
- Update `requirements.txt` when you add a dependency (for example `shap`), keeping it a short hand-pinned list of direct dependencies only.
- Do not push and do not merge. Leave that to me.

## 5. Steps

### Step 1: verify and finish feature engineering (`src/features.py`)

A draft already exists. Verify it before extending it.

1. Run it and read the printed summary. Check: no NaNs in features; `ACTION_GROUP` has an `other` category that is a small minority (if it is large, inspect the raw `ACTION_TYPE` values and improve the keyword mapping); `SHOT_DISTANCE_FT` is mostly under about 40 ft with a few long heaves; `SHOT_ANGLE` spans roughly ±180 degrees (shots from behind the hoop exceed ±90); `SCORE_MARGIN` stays within ±5; `TIME_REMAINING_SEC` is within 0 to 300.
2. Features to build (some already drafted): `TIME_REMAINING_SEC`, `SCORE_MARGIN`, `SHOT_DISTANCE_FT` and `SHOT_ANGLE` from `LOC_X`/`LOC_Y`, `SHOT_VALUE`, `IS_OT`, `IS_HOME`, `IS_PLAYOFFS`, `LOC_X`, `LOC_Y`, and `ACTION_GROUP` (collapsed shot families as a pandas `category`). Season is derived from `GAME_ID` (`2000 + int(GAME_ID[3:5])`), and playoffs from the first three characters being `004`.
3. Do **not** add a `PRESSURE_INDEX` yet. Trees learn the time × margin interaction on their own. Mention in your summary if you think one would help.
4. Keep the code as a function that takes a DataFrame and returns features, and make sure the same transformation can be applied to a **single row** (a dict or one-row DataFrame). A later live-timing mode will need this. Do not build the live mode now.
5. Add `tests/test_features.py` with a few small pytest tests on synthetic rows: shot value for 2PT and 3PT, polar coordinates at known points (for example `LOC_X=0, LOC_Y=100` gives distance 10 ft and angle 0), `IS_OT`, and the action-group mapping for a handful of real action strings.
6. Commit on `feat/feature-engineering`.

**Stop here and report** the printed summary and test results before continuing.

### Step 2: evaluation utilities (`src/evaluate.py`)

Create reusable functions:

- Metrics: Brier score, log loss, ROC AUC, and expected calibration error (10 equal-frequency bins).
- A reliability (calibration) plot function that takes true labels and predicted probabilities for one or more models and saves a PNG to `docs/`.
- A function that reports metrics with a simple bootstrap confidence interval over **games** (resample `GAME_ID`s, not shots), around 500 resamples.

### Step 3: baselines and models (`src/train.py`)

Build and compare, in this order, all on the same splits:

1. **Constant baseline:** predict the training-set make rate for every shot. This is the Brier score a model must beat to be worth anything.
2. **Logistic regression baseline** using only `SHOT_DISTANCE_FT`, `SHOT_ANGLE`, `SHOT_VALUE`, `TIME_REMAINING_SEC`, `SCORE_MARGIN` (standardized).
3. **XGBoost** on all features with `enable_categorical=True`, `tree_method="hist"`. The dataset is small (about 5,700 training rows), so regularize heavily: shallow trees (`max_depth` 2 to 4), `learning_rate` about 0.03 to 0.05, high `min_child_weight` (for example 10 to 50), `subsample` and `colsample_bytree` around 0.7 to 0.9, meaningful `reg_lambda`, and early stopping inside the grouped CV. Tune with a small, bounded search (for example 20 to 40 configurations) scored by **grouped-CV log loss or Brier score**, not accuracy.

Calibration:

- Get **out-of-fold predictions** on the training set with `GroupKFold` (5 folds) for the chosen model.
- Fit a calibrator on those out-of-fold predictions. Try Platt scaling (logistic regression on the logit) and isotonic regression, compare them by out-of-fold Brier score and reliability, and prefer the simpler one unless isotonic is clearly better (isotonic overfits easily on small data).
- Train the final model on the full training set, apply the chosen calibrator, and evaluate **once** on the held-out test set.

Outputs:

- A results table (constant, logistic, XGBoost uncalibrated, XGBoost calibrated) with Brier, log loss, AUC, and ECE on the test set, with game-level bootstrap intervals.
- A reliability plot saved to `docs/calibration.png`.
- Saved artifacts with `joblib` under `models/`: the fitted model, the calibrator, and the ordered feature list. Add `models/` to `.gitignore` unless the files are tiny (under about 1 MB), in which case tell me and let me decide.
- A `metrics.json` (or similar) with the numbers, so the README can quote them.

**Sanity expectations, not targets.** Shot-quality models without defender information usually show only modest discrimination, AUC likely in the low to mid 0.6s. If AUC is above about 0.75, assume there is leakage and investigate before reporting anything. Do not chase a higher score by adding features that violate the hard rules. It is perfectly acceptable, and honest, if XGBoost only slightly beats the logistic baseline; say so.

**Stop and report** the results table, the reliability plot, and any surprises.

### Step 4: explanations (`src/explain.py`)

- Compute SHAP values for the final XGBoost model (tree explainer) on the test set.
- Save a summary plot and a dependence plot for `SHOT_DISTANCE_FT` and for `SCORE_MARGIN` to `docs/`.
- In your report, describe in plain language what the model learned (for example whether trailing or leading changes shot quality, and how time remaining matters). Flag anything that contradicts basketball sense; that is often a bug.

### Step 5: prediction API (`src/predict.py`)

- Load the saved model, calibrator, and feature list.
- Expose `predict_make_probability(shot: dict) -> float` and `expected_points(shot: dict) -> float` (`P(make) × SHOT_VALUE`), reusing the exact feature transformation from step 1 (no duplicated logic).
- Include a tiny `if __name__ == "__main__"` demo that prints xP for a few hand-made shots (a corner three, a rim attempt, a long mid-range jumper) with the game state set to, for example, 90 seconds left and a tied game.
- Add a test that the function returns a probability between 0 and 1 and that results match batch predictions for the same rows.

### Step 6 (optional, do last): player residuals with shrinkage (`src/player_effects.py`)

Only do this if steps 1 to 5 are complete and clean.

- Use **out-of-fold** predictions (never in-sample) to compute, per player, the residual `mean(SHOT_MADE_FLAG - P_hat)` and the shot count `n`.
- Shrink toward zero with an empirical-Bayes estimate: estimate between-player variance by method of moments, then multiply each raw residual by `tau² / (tau² + sigma²/n)`. If the estimated between-player variance is about zero, say so explicitly: it would mean the data show no detectable player effect beyond the shot context.
- Output a table with `PLAYER_NAME`, `n`, raw residual, shrunk residual, and only list players with `n >= 30` in the headline view.
- Add a clear comment and README note that these are noisy estimates from a single season.

## 6. Definition of done

- `python src/features.py`, `python src/train.py`, `python src/explain.py`, and `python src/predict.py` all run from the repo root on a fresh clone with `data/` present, and `pytest` passes.
- The results table, calibration plot, and SHAP plots exist in `docs/`.
- README updated **only with real numbers from your runs**: flip Phase 3 to done only if the steps are finished, add a short "Results" section with the test-set table and the calibration plot, and update the project structure and usage sections. Keep the existing "Known limitations" honest; add a note if results show the model barely beats the logistic baseline.
- A short final report listing what was built, the main numbers, anything suspicious you noticed, and what you would do next (for example adding more seasons).

## 7. When to stop and ask me

Stop and ask rather than guess if: a hard rule seems to conflict with the task; the data looks different from what this brief describes (missing columns, different row counts, unexpected NaNs); you want to change a validated pipeline file; or a result looks too good to be true.