"""Programmatically generate docs/simulator_export.md from JSON/CSV artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "export"
DOCS = ROOT / "docs"


def build_simulator_export_markdown(root: Path = ROOT) -> str:
    """Build docs/simulator_export.md text from export/ and docs/ JSON/CSV artifacts."""
    export_dir = root / "export"
    docs_dir = root / "docs"

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    shifts = json.loads((export_dir / "clutch_shift.json").read_text(encoding="utf-8"))
    quality = json.loads((docs_dir / "baseline_quality.json").read_text(encoding="utf-8"))
    _pred_comp = json.loads((docs_dir / "predictive_comparison.json").read_text(encoding="utf-8"))

    sizes = {
        name: (export_dir / name).stat().st_size
        for name in ("model_a.json", "player_offsets.csv", "clutch_shift.json", "manifest.json")
    }
    total_bytes = sum(sizes.values())

    sa = shifts["model_a"]
    sb = shifts["model_b"]
    comp = shifts["phase3b_fold_comparison_model_b"]
    exp = shifts["exploratory"]
    ed = exp["model_b_3pt_minus_2pt"]
    dfacts = manifest["data_facts"]
    po = manifest["player_offsets"]
    m_train = manifest["model_a_training"]
    hp = m_train["hyperparameters"]
    pred_cfg = manifest["prediction_formula"]
    versions = manifest["versions"]

    fold_deltas_str = ", ".join(f"`{d:+.5f}`" for d in comp["per_fold_deltas_logit"])
    features_str = ", ".join(f"`{c}`" for c in manifest["features"])
    cats_str = ", ".join(f"`{c}`" for c in manifest["action_group_categories"])

    return f"""# Portable Simulator Model Export (Phase 4a)

This document describes the portable make-probability export under `export/` and the standalone loader in `src/simulator_model.py`. The export allows the simulator to run without `data/`, `models/`, or `pickle` files.

## 1. Contents of `export/`

| File | Format | Size (bytes) | Purpose |
| :--- | :--- | ---: | :--- |
| `export/model_a.json` | XGBoost JSON | {sizes['model_a.json']:,} | Final Model A fitted on all `{dfacts['regular_shots']:,}` normal-play (`REGULAR`) shots (`n_estimators={m_train['n_estimators']}`) |
| `export/player_offsets.csv` | CSV | {sizes['player_offsets.csv']:,} | Shrunk per-player offsets (`PLAYER_ID`, `PLAYER_NAME`, `n_regular_shots`, `offset`) for `{po['n_players_in_table']:,}` players |
| `export/clutch_shift.json` | JSON | {sizes['clutch_shift.json']:,} | Constant logit-scale clutch shift estimates with 95% game-level bootstrap intervals |
| `export/manifest.json` | JSON | {sizes['manifest.json']:,} | Exact feature order, `ACTION_GROUP` categories, inference guardrails, versions, and limitations |
| **Total** | | **{total_bytes:,}** | **{total_bytes / 1024:.1f} KB** |

## 2. How the Export Is Built and Rebuilt

To regenerate all files in `export/` from the Phase 3b processed artifacts (`data/processed/all_shots.parquet`, `data/processed/oof_predictions.parquet`, `docs/baseline_quality.json`, and `docs/predictive_comparison.json`), run from the repository root:

```bash
python src/export_model.py
```

Key training and estimation separation rules:

- **Baseline fitted on normal play only:** Model A and the per-player offsets are fitted exclusively on the `{dfacts['regular_shots']:,}` `REGULAR` shots (`{dfacts['n_games']:,}` games, `{dfacts['season']}`). No `CLUTCH` or `LATE_BLOWOUT` shot enters baseline training.
- **Model A hyperparameters:** Reuses the Phase 3b tuned configuration from `docs/baseline_quality.json` without retuning (`max_depth={hp['max_depth']}`, `learning_rate={hp['learning_rate']}`, `min_child_weight={hp['min_child_weight']}`, `subsample={hp['subsample']}`, `colsample_bytree={hp['colsample_bytree']}`, `reg_lambda={hp['reg_lambda']}`, `tree_method="{m_train['tree_method']}"`, `random_state={m_train['random_seed']}`), with `n_estimators={m_train['n_estimators']}` (the median of Phase 3b per-fold tree counts `{quality['trees_per_fold']}`). Double-fit determinism check on `{m_train['double_fit_determinism']['n_shots_checked']:,}` shots: `exact_match={m_train['double_fit_determinism']['exact_match']}` (`max_abs_diff={m_train['double_fit_determinism']['max_abs_diff']:.1e}`).
- **Player offsets:** Estimated with the DerSimonian-Laird empirical-Bayes estimator (`src/shooter_offsets.py`) on all `{po['n_regular_shots']:,}` `REGULAR` out-of-fold residuals from Model A (`tau2={po['tau2']:.6f}`, `tau={po['tau_pp']:.3f}` pp, `rbar={po['rbar']:+.6f}`, average shrinkage `{po['avg_shrinkage_factor']:.3f}`). Of the `{po['n_players_total_dataset']:,}` players in the dataset, `{po['n_players_in_table']:,}` have at least one `REGULAR` shot and appear in `export/player_offsets.csv`; the remaining `{po['n_players_absent_zero_regular_shots']}` players with zero `REGULAR` shots are absent and receive offset `0.0`.
- **Clutch outcomes used only for the shift:** The `{dfacts['clutch_shots']:,}` `CLUTCH` shots (`{dfacts['n_clutch_games']:,}` games) are used solely to estimate the descriptive constant logit-scale clutch shift from out-of-fold `p_A` and `p_B` predictions.

## 3. Prediction Formula and Inference Rules

Given shot features `X`, optional `player_id`, and `clutch_shift` toggle (`True` or `False`):

1. **Model A baseline (`p_A`):** Evaluate `export/model_a.json` on the ordered baseline features: {features_str}.
   - `ACTION_GROUP` is constructed as a `pandas.Categorical` with exact ordered categories from `manifest.json`: {cats_str}. Unknown or missing `ACTION_GROUP` values raise `ValueError`.
   - **1 ft distance clamp:** If `IS_PUTBACK == 0` and `SHOT_DISTANCE_FT < 1.0`, `SHOT_DISTANCE_FT` is clamped to `1.0` ft (matching `src/predict.py`, since `(0, 0)` coordinates in training data occur almost exclusively on tips and putbacks).
2. **Base probability (`p_base`):**
   - If no shooter is chosen (`player_id=None`): `p_base = p_A`, using the Model A clutch shift when enabled.
   - If a shooter is chosen (`player_id` provided): look up `b_player` in `export/player_offsets.csv` (`0.0` if absent or unknown). When `b_player == 0`, `p_B = p_A`; otherwise `p_B = clip(p_A + b_player, {pred_cfg['p_b_clip_low']}, {pred_cfg['p_b_clip_high']})`. Uses the Model B clutch shift when enabled.
3. **Final probability (`p_final`):**

```text
p_final = sigmoid(logit(clip(p_base, {pred_cfg['logit_clip_eps']}, 1 - {pred_cfg['logit_clip_eps']})) + shift)
```

where `shift = 0` when `clutch_shift=False`.

Example using `src/simulator_model.py` (depends only on `export/`, `numpy`, `pandas`, and `xgboost`):

```python
from src.simulator_model import load_export, predict_make_probability

model = load_export()
probs = predict_make_probability(
    model,
    [{{"LOC_X": 0.0, "LOC_Y": 250.0, "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "3PT Field Goal"}}],
    player_id=None,
    clutch_shift=True,
)
```

## 4. Clutch Shift Estimates

The clutch shift is a **descriptive** quantity (the intercept of a logistic regression of `SHOT_MADE_FLAG` on offset `logit(clip(p_base, 1e-6, 1 - 1e-6))` over the `{sa['n_shots']:,}` `CLUTCH` shots in `{sa['n_games']:,}` games). It is a clutch gap that bundles defensive intensity, fatigue, shot selection within a shot type, and shooter composition — descriptive only, not a causal effect.

Intervals below are 95% game-level cluster bootstrap intervals (`{shifts['meta']['n_bootstrap']:,}` resamples, `numpy.random.default_rng({shifts['meta']['rng_seed']})`).

| Estimand | `n_shots` | `n_games` | Shift (logit) | 95% Bootstrap CI |
| :--- | ---: | ---: | ---: | :---: |
| **Model A (overall)** | {sa['n_shots']:,} | {sa['n_games']:,} | `{sa['shift_logit']:+.6f}` | `[{sa['ci_lower_logit']:+.6f}, {sa['ci_upper_logit']:+.6f}]` |
| **Model B (overall)** | {sb['n_shots']:,} | {sb['n_games']:,} | `{sb['shift_logit']:+.6f}` | `[{sb['ci_lower_logit']:+.6f}, {sb['ci_upper_logit']:+.6f}]` |
| *Exploratory* Model A — 2PT | {exp['model_a_2pt']['n_shots']:,} | {exp['model_a_2pt']['n_games']:,} | `{exp['model_a_2pt']['shift_logit']:+.6f}` | `[{exp['model_a_2pt']['ci_lower_logit']:+.6f}, {exp['model_a_2pt']['ci_upper_logit']:+.6f}]` |
| *Exploratory* Model A — 3PT | {exp['model_a_3pt']['n_shots']:,} | {exp['model_a_3pt']['n_games']:,} | `{exp['model_a_3pt']['shift_logit']:+.6f}` | `[{exp['model_a_3pt']['ci_lower_logit']:+.6f}, {exp['model_a_3pt']['ci_upper_logit']:+.6f}]` |
| *Exploratory* Model B — 2PT | {exp['model_b_2pt']['n_shots']:,} | {exp['model_b_2pt']['n_games']:,} | `{exp['model_b_2pt']['shift_logit']:+.6f}` | `[{exp['model_b_2pt']['ci_lower_logit']:+.6f}, {exp['model_b_2pt']['ci_upper_logit']:+.6f}]` |
| *Exploratory* Model B — 3PT | {exp['model_b_3pt']['n_shots']:,} | {exp['model_b_3pt']['n_games']:,} | `{exp['model_b_3pt']['shift_logit']:+.6f}` | `[{exp['model_b_3pt']['ci_lower_logit']:+.6f}, {exp['model_b_3pt']['ci_upper_logit']:+.6f}]` |
| *Exploratory* Model B — 3PT minus 2PT | {ed['n_shots_2pt']:,} / {ed['n_shots_3pt']:,} | {ed['n_games_pool']:,} | `{ed['diff_logit']:+.6f}` | `[{ed['ci_lower_logit']:+.6f}, {ed['ci_upper_logit']:+.6f}]` |

- **Comparison with Phase 3b per-fold shifts (`docs/predictive_comparison.json`):** The `{len(comp['per_fold_deltas_logit'])}` cross-fitted Model B clutch shifts in Phase 3b were {fold_deltas_str} (range `{comp['fold_min_logit']:+.5f}` to `{comp['fold_max_logit']:+.5f}`, mean `{comp['fold_mean_logit']:+.6f}`). The exported Model B overall shift (`{sb['shift_logit']:+.6f}`) falls inside this per-fold range.
- **2-point vs 3-point policy:** {exp['warning']} Across the `{shifts['meta']['n_bootstrap']:,}` bootstrap resamples, the correlation between the Model B 2-point and 3-point shifts is `{ed['corr_2pt_3pt_bootstrap']:+.6f}` (`std_err_logit` of difference = `{ed['std_err_logit']:.6f}`).
- **Interval comparison note:** The 2-point and 3-point intervals each include the overall shift, but this does not test whether the two shifts differ, and the paired difference above is the relevant quantity.

## 5. Limitations

- **Single season and observational comparison:** Data covers only the `{dfacts['season']}` (`{dfacts['total_shots']:,}` total shots across `{dfacts['n_games']:,}` games).
- **No defender or shot-clock tracking:** No closest-defender distance, contest level, or shot-clock data is available.
- **Descriptive clutch gap:** The clutch shift bundles defensive intensity, fatigue, shot selection within shot type, and shooter composition; it is descriptive only, not a causal effect.
- **Out-of-fold vs full-data mismatch:** The clutch shift was estimated with out-of-fold `p_A`/`p_B` (from models trained on 80% of games), whereas the exported Model A is fitted on 100% of regular-play shots (a second-order difference).
- **All-regular-play player offsets:** The offsets in the export use all `{po['n_regular_shots']:,}` regular-play shots (out-of-fold residuals), so players with few shots are shrunk more toward the grand mean, and the `{po['n_players_absent_zero_regular_shots']}` players with no regular-play shots are absent from `export/player_offsets.csv` and receive offset `0.0`.
- **Fixed-offset uncertainty:** Treating the player offsets as fixed when bootstrapping the clutch shift slightly understates the uncertainty of the Model B shift.

*(Built with Python {versions['python']}, XGBoost {versions['xgboost']}, pandas {versions['pandas']}, NumPy {versions['numpy']}, scikit-learn {versions['scikit_learn']}.)*
"""


def main() -> None:
    """Regenerate docs/simulator_export.md from exported and docs artifacts."""
    md = build_simulator_export_markdown(ROOT)
    out_path = DOCS / "simulator_export.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
