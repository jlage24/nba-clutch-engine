# NBA Clutch Decision Simulator App (`app/streamlit_app.py`)

This document describes the interactive Streamlit web application in `app/streamlit_app.py`, its half-court heatmap module in `src/heatmap.py`, the court defaults artifact in `export/court_defaults.json`, and deployment on Streamlit Community Cloud.

For model export details and the prediction formula, see [docs/simulator_export.md](simulator_export.md). For end-state decision formulas, presets, and flags, see [docs/decision_layer.md](decision_layer.md).

## 1. What the App Does

The app lets a visitor configure a late-game NBA situation (period, seconds remaining, pre-shot score margin from the offense's perspective, optional player, clutch-shift mode, overtime value, decision time gate, and left-right smoothing) and compare field-goal options in three tabs:

1. **Decision:** Displays a live state headline, four quick example scenario buttons (`Down 3 · 8 s`, `Down 2 · 8 s`, `Down 1 · 6 s`, `Tied · 5 s`), seven real-shot zone presets, an optional custom court spot in feet relative to the centre of the hoop, a plain-language answer block (main conclusion, next option line, clutch-adjustment stability line), a compact five-column table (`Option`, `Make prob.`, `xP`, `Outcome score`, `Rank`), a stacked horizontal bar showing ending `ahead, tied or behind after the shot, if nothing else happens`, and a collapsed `Details` expander (clutch-shift sensitivity, flags, and assumptions). Outcome scores (`V`) and end-state shares appear only when `time_left_sec <= time_gate_sec` (`24` s by default).
2. **Map (click-to-inspect):** Renders an interactive layered Altair half-court heatmap (`600 × 564` CSS px plot area, `470 / 500` equal x/y scale across `[-25.0, 25.0]` by `[-4.75, 42.25]` ft matching `src/court.py`; fits plot columns of about `610` CSS px or wider without horizontal scrolling and scrolls inside its chart container below that) for `Make probability`, `Expected points (xP)`, or `Outcome score`, using either the `Typical mix` of shot types at each distance or any of the `8` active `ACTION_GROUP` values. Clicking any unmasked 1 ft spot updates the inspect card (`22 ft from the basket · three‑pointer` using a non-breaking hyphen, position relative to the hoop centre, metric readouts) and `"Exact position"` inputs; clicking `"Use this spot as a custom shot"` copies the spot onto the Decision tab's custom shot.
3. **About:** Summarizes the data scope, normal-play vs. clutch separation, limitations, repository link, and runtime versions from `export/manifest.json`.

**Run-time guarantees:** Nothing is refitted at run time, no network requests are made (Streamlit usage stats are disabled in `.streamlit/config.toml`), and the app runs without `data/` or `models/`. No player names are written in docs, tests, or rankings; the optional `Player (optional)` selector lists the `572` exported players alphabetically (default `"No player"`) solely for selection.

## 2. Running Locally

Run from the repository root (`nba-clutch-engine/`):

```bash
venv/bin/streamlit run app/streamlit_app.py
```

If rebuilding `export/court_defaults.json` is needed (requires `data/processed/all_shots.parquet` and `export/model_a.json`):

```bash
venv/bin/python src/export_court_defaults.py
```

Both `app/streamlit_app.py` and the `tests/` suite ensure the repository root is on `sys.path` and import `src/` modules cleanly regardless of working directory.

## 3. File Layout

| File | Size (bytes) | Purpose |
| :--- | ---: | :--- |
| `app/streamlit_app.py` | — | Streamlit UI entrypoint (sidebar, Decision, Map, About, caching) |
| `app/formatting.py` | — | Pure UI formatting, Altair court chart builder, selection parser, summary templates |
| `app/__init__.py` | — | App package marker |
| `app/requirements.txt` | — | Slim runtime dependencies pinned next to the Cloud entrypoint |
| `app/assets/nba-clutch.png` | 4,689 | App logo rendered via `st.logo` (theme colours derived from it) |
| `.streamlit/config.toml` | — | Dark theme palette and `gatherUsageStats = false` |
| `src/heatmap.py` | — | Pure-Python court grid, typical-mix probability engine, court polylines, Matplotlib renderer |
| `src/export_court_defaults.py` | — | Generates `export/court_defaults.json` from shot context only (`seed=42`) |
| `export/court_defaults.json` | 36,403 | Per-distance `ACTION_GROUP` mix weights, `2 ft` density mask, and validation checks |
| `export/` runtime bundle (6 files) | 1,455,751 | `model_a.json` (`819,320`), `zone_presets.json` (`565,042`), `court_defaults.json` (`36,403`), `player_offsets.csv` (`27,040`), `manifest.json` (`5,031`), `clutch_shift.json` (`2,915`) |
| `tests/test_court_defaults.py` | — | Schema, non-outcome guardrail, and shot-count tests for `export/court_defaults.json` |
| `tests/test_heatmap.py` | — | Symmetry, `1e-9` cross-check, mask, polylines vs. `src/court.py`, clean-copy runtime |
| `tests/test_streamlit_app.py` | — | Headless `AppTest` behavior, wording scan, contrast ratios, click-inspect & clean-copy checks |

## 4. How the Decision Table and Half-Court Map Are Computed

### Decision Table (`src/decision_layer.py`)

All option probabilities, `xP`, post-shot outcomes (`P_ahead`, `P_tied`, `P_behind`), end-state value `V`, flags, and clutch-shift sensitivities are computed directly by `evaluate_options(...)` in `src/decision_layer.py` using the formulas documented in [docs/decision_layer.md](decision_layer.md) and [docs/simulator_export.md](simulator_export.md). In brief:

- Baseline `p_A` (`Model A`, `168` trees fitted solely on the `196,198` `REGULAR` shots across `1,230` games in the `2025-26 regular season`) uses `LOC_X`, `LOC_Y`, `SHOT_DISTANCE_FT` (1 ft floor clamp for `IS_PUTBACK = 0`), `SHOT_ANGLE`, `SHOT_VALUE`, `ACTION_GROUP`, and `IS_PUTBACK`.
- An optional shooter applies the shrunk empirical-Bayes offset `b_player` (`572` players in `export/player_offsets.csv`, `0.0` for `10` players with zero regular-play shots or unknown shooters), clipped to `[0.01, 0.99]`.
- In clutch states (`period >= 4`, `time_left_sec <= 300`, `abs(margin) <= 5`), the logit-scale clutch shift (`-0.127007` Model A overall, `-0.150672` Model B overall; exploratory 2PT/3PT cuts in `export/clutch_shift.json`) is applied before averaging rows (`p_final = sigmoid(logit(clip(p_base, 1e-6, 1 - 1e-6)) + shift)`).
- End-state score `V = P_ahead + tie_value * P_tied` (`tie_value = 0.5` by default) is reported only when `time_left_sec <= time_gate_sec` (`24` s by default).
- When left-right smoothing is enabled on a **custom shot**, the app passes two mirrored coordinates (`+LOC_X` and `-LOC_X`, geometry recomputed via `src/shot_geometry.py`), repeated in proportion to integer `ACTION_GROUP` weights under `Typical mix`, so `evaluate_options` averages identically to the map to within `1e-9`.

### Half-Court Heatmap (`src/heatmap.py` and `export/court_defaults.json`)

- **Grid & court lines:** Evaluates a `50 × 48` grid of `1 ft` squares (`LOC_X` centers `-245` to `+245` tenths, `LOC_Y` centers `-45` to `+415` tenths; coordinates in tenths of a foot with hoop centre `(0, 0)`). Features derive from `src/shot_geometry.py` with `IS_PUTBACK = 0`. Altair renders unmasked squares inside the exact `src/court.py` domain (`[-25.0, 25.0]` by `[-4.75, 42.25]` ft) with `autosize="pad"` and fixed `600 × 564` CSS px dimensions (`564 / 600 = 470 / 500`) so `1 ft` horizontal equals `1 ft` vertical; this fits plot columns of about `610` CSS px or wider cleanly and scrolls horizontally inside the chart container on narrower columns without page-level sideways scroll.
- **Typical mix default (`export/court_defaults.json`):** Among the `187,427` non-putback `REGULAR` shots (`196,198` `REGULAR` total minus `8,771` putbacks), `8` of `9` manifest `ACTION_GROUP` values are active (`other` has count `0`). For each integer `SHOT_DISTANCE_FT` with at least `100` such shots (`0` to `33` ft), weights out of `1,000` are computed by largest-remainder rounding so they sum to exactly `1,000`; distances with `< 100` shots fall back to the pooled `>= 22 ft` non-putback distribution (`81,973` shots: `jump_shot 743`, `pullup 172`, `stepback 81`, `fadeaway 3`, `floater 1`).
- **Map probability convention:** At each spot, `predict_make_probability` is evaluated per active `ACTION_GROUP` (with shooter offset and clutch shift applied on the logit scale per group), then combined by the `weights_per_1000`. Under left-right smoothing, final probabilities at `(LOC_X, LOC_Y)` and `(-LOC_X, LOC_Y)` are averaged before computing `xP` and `V` via `evaluate_options`.
- **Density and shot-type masking:** A `25 × 24` (`600` cell) `2 ft` grid over `[-250, 250] × [-50, 430]` tenths requires `>= 10` `REGULAR` shots (counts only, zero outcome leakage). Coverage shares use all `196,198` `REGULAR` shots as the denominator; `30` shots fall outside the `2 ft` grid box (`196,168` inside the box):
  - Raw mask: `401` of `600` cells active, covering `195,852 / 196,198` (**`99.8236%`**) of all `REGULAR` shots (`99.8389%` of the `196,168` in-box shots).
  - Mirror-AND mask (when smoothing is enabled): `397` symmetric cells (`401` minus `4` asymmetric cells whose mirror has `< 10` shots), covering `195,804 / 196,198` (**`99.7992%`**) of all `REGULAR` shots (`99.8144%` of in-box shots).
  - Specific `ACTION_GROUP` mode: additionally masks distances where that group's weight is `< 10 / 1,000`.
- **Court-defaults accuracy checks (`export/court_defaults.json`):**
  - *Primary cell-level check (`cell_level_typical_mix_vs_actual`):* Evaluating the distance-based typical mix at real shot coordinates across the `256` `2 ft` cells with `>= 100` non-putback `REGULAR` shots (`180,739` shots, `96.4317%` of non-putback regular play) yields shot-weighted mean `|Δp_A| = 0.004165`, max `0.024458`, and `0` cells `>= 0.03` (largest cell at `(0.0, -20.0)` with `168` shots, `35.12%` dunks, actual `0.796297` vs. mix `0.771840`).
  - *Within-cell geometry diagnostic (`cell_centre_diagnostic`):* Comparing all `REGULAR` shots in `>= 100`-shot cells against typical-mix `p_A` at cell centres (`257` cells, `189,596` shots) yields weighted mean `0.015022` and max `0.119435` (`10` rim cells `>= 0.03`), reflecting shot bunching closer to the rim inside `2 ft` squares plus putbacks.
  - *Modal baseline comparison (`modal_comparison`):* Using only the distance-modal `ACTION_GROUP` (`min_shots=30`) matched `65.50%` of `REGULAR` shots (`64.59%` non-putback) with `20,000`-shot (`seed=42`) mean `|Δp_A| = 0.021865` (max `0.380794`), motivating the typical mix.
- **Caching:** `@st.cache_resource` loads exported artifacts once per server process; `@st.cache_data` caches the computed make-probability, `xP`, and `V` arrays across `(period, time_left_sec, margin, player_id, shift_mode, shot_type, smooth, time_gate_sec, tie_value)`.

## 5. Limitations

- **Single season only:** `219,160` shots (`2025-26 regular season`: `196,198` normal-play, `15,858` late-blowout, `7,104` clutch across `543` clutch games).
- **Field goals only:** No free throws, shooting fouls, offensive rebounds, turnovers, or subsequent opponent possessions.
- **No defender or shot-clock tracking:** Public shot-chart context only. Baseline make probability does not depend on time remaining or margin, and very short clocks may disallow some shot types in real play.
- **Baseline vs. clutch separation & exploratory cuts:** Model A is fitted solely on normal-play shots; clutch outcomes inform only the descriptive clutch shift for close late-game states (`period >= 4`, `time_left_sec <= 300`, `|margin| <= 5`), which bundles defense, fatigue, within-type selection, and shooter mix. The by-shot-type cut is exploratory (Model B 3PT minus 2PT = `-0.116925` logit, rounded `-0.117`, 95% bootstrap CI `[-0.217228, -0.017297]`; interval close to zero).
- **Decision-layer assumptions:** Overtime `tie_value = 0.5` (coin flip) and the `24` s decision gate are modelling choices, not estimated parameters. Shift sensitivity reflects sensitivity to the clutch shift only, not a confidence interval for make probability.
- **Shooter offsets:** Shrunk single-season normal-play estimates pulling low-volume shooters toward `0.0`; no player-specific clutch effect is modelled, and the selector is not a player ranking.
- **Zone presets vs. pointwise map limits:** Zone presets average over fixed real-shot samples (`300` or `1,000` rows) to wash out Model A's local left-right tree splits (up to `0.042` difference between mirrored points); custom shots and unsmoothed map points retain left-right asymmetry unless smoothing is on. Because the map is a **pointwise** model using a distance-only typical mix: (1) within about `4` ft of the rim the typical mix can understate real shots because dunk share depends on court location while the mix depends only on distance (up to `0.024458` gap), and (2) a pointwise map cannot depict how shots bunch closer to the rim inside `2 ft` squares (cell-centre diagnostic gap up to `0.119435`).
