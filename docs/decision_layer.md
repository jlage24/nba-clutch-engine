# Clutch Decision Layer (Phase 4b, Field Goals Only)

This document describes the pure-Python clutch decision layer in `src/decision_layer.py`, the court geometry helper in `src/shot_geometry.py`, and the real-shot zone presets in `export/zone_presets.json`. All modules run strictly on `export/` without `data/` or `models/`.

**Scope of v1: field goals only.** No free throws, fouls on the shot, offensive rebounds, turnovers, or opponent possession after the shot are modelled.

## 1. Public API and Inputs

Entry point: `evaluate_options(state, options, player_id=None, shift_mode="overall", time_gate_sec=24.0, tie_value=0.5, model=None, presets=None)` returning `DecisionResult(table, meta)`.

- **`state`** (`GameState` or `dict`):
  - `time_left_sec` (`float`, validated in `[0, 3000]`): seconds remaining in the period.
  - `margin` (`int`, bool/non-integer rejected): offense minus defense **before** the shot.
  - `period` (`int >= 4`, default `4`).
- **`options`** (non-empty sequence): preset keys from `export/zone_presets.json` and/or custom shot dicts containing `label` and the seven baseline features (`LOC_X`, `LOC_Y`, `SHOT_DISTANCE_FT`, `SHOT_ANGLE`, `SHOT_VALUE`, `ACTION_GROUP`, `IS_PUTBACK`). Unknown presets or unknown `ACTION_GROUP` values raise `ValueError`.
- **`player_id`** (optional `int | str | None`): chosen once for all options. Unknown or absent players receive offset `0.0` (falling back to Model A) and set the flag `unknown_player_offset_zero`.
- **`shift_mode`**: `"overall"` (default), `"by_shot_type"` (exploratory), or `"none"`.
- **`time_gate_sec`** (`float >= 0`, default `24.0`): decision gate threshold.
- **`tie_value`** (`float` in `[0, 1]`, default `0.5`): weight assigned to a tied post-shot score in `V`.

## 2. Geometry and Zone Presets

### Court Coordinate Geometry (`src/shot_geometry.py`)

Verified across all `219,160` shots in `data/processed/all_shots.parquet`:
- `SHOT_DISTANCE_FT = floor(hypot(LOC_X, LOC_Y) / 10)` (truncation, matching `100.0%` of stored shots; non-putback shots below `1.0` ft are clamped to `1.0` ft to match `src/simulator_model.py`).
- `SHOT_ANGLE = degrees(arctan2(LOC_X, LOC_Y))` (`100.0%` exact match).
- `SHOT_VALUE` assigns `3` when `|LOC_X| >= 220` in the corners (`y <= sqrt(237.5^2 - 220^2)`) or `hypot(LOC_X, LOC_Y) >= 237.5` on the arc, else `2` (`99.9909%` exact agreement; `20` mismatches on the three-point line).

### Real-Shot Zone Presets (`export/zone_presets.json`)

A single representative coordinate per zone is fragile because Model A uses signed `LOC_X` and `SHOT_ANGLE`, so make probability at a single point and its mirror image can differ by up to `0.042` (restricted area) due to local tree splits. **Custom single-point options inherit the model's local structure, including left-right differences in Model A; zone presets average over fixed samples of real regular-play shots to avoid this.**

Presets are drawn from `REGULAR` shots only (`196,198` shots, `seed=42`, no `GAME_ID`, `PLAYER_ID`, or `SHOT_MADE_FLAG`): `300` shots per zone by default, redrawn with `1,000` rows if `|mean_p_A_sample - mean_p_A_zone| > 0.003`. In the decision layer, a preset's probability is the mean over its `n_rows` of the per-row final probabilities (player offset and clutch shift applied per row on the logit scale before averaging).

| Preset | `n_zone` | `n_sample` | Mean `p_A` (zone) | SD `p_A` (zone) | SE sample mean | Mean `p_A` (sample) | Abs diff | `abs_diff / SE` | L/R/C share (zone) | L/R/C share (sample) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `restricted_area` | 47,257 | 1,000 | 0.672248 | 0.139727 | 0.004419 | 0.667921 | 0.004327 | 0.9794 | 0.459/0.507/0.034 | 0.473/0.498/0.029 |
| `putback` | 8,771 | 1,000 | 0.597027 | 0.109569 | 0.003465 | 0.598056 | 0.001028 | 0.2968 | 0.183/0.211/0.606 | 0.190/0.212/0.598 |
| `paint` | 39,038 | 1,000 | 0.448208 | 0.055498 | 0.001755 | 0.446677 | 0.001531 | 0.8725 | 0.469/0.518/0.013 | 0.479/0.506/0.015 |
| `short_mid_range` | 10,207 | 300 | 0.435051 | 0.028492 | 0.001645 | 0.435925 | 0.000874 | 0.5313 | 0.492/0.504/0.003 | 0.477/0.513/0.010 |
| `long_mid_range` | 9,604 | 300 | 0.406014 | 0.031081 | 0.001794 | 0.405658 | 0.000356 | 0.1984 | 0.506/0.491/0.003 | 0.490/0.510/0.000 |
| `corner_three` | 21,230 | 300 | 0.389943 | 0.015777 | 0.000911 | 0.389889 | 0.000054 | 0.0592 | 0.518/0.482/0.000 | 0.473/0.527/0.000 |
| `above_the_break_3` | 60,044 | 300 | 0.355492 | 0.018909 | 0.001092 | 0.356831 | 0.001339 | 1.2261 | 0.521/0.477/0.002 | 0.493/0.500/0.007 |

## 3. Decision Formulas, Clutch Shift, and Gate

### Baseline and Clutch Shift

The baseline model was fitted on normal-play (`REGULAR`) shots only; clutch outcomes were used only to estimate the descriptive **clutch shift** (or **clutch gap**), a descriptive quantity bundling defense, fatigue, shot selection within a shot type, and shooter composition:

```text
p_final = sigmoid(logit(clip(p_base, 1e-6, 1 - 1e-6)) + shift)
```

The clutch shift is applied **only** in a clutch state (`period >= 4`, `time_left_sec <= 300`, `abs(margin) <= 5`). Outside a clutch state, no shift is applied and `shift_not_applied_outside_clutch` is set.

- **`"overall"` (default):** Model A `-0.127007` (`95% CI [-0.174140, -0.081083]`) when `player_id=None`; Model B `-0.150672` (`95% CI [-0.196396, -0.105809]`) when a known shooter is chosen.
- **`"by_shot_type"` (exploratory, not pre-specified):** uses 2PT vs 3PT shifts (Model A: 2PT `-0.082224`, 3PT `-0.195819`; Model B: 2PT `-0.104636`, 3PT `-0.221561`). In this cut the 3PT shift is more negative than the 2PT shift (Model B 3PT minus 2PT = `-0.117` logit, 95% CI `-0.217` to `-0.017`; one of several exploratory cuts, interval close to zero).
- **`"none"`:** no clutch shift applied (`shift = 0`).

### Outcomes After the Shot and End-State Value `V`

With make probability `p`, shot value `v` (`2` or `3`), and pre-shot margin `m` (offense minus defense), the probabilities of ending **ahead, tied or behind after the shot, if nothing else happens** are:

```text
P_ahead  = p * [m + v > 0] + (1 - p) * [m > 0]
P_tied   = p * [m + v == 0] + (1 - p) * [m == 0]
P_behind = 1 - P_ahead - P_tied
V        = P_ahead + tie_value * P_tied
xP       = p * v
```

- Default `tie_value = 0.5` treats overtime as a coin flip; this is a pre-specified modelling choice, not an estimate.
- Options are ranked by `V`, then `xP`, then `label`. When all options have identical `V` (differences `< 1e-9`), `no_option_changes_end_state` is set and options are ranked by `xP` only.
- `meta["v_gap_first_second"]` records `V(rank 1) - V(rank 2)` when the gate is open and at least two options are provided (otherwise `None`).

### Time Gate and Sensitivity to the Clutch Shift

- **Time gate (`time_gate_sec = 24.0`):** End-state columns (`p_ahead`, `p_tied`, `p_behind`, `v`, `rank_v`, `rank_xp`) are computed only when `time_left_sec <= time_gate_sec`. Above the gate, the "nothing else happens" assumption is not credible, so decision columns are empty (`NaN` / `<NA>`) and `decision_gate_closed` is set. The 24-second threshold (about one shot clock) is a modelling choice, not estimated from data.
- **Sensitivity to the clutch shift:** Each option is also evaluated under (a) no shift, (b) the lower 95% shift bound, and (c) the upper 95% shift bound, recomputing `V` for each. This is a **sensitivity to the clutch shift only**, not a confidence interval for make probability (it excludes baseline model error, player-offset error, and unmodelled factors).
- **`ranking_robust`:** `ranking_robust` concerns the **top-ranked option only** (True iff rank 1 by `V` is the same under no shift, lower bound, point shift, and upper bound). Ranks 2 and 3 can still swap between shift modes or shift bounds.

## 4. Demo Scenarios (`docs/decision_demo.json`)

### Primary Scenarios (`shift_mode="overall"`, no shooter)

#### Scenario 1: Down 3, 8s left (overall, no shooter)

- **State:** `time_left_sec=8.0`, `margin=-3`, `period=4` | **Shift mode:** `overall` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.015942` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.328279 | 0.984836 | 0.000000 | 0.328279 | 0.671721 | 0.164139 | 0.178415 | 0.158992 | 0.169235 | 2 | 3 |
| Corner Three | 300 | 3 | 0.360163 | 1.080489 | 0.000000 | 0.360163 | 0.639837 | 0.180081 | 0.194945 | 0.174692 | 0.185401 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.375561 | 0.751121 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.641813 | 1.283626 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1 |

#### Scenario 2: Down 2, 8s left (overall, no shooter)

- **State:** `time_left_sec=8.0`, `margin=-2`, `period=4` | **Shift mode:** `overall` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.031884` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.328279 | 0.984836 | 0.328279 | 0.000000 | 0.671721 | 0.328279 | 0.356831 | 0.317984 | 0.338470 | 2 | 3 |
| Corner Three | 300 | 3 | 0.360163 | 1.080489 | 0.360163 | 0.000000 | 0.639837 | 0.360163 | 0.389889 | 0.349385 | 0.370802 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.375561 | 0.751121 | 0.000000 | 0.375561 | 0.624439 | 0.187780 | 0.202829 | 0.182309 | 0.193173 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.641813 | 1.283626 | 0.000000 | 0.641813 | 0.358187 | 0.320907 | 0.333960 | 0.315969 | 0.325671 | 3 | 1 |

#### Scenario 3: Down 1, 6s left (overall, no shooter)

- **State:** `time_left_sec=6.0`, `margin=-1`, `period=4` | **Shift mode:** `overall` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.266252` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.328279 | 0.984836 | 0.328279 | 0.000000 | 0.671721 | 0.328279 | 0.356831 | 0.317984 | 0.338470 | 4 | 3 |
| Corner Three | 300 | 3 | 0.360163 | 1.080489 | 0.360163 | 0.000000 | 0.639837 | 0.360163 | 0.389889 | 0.349385 | 0.370802 | 3 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.375561 | 0.751121 | 0.375561 | 0.000000 | 0.624439 | 0.375561 | 0.405658 | 0.364617 | 0.386347 | 2 | 4 |
| Restricted Area | 1000 | 2 | 0.641813 | 1.283626 | 0.641813 | 0.000000 | 0.358187 | 0.641813 | 0.667921 | 0.631937 | 0.651342 | 1 | 1 |

#### Scenario 4: Tied, 5s left (overall, no shooter)

- **State:** `time_left_sec=5.0`, `margin=0`, `period=4` | **Shift mode:** `overall` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.133126` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.328279 | 0.984836 | 0.328279 | 0.671721 | 0.000000 | 0.664139 | 0.678415 | 0.658992 | 0.669235 | 4 | 3 |
| Corner Three | 300 | 3 | 0.360163 | 1.080489 | 0.360163 | 0.639837 | 0.000000 | 0.680081 | 0.694945 | 0.674692 | 0.685401 | 3 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.375561 | 0.751121 | 0.375561 | 0.624439 | 0.000000 | 0.687780 | 0.702829 | 0.682309 | 0.693173 | 2 | 4 |
| Restricted Area | 1000 | 2 | 0.641813 | 1.283626 | 0.641813 | 0.358187 | 0.000000 | 0.820907 | 0.833960 | 0.815969 | 0.825671 | 1 | 1 |

#### Scenario 5: Down 3, 40s left (gate closed)

- **State:** `time_left_sec=40.0`, `margin=-3`, `period=4` | **Shift mode:** `overall` | **Shooter:** `none` | **Gate open:** `False` | **`v_gap_first_second`:** `None` | **`ranking_robust`:** `None` | **Flags:** `['decision_gate_closed']`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.328279 | 0.984836 | — | — | — | — | — | — | — | — | — |
| Corner Three | 300 | 3 | 0.360163 | 1.080489 | — | — | — | — | — | — | — | — | — |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.375561 | 0.751121 | — | — | — | — | — | — | — | — | — |
| Restricted Area | 1000 | 2 | 0.641813 | 1.283626 | — | — | — | — | — | — | — | — | — |

#### Scenario 6: Down 5, 10s left (no end-state change)

- **State:** `time_left_sec=10.0`, `margin=-5`, `period=4` | **Shift mode:** `overall` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.000000` | **`ranking_robust`:** `True` | **Flags:** `['no_option_changes_end_state']`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.328279 | 0.984836 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 3 |
| Corner Three | 300 | 3 | 0.360163 | 1.080489 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.375561 | 0.751121 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.641813 | 1.283626 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | 1 |

### Scenario 2 Break-Even `tie_value` Sensitivity

In Scenario 2 (down 2, 8 seconds left, `overall` shift, no shooter), `Corner Three` gives `P_ahead = 0.360163` and `P_tied = 0`, so `V(Corner Three) = 0.360163` regardless of `tie_value`. `Restricted Area` gives `P_ahead = 0` and `P_tied = 0.641813`, so `V(Restricted Area) = tie_value * 0.641813`. Their `V` values are equal at `tie_value = 0.360163 / 0.641813 = 0.561165`. The ranking of these two options depends on this assumption: `Corner Three` leads for `tie_value < 0.561165` (including default `0.5`), whereas `Restricted Area` leads for `tie_value > 0.561165`.

### Scenarios 1 and 2 Sensitivity Reruns (`by_shot_type` & Offset-Bucket Shooters)

**Shooter selection rule:** Among players in export/player_offsets.csv with at least 1,000 regular-play shots, sort by offset ascending (tie-broken by PLAYER_ID) and select index int(0.5 * (N - 1)) for 'median offset' and index int(0.9 * N) for 'top-decile offset'. No PLAYER_ID, name, exact shot count, or exact offset is stored. (25 qualifying players with `>= 1,000` regular-play shots; bucket labels only: `"median offset"` and `"top-decile offset"`).

**Rank 2 / Rank 3 swap in Scenario 2 (`overall` vs `by_shot_type`):** `ranking_robust` concerns the top-ranked option only (`Corner Three` remains rank 1 in all reruns, so `ranking_robust = True`). However, ranks 2 and 3 swap between shift modes when no shooter is selected: under `overall`, `Above-the-Break Three` is rank 2 (`V = 0.328279`) ahead of `Restricted Area` (`V = 0.320907`), whereas under exploratory `by_shot_type`, `Restricted Area` moves to rank 2 (`V = 0.325553`) ahead of `Above-the-Break Three` (`V = 0.313307`). The same rank 2/3 swap appears for the `"median offset"` shooter (`overall`: `Above-the-Break Three` `V = 0.337379` vs `Restricted Area` `V = 0.326292`; `by_shot_type`: `Restricted Area` `V = 0.330981` vs `Above-the-Break Three` `V = 0.321738`).

#### Scenario 1 rerun (by_shot_type, no shooter): margin=-3, 8s left

- **State:** `time_left_sec=8.0`, `margin=-3`, `period=4` | **Shift mode:** `by_shot_type` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.015585` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.313307 | 0.939922 | 0.000000 | 0.313307 | 0.686693 | 0.156654 | 0.178415 | 0.148517 | 0.165415 | 2 | 3 |
| Corner Three | 300 | 3 | 0.344478 | 1.033434 | 0.000000 | 0.344478 | 0.655522 | 0.172239 | 0.194945 | 0.163675 | 0.181414 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.386077 | 0.772155 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.651107 | 1.302213 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1 |

#### Scenario 1 rerun (median offset, overall): margin=-3, 8s left

- **State:** `time_left_sec=8.0`, `margin=-3`, `period=4` | **Shift mode:** `overall` | **Shooter:** `median offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.015892` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.337379 | 1.012136 | 0.000000 | 0.337379 | 0.662621 | 0.168689 | 0.185897 | 0.163624 | 0.173733 | 2 | 3 |
| Corner Three | 300 | 3 | 0.369163 | 1.107490 | 0.000000 | 0.369163 | 0.630837 | 0.184582 | 0.202427 | 0.179295 | 0.189830 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.384548 | 0.769097 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.652584 | 1.305169 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1 |

#### Scenario 1 rerun (median offset, by_shot_type): margin=-3, 8s left

- **State:** `time_left_sec=8.0`, `margin=-3`, `period=4` | **Shift mode:** `by_shot_type` | **Shooter:** `median offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.015545` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.321738 | 0.965215 | 0.000000 | 0.321738 | 0.678262 | 0.160869 | 0.185897 | 0.152544 | 0.169767 | 2 | 3 |
| Corner Three | 300 | 3 | 0.352829 | 1.058486 | 0.000000 | 0.352829 | 0.647171 | 0.176414 | 0.202427 | 0.167680 | 0.185705 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.395456 | 0.790913 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.661963 | 1.323926 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1 |

#### Scenario 1 rerun (top-decile offset, overall): margin=-3, 8s left

- **State:** `time_left_sec=8.0`, `margin=-3`, `period=4` | **Shift mode:** `overall` | **Shooter:** `top-decile offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.016091` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.377803 | 1.133410 | 0.000000 | 0.377803 | 0.622197 | 0.188902 | 0.206884 | 0.183565 | 0.194196 | 2 | 3 |
| Corner Three | 300 | 3 | 0.409986 | 1.229958 | 0.000000 | 0.409986 | 0.590014 | 0.204993 | 0.223414 | 0.199492 | 0.210435 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.425565 | 0.851130 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.696870 | 1.393739 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1 |

#### Scenario 1 rerun (top-decile offset, by_shot_type): margin=-3, 8s left

- **State:** `time_left_sec=8.0`, `margin=-3`, `period=4` | **Shift mode:** `by_shot_type` | **Shooter:** `top-decile offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.015830` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.361310 | 1.083930 | 0.000000 | 0.361310 | 0.638690 | 0.180655 | 0.206884 | 0.171824 | 0.190035 | 2 | 3 |
| Corner Three | 300 | 3 | 0.392970 | 1.178910 | 0.000000 | 0.392970 | 0.607030 | 0.196485 | 0.223414 | 0.187332 | 0.206159 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.436811 | 0.873623 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.705506 | 1.411013 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 1 |

#### Scenario 2 rerun (by_shot_type, no shooter): margin=-2, 8s left

- **State:** `time_left_sec=8.0`, `margin=-2`, `period=4` | **Shift mode:** `by_shot_type` | **Shooter:** `none` | **Gate open:** `True` | **`v_gap_first_second`:** `0.018925` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.313307 | 0.939922 | 0.313307 | 0.000000 | 0.686693 | 0.313307 | 0.356831 | 0.297033 | 0.330829 | 3 | 3 |
| Corner Three | 300 | 3 | 0.344478 | 1.033434 | 0.344478 | 0.000000 | 0.655522 | 0.344478 | 0.389889 | 0.327351 | 0.362829 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.386077 | 0.772155 | 0.000000 | 0.386077 | 0.613923 | 0.193039 | 0.202829 | 0.186279 | 0.200561 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.651107 | 1.302213 | 0.000000 | 0.651107 | 0.348893 | 0.325553 | 0.333960 | 0.319562 | 0.332040 | 2 | 1 |

#### Scenario 2 rerun (median offset, overall): margin=-2, 8s left

- **State:** `time_left_sec=8.0`, `margin=-2`, `period=4` | **Shift mode:** `overall` | **Shooter:** `median offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.031785` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.337379 | 1.012136 | 0.337379 | 0.000000 | 0.662621 | 0.337379 | 0.371794 | 0.327247 | 0.347467 | 2 | 3 |
| Corner Three | 300 | 3 | 0.369163 | 1.107490 | 0.369163 | 0.000000 | 0.630837 | 0.369163 | 0.404853 | 0.358590 | 0.379661 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.384548 | 0.769097 | 0.000000 | 0.384548 | 0.615452 | 0.192274 | 0.210311 | 0.186914 | 0.197589 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.652584 | 1.305169 | 0.000000 | 0.652584 | 0.347416 | 0.326292 | 0.341442 | 0.321587 | 0.330863 | 3 | 1 |

#### Scenario 2 rerun (median offset, by_shot_type): margin=-2, 8s left

- **State:** `time_left_sec=8.0`, `margin=-2`, `period=4` | **Shift mode:** `by_shot_type` | **Shooter:** `median offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.021847` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.321738 | 0.965215 | 0.321738 | 0.000000 | 0.678262 | 0.321738 | 0.371794 | 0.305088 | 0.339535 | 3 | 3 |
| Corner Three | 300 | 3 | 0.352829 | 1.058486 | 0.352829 | 0.000000 | 0.647171 | 0.352829 | 0.404853 | 0.335359 | 0.371410 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.395456 | 0.790913 | 0.000000 | 0.395456 | 0.604544 | 0.197728 | 0.210311 | 0.191246 | 0.205366 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.661963 | 1.323926 | 0.000000 | 0.661963 | 0.338037 | 0.330981 | 0.341442 | 0.325397 | 0.337389 | 2 | 1 |

#### Scenario 2 rerun (top-decile offset, overall): margin=-2, 8s left

- **State:** `time_left_sec=8.0`, `margin=-2`, `period=4` | **Shift mode:** `overall` | **Shooter:** `top-decile offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.032183` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.377803 | 1.133410 | 0.377803 | 0.000000 | 0.622197 | 0.377803 | 0.413769 | 0.367130 | 0.388392 | 2 | 3 |
| Corner Three | 300 | 3 | 0.409986 | 1.229958 | 0.409986 | 0.000000 | 0.590014 | 0.409986 | 0.446828 | 0.398983 | 0.420870 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.425565 | 0.851130 | 0.000000 | 0.425565 | 0.574435 | 0.212783 | 0.231298 | 0.207235 | 0.218262 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.696870 | 1.393739 | 0.000000 | 0.696870 | 0.303130 | 0.348435 | 0.362344 | 0.344090 | 0.352644 | 3 | 1 |

#### Scenario 2 rerun (top-decile offset, by_shot_type): margin=-2, 8s left

- **State:** `time_left_sec=8.0`, `margin=-2`, `period=4` | **Shift mode:** `by_shot_type` | **Shooter:** `top-decile offset` | **Gate open:** `True` | **`v_gap_first_second`:** `0.031660` | **`ranking_robust`:** `True` | **Flags:** `[]`

| Option | `n_rows` | `v` | `p_make` | `xp` | `p_ahead` | `p_tied` | `p_behind` | `V` | `V_no_shift` | `V_low` | `V_high` | `rank_v` | `rank_xp` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Above-the-Break Three | 300 | 3 | 0.361310 | 1.083930 | 0.361310 | 0.000000 | 0.638690 | 0.361310 | 0.413769 | 0.343649 | 0.380069 | 2 | 3 |
| Corner Three | 300 | 3 | 0.392970 | 1.178910 | 0.392970 | 0.000000 | 0.607030 | 0.392970 | 0.446828 | 0.374663 | 0.412318 | 1 | 2 |
| Long Mid-Range (16-24 ft) | 300 | 2 | 0.436811 | 0.873623 | 0.000000 | 0.436811 | 0.563189 | 0.218406 | 0.231298 | 0.211720 | 0.226245 | 4 | 4 |
| Restricted Area | 1000 | 2 | 0.705506 | 1.411013 | 0.000000 | 0.705506 | 0.294494 | 0.352753 | 0.362344 | 0.347609 | 0.358635 | 3 | 1 |

## 5. Limitations

- **Field goals only:** no free throws, fouls on the shot, offensive rebounds, or turnovers are modelled.
- **No opponent possession after the shot:** end-state probabilities describe being ahead, tied, or behind immediately after the shot if nothing else happens.
- **No defender or shot-clock data:** closest-defender distance, contest level, and shot-clock time are absent from the data.
- **Single season:** data covers only the `2025-26` NBA regular season.
- **No time/margin effect on baseline make probability:** Phase 3 found no detectable effect of time remaining or score margin on make probability.
- **Clock feasibility:** very short times remaining (a few seconds) may not allow some shot types in practice.
- **Clutch shift scope:** the clutch shift is estimated on close late-game shots only (`period >= 4`, `time_left_sec <= 300`, `abs(margin) <= 5`) and is a descriptive gap rather than a causal effect.
