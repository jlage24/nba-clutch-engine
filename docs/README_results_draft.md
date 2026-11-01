# Results & Empirical Findings (Phase 3 v1)

This section documents the performance benchmarks, calibration audits, feature ablations, and interpretability insights of the NBA Clutch Decision Engine. All models predict the pre-shot probability of a field goal being made, $P(\text{make} \mid X)$, in clutch situations (defined as the final 5 minutes of regulation or overtime with a score margin within 5 points).

---

## 1. Held-Out Test Set Benchmark

The models were trained and evaluated on the 2025–26 NBA regular season dataset (all `GAME_ID` values start with `00225`). 
- **Training set:** 5,550 shots across 434 games (empirical make rate 43.77%).
- **Held-out test set:** 1,554 shots across 109 games (empirical make rate 43.69%), created via a 20% grouped-by-game holdout from the **same 2025–26 season** (shots from the same game never appear on both sides of any split).
- A fresh, untouched season (e.g., 2024–25) is planned as the final external test.

To account for within-game correlation, 95% confidence intervals were generated via 500 game-level cluster bootstrap resamples. Expected Calibration Error (ECE) is reported as a point estimate using 10 equal-frequency (quantile) bins; resampling intervals for ECE are omitted because percentile bootstrap intervals are systematically biased upward under resampling.

| Model | Brier Score (95% CI) | Log Loss (95% CI) | ROC AUC (95% CI) | ECE (10 quantile bins) | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Constant Baseline** | 0.2460 (0.2435, 0.2486) | 0.6852 (0.6801, 0.6903) | 0.5000 (0.5000, 0.5000) | 0.0007 | Predicts training set empirical mean (0.4377) |
| **Logistic Regression (Simple)** | 0.2336 (0.2263, 0.2403) | 0.6600 (0.6444, 0.6741) | 0.6356 (0.6091, 0.6656) | 0.0462 | Distance, angle, value, time, margin |
| **Logistic Regression (Logistic+)** | 0.2328 (0.2257, 0.2393) | 0.6587 (0.6431, 0.6731) | 0.6377 (0.6111, 0.6657) | 0.0409 | + Distance polynomials ($d^2, d^3$), leverage, interaction |
| **Logistic Regression (Logistic++)** | 0.2258 (0.2181, 0.2329) | 0.6430 (0.6246, 0.6609) | 0.6522 (0.6272, 0.6779) | 0.0376 | + One-hot `ACTION_GROUP`, `IS_OT`, `IS_HOME`, `IS_PUTBACK` |
| **XGBoost (Uncalibrated, Default)** | **0.2243** (0.2169, 0.2306) | **0.6391** (0.6231, 0.6523) | **0.6644** (0.6399, 0.6915) | **0.0294** | 11 features, tree depth 3, 49 trees |
| **XGBoost (Platt Calibrated)** | 0.2247 (0.2165, 0.2315) | 0.6399 (0.6217, 0.6548) | 0.6644 (0.6399, 0.6915) | 0.0339 | Post-hoc sigmoid calibration ($A=1.1449, B=0.0440$) |

> [!NOTE]
> On the 1,554-shot test set (~155 shots per quantile bin), binning sampling noise alone produces an expected ECE of ~0.03 even for a perfectly calibrated model. Therefore, out-of-fold calibration on the full training set (OOF ECE: 0.0132 for Platt, 0.0113 for raw XGBoost) and visual reliability diagrams provide more reliable calibration assessments than test-set ECE alone.

### Paired Game-Level Bootstrap Differences (Test Set)

Because shots within the same game share contextual factors, pairwise model comparisons are evaluated on identical resampled games:

- **XGBoost vs. Constant Baseline:**
  - $\Delta \text{Brier} = -0.0217$, 95% CI $[-0.0283, -0.0159]$, $p < 0.001$
  - $\Delta \text{Log Loss} = -0.0461$, 95% CI $[-0.0602, -0.0337]$, $p < 0.001$
  - $\Delta \text{ROC AUC} = +0.1644$, 95% CI $[+0.1399, +0.1915]$, $p < 0.001$
- **XGBoost vs. Logistic++ (Fair Linear Baseline):**
  - $\Delta \text{Brier} = -0.0015$, 95% CI $[-0.0050, +0.0010]$, $p = 0.332$
  - $\Delta \text{Log Loss} = -0.0039$, 95% CI $[-0.0127, +0.0020]$, $p = 0.288$
  - $\Delta \text{ROC AUC} = +0.0122$, 95% CI $[+0.0014, +0.0247]$, $p = 0.020$

The XGBoost advantage over `Logistic++` is small (~0.001–0.002 Brier score) and borderline detectable. Notice that whether the advantage is deemed statistically significant depends on an arbitrary 0.05 threshold and the fact that feature sets changed between versions (e.g. Brier difference $p = 0.060$ earlier vs. $p = 0.332$ now). The AUC difference of $+0.0122$ ($p = 0.020$) is one of three uncorrected pairwise comparisons and constitutes weak evidence on its own. We report effect sizes with confidence intervals rather than binary significance verdicts.

---

## 2. Calibration & Model Selection

Calibrator selection was conducted strictly via 5-fold nested grouped cross-validation on the training set, leaving the test set untouched:

- **Platt Scaling (Nested CV):** Brier = 0.2203, Log Loss = 0.6295, ROC AUC = 0.6690, ECE = 0.0132
- **Isotonic Regression (Nested CV):** Brier = 0.2211, Log Loss = 0.6588, ROC AUC = 0.6623, ECE = 0.0193

Isotonic regression suffers a substantial log loss penalty (+0.0293) due to flat probability step-plateaus at extreme bins. In contrast, Platt scaling parameters ($A = 1.1449$, $B = 0.0440$) are very close to the identity mapping, confirming that raw XGBoost probability outputs are already well-calibrated. 

Consequently, **uncalibrated XGBoost is shipped as the production default**, while the Platt calibrator is preserved as an optional flag (`--calibrated`) in `src/predict.py`.

The full reliability diagrams and predicted probability distributions are displayed in `docs/calibration.png`.

---

## 3. Feature Ablation: What Drives Predictability?

To determine which feature groups contribute genuine generalization power, 5-fold grouped cross-validation was run with cumulative feature subsets on the final 11 features (using the optimal hyperparameters: depth 3, $\lambda=3.0$, learning rate 0.05, 49 trees):

| Feature Subset | OOF Brier | OOF Log Loss | OOF ROC AUC | Step Difference vs. Prior Subset ($\Delta \text{Brier}$) | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(a) Location Only** (`LOC_X`, `LOC_Y`, `SHOT_DISTANCE_FT`, `SHOT_ANGLE`) | 0.22431 | 0.63991 | 0.6577 | — | — |
| **(b) + Shot Type** (`ACTION_GROUP`, `SHOT_VALUE`, `IS_PUTBACK`) | **0.22091** | **0.63137** | **0.6649** | **-0.00340** $[-0.00456, -0.00217]$ | **< 0.001** |
| **(c) + TIME_REMAINING_SEC** | 0.22082 | 0.63119 | 0.6669 | -0.00009 $[-0.00055, +0.00038]$ | 0.704 |
| **(d) + SCORE_MARGIN** | 0.22071 | 0.63096 | 0.6683 | -0.00010 $[-0.00037, +0.00016]$ | 0.392 |
| **(e) + IS_HOME** | 0.22043 | 0.63030 | 0.6690 | -0.00028 $[-0.00053, -0.00006]$ | 0.012 |
| **(f) + IS_OT (Full Model, 11 features)** | 0.22039 | 0.63025 | 0.6701 | -0.00004 $[-0.00031, +0.00024]$ | 0.756 |

### Joint Ablation: Location + Shot Type vs. Full Model

A direct paired bootstrap comparison between **Subset (b)** (location and shot type only) and the **Full Model** shows:
- **Out-of-Fold (Train):** 
  - $\Delta \text{Brier} = -0.00052$, 95% CI $[-0.00099, +0.00001]$, $p = 0.064$
  - $\Delta \text{Log Loss} = -0.00112$, 95% CI $[-0.00219, +0.00012]$, $p = 0.068$
  - $\Delta \text{ROC AUC} = +0.0052$, 95% CI $[+0.0013, +0.0090]$, $p = 0.008$
- **Held-Out Test Set:** 
  - $\Delta \text{Brier} = -0.00048$, 95% CI $[-0.00149, +0.00056]$, $p = 0.348$
  - $\Delta \text{Log Loss} = -0.00104$, 95% CI $[-0.00331, +0.00129]$, $p = 0.364$
  - $\Delta \text{ROC AUC} = +0.0047$, 95% CI $[-0.0025, +0.0122]$, $p = 0.180$

**Takeaway:** Only shot location and shot type provide detectable predictive gain. Adding all game context features together (`TIME_REMAINING_SEC`, `SCORE_MARGIN`, `IS_HOME`, `IS_OT`) yields no statistically detectable out-of-fold scoring rule gain over location and shot type alone (test Brier difference 95% CI $[-0.00149, +0.00056]$, $p = 0.348$). 

The ablation suite evaluated ~15 uncorrected comparisons across metrics and subsets; isolated nominal differences (such as the step difference for home court $p = 0.012$ or OOF AUC $p = 0.008$) represent weak evidence and reflect partition noise rather than robust predictive necessity.

---

## 4. Model Explainability & SHAP Insights

TreeExplainer SHAP values were computed on the held-out test set to quantify feature attributions.

### Grouped Domain Importance

| Feature Group | Features Included | Mean \|SHAP\| | % of Total Importance |
| :--- | :--- | :---: | :---: |
| **Location** | `SHOT_DISTANCE_FT`, `LOC_X`, `LOC_Y`, `SHOT_ANGLE` | 0.4418 | **64.59%** |
| **Shot Type** | `ACTION_GROUP`, `SHOT_VALUE`, `IS_PUTBACK` | 0.1652 | **24.14%** |
| **Game Context** | `TIME_REMAINING_SEC`, `SCORE_MARGIN`, `IS_HOME`, `IS_OT` | 0.0771 | **11.27%** |

Together, **location and shot type account for 88.73% of total mean |SHAP|**.

### Feature-Specific SHAP Contributions

1. **Shot Action Group (`ACTION_GROUP`):**
   - On dunks ($n = 100$), `ACTION_GROUP` adds an average of **+1.07 log-odds** to make probability, driving predicted conversion rates above 85–90%.
   - On layups ($n = 415$), `ACTION_GROUP` contributes **-0.14 log-odds** relative to the rim baseline (since dunks dominate conversion at the rim), though rim distance itself contributes a massive positive attribution.
   - On perimeter shot types, categories like pullups ($-0.055$), stepbacks ($-0.047$), and standard jump shots ($-0.056$) receive modest negative adjustments.
2. **Putback Indicator (`IS_PUTBACK`):**
   - Tip and putback attempts (`IS_PUTBACK == 1`, $n = 75$) receive an explicit penalty of **-0.106 log-odds** (mean $|SHAP| = 0.0152$ overall). This isolates the scramble nature and lower conversion rate of tip-ins without distorting normal layup estimates.
   - Non-putbacks (`IS_PUTBACK == 0`, $n = 1,479$) receive a baseline attribution of $+0.011$ log-odds.
3. **Distance Attribution (`SHOT_DISTANCE_FT`):**
   - Close rim proximity (0–1 ft) provides strong positive log-odds (+0.68 to +1.05), which outweighs the putback penalty on tip/putbacks for a net positive make probability.
4. **Coordinates at $(0, 0)$:**
   - Coordinates at $(0, 0)$ occur almost exclusively on tip and putback attempts (98.65% in clutch; 98.11% league-wide). Layups at $(0, 0)$ convert at 53.51% FG%, compared to 70.31% for layups at 0.5–1.5 ft. Adding `IS_PUTBACK` cleanly accounts for this deficit.
   - In production inference (`src/predict.py`), shots specified with distance $< 1.0$ ft are automatically clamped to a minimum of 1.0 ft unless `IS_PUTBACK` is explicitly true, preventing non-putback attempts specified at $(0, 0)$ from receiving tip/putback conversion rates.
5. **Collinearity of Spatial Coordinates:**
   - `LOC_X`, `LOC_Y`, `SHOT_DISTANCE_FT`, and `SHOT_ANGLE` are geometrically collinear. Individual SHAP shares inside the location group reflect tree split competition and are arbitrary; only the group total (64.59%) is physically meaningful.

Visualizations are archived in `docs/`:
- Beeswarm summary: `docs/shap_summary.png`
- Distance dependence: `docs/shap_dependence_distance.png`
- Margin dependence: `docs/shap_dependence_margin.png`
- Action group mean impact: `docs/shap_action_group_bar.png`

---

## 5. What We Did NOT Find

To maintain rigorous scientific skepticism, we explicitly document null findings:

1. **No Detectable Effect of Clock Time or Score Margin on Make Probability:**
   - While raw descriptive statistics show a drop in FG% in the final 30 seconds (40.2% vs. 45.7% from 60–180s), and trailing teams attempt more three-pointers (46.1% vs. 37.7% when leading), controlled out-of-fold ablation demonstrates that neither `TIME_REMAINING_SEC` nor `SCORE_MARGIN` improves out-of-fold scoring rule accuracy.
   - The tree-based SHAP attribution assigned to `TIME_REMAINING_SEC` (rank 4, 6.8% of mean |SHAP|) reflects credit-sharing across correlated features (which likely reflects shot-mix composition) rather than an incremental causal or predictive necessity.
2. **No Detectable Non-Linear Advantage Over Logistic++:**
   - When logistic regression is given access to the same domain features (one-hot action groups, putback flag, distance polynomials, angle, and leverage), the gap between XGBoost and logistic regression shrinks to ~0.001–0.002 Brier score (paired test difference 95% CI $[-0.0050, +0.0010]$, $p = 0.332$). Most non-linear structure in shot making is captured by categorical shot type and smooth distance effects.

---

## 6. Critical Methodological Limitations

1. **Clutch-Only Dataset Restriction (No "Clutch Effect"):**
   - The dataset consists exclusively of clutch-time shots (final 5 min of Q4/OT, margin $\le 5$). Because no non-clutch baseline data is included, **this model cannot measure whether players shoot better or worse in the clutch than in normal game flow**. Any claim regarding "clutch performance differentials" or "choking" is physically unverifiable from this dataset alone.
2. **Repeated Test Set Inspection:**
   - During iterative model diagnostics, test-set performance and SHAP values were inspected repeatedly (for example, diagnosing the $(0, 0)$ distance anomaly motivated the `IS_PUTBACK` feature). Consequently, the 2025–26 held-out test set is no longer completely pristine. An out-of-sample fresh season (such as 2024–25) will be required as the final untouched evaluation.
3. **Absence of Defender Tracking Data:**
   - The model relies strictly on pre-shot optical coordinates and play-by-play metadata. Without defender distance, contest level, or passer velocity, the model predicts the average conversion probability of a shot taken from that location and type, not the defensive difficulty of the attempt.
