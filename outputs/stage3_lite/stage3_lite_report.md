# Stage 3 Lite: Optimization & Accuracy Lift Report
**Bangladesh Integrated Household Survey (BIHS) Round 3**  
*Benchmarking Feature Engineering, Threshold Optimization, and Ensembling*

---

## 1. Executive Summary

**Stage 3 Lite** was executed to evaluate how much classification accuracy, recall, and discriminative power can be extracted from **pure agro-environmental exposure variables** without violating the strict zero-leakage rule.

### Key Performance Breakthroughs (Held-Out Test Set, $n = 1,121$):
1. **Accuracy Crosses the 80% Mark:** The Soft-Voting Ensemble with threshold tuning ($\tau = 0.40$) achieves **80.55% accuracy**, up from 66.64% (Naive Baseline) and 79.75% (Stage 3 XGBoost Baseline).
2. **Substantial Recall Surge (+15.0% to +21.4%):** 
   - Under standard thresholds ($\tau = 0.50$), previous models missed ~35% of high-vulnerability households.
   - Tuning the threshold to $\tau = 0.40$ on the Soft-Voting Ensemble lifts Recall to **80.21%** (identifying 300 of 374 vulnerable households, cutting false negatives nearly in half from 130 to 74).
   - Tuning for maximum $F_1$ on XGBoost ($\tau = 0.285$) achieves an unprecedented **86.63% Recall** (identifying 324 of 374 vulnerable households).
3. **$F_1$-Score Jump:** Rises from **0.6825 to 0.7335** (+5.10 percentage points lift over Stage 3 baseline).
4. **All-Time High ROC-AUC:** Reaches **0.8686** (Soft-Voting Ensemble), confirming excellent probabilistic discrimination.
5. **3-Class Performance Lift:** Multi-class accuracy increases to **61.82%**, with macro $F_1$ rising to **0.6057** and multiclass ROC-AUC rising to **0.7938**.

---

## 2. Benchmark Comparison Table

| Framework | Task | Model Configuration | Threshold ($\tau$) | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **Stage 3 Baseline** | Binary | Naive Majority Class Predictor | 0.500 | 66.64% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.2223 |
| **Stage 3 Baseline** | Binary | Logistic Regression ($C=0.01$) | 0.500 | 79.48% | 72.50% | 62.03% | 0.6686 | 0.8660 | 0.1395 |
| **Stage 3 Baseline** | Binary | Random Forest ($d=5, n=200$) | 0.500 | 79.66% | **74.01%** | 60.16% | 0.6637 | 0.8626 | 0.1418 |
| **Stage 3 Baseline** | Binary | XGBoost Baseline ($d=3, n=200$) | 0.500 | 79.75% | 71.55% | 65.24% | 0.6825 | 0.8634 | 0.1389 |
| **Stage 3 Lite** | **Binary** | **XGBoost (Engineered Features)** | 0.500 | 79.30% | 69.94% | 66.58% | 0.6822 | 0.8618 | 0.1380 |
| **Stage 3 Lite** | **Binary** | **XGBoost (Threshold Tuned - Max $F_1$)** | **0.285** | 78.06% | 62.31% | **86.63%** | 0.7248 | 0.8618 | 0.1380 |
| **Stage 3 Lite** | **Binary** | **Soft-Voting Ensemble (LR+RF+XGB)** | 0.500 | 79.84% | 71.89% | 64.97% | 0.6826 | **0.8686** | **0.1363** |
| **Stage 3 Lite** | **Binary** | **Soft-Voting Ensemble (Tuned $\tau=0.40$)** | **0.400** | **80.55%** | 67.57% | **80.21%** | **0.7335** | **0.8686** | **0.1363** |
| **Stage 3 Baseline** | 3-Class | Naive Majority Class Predictor | N/A | 33.36% | 11.12% | 33.33% | 0.1668 | 0.5000 | N/A |
| **Stage 3 Baseline** | 3-Class | Logistic Regression | N/A | 61.37% | 59.78% | 61.39% | 0.5992 | 0.7850 | N/A |
| **Stage 3 Baseline** | 3-Class | XGBoost Baseline | N/A | 60.30% | 58.27% | 60.32% | 0.5844 | 0.7803 | N/A |
| **Stage 3 Lite** | **3-Class** | **XGBoost (Nominal Multi-class)** | N/A | 61.20% | 59.65% | 61.21% | 0.5993 | 0.7884 | N/A |
| **Stage 3 Lite** | **3-Class** | **XGBoost (Ordinal Frank & Hall)** | N/A | 60.93% | 59.55% | 60.94% | 0.5994 | 0.7896 | N/A |
| **Stage 3 Lite** | **3-Class** | **Soft-Voting Ensemble (LR+RF+XGB)** | N/A | **61.82%** | **60.33%** | **61.83%** | **0.6057** | **0.7938** | N/A |

---

## 3. Four Methodological Enhancements Implemented in Stage 3 Lite

### 1. Zero-Leakage Environmental Feature Engineering ($p = 39$)
The feature matrix was expanded from 25 to 39 features by deriving domain-informed agro-climatic indices while strictly quarantining all socioeconomic indicators:
- **Seasonal Aridity / Water Deficit Ratios:** $\frac{\text{clim\_evapotrans\_avg}}{\text{clim\_rain\_total\_mm} + 1.0}$ for Aus, Aman, and Boro seasons. High ratios pinpoint severe agricultural drought stress during reproductive rice phases.
- **Flood-Monsoon Compounding Interactions:** $\text{flood\_depth\_ft\_mean} \times \text{clim\_aman\_rain\_total\_mm}$. Captures the non-linear risk multiplier when household inundation occurs simultaneously with heavy monsoon downpours.
- **Regional Agricultural Stress:** Gap from regional yield potential ($\max(0, 1.0 - \text{yield\_ratio})$) across all three growing seasons.
- **Plot-Level Physical Context:** `plot_distance_m_mean`, `total_plot_rainfall`, `mean_plot_temperature`, `max_plot_temperature`, `mean_plot_humidity`, and `mean_plot_solar_rad`. These continuous metrics provide granular, household-specific physical exposure that breaks the district-level clustering barrier.

### 2. Decision Threshold Optimization
Because high vulnerability represents 33.3% of the sample (a 1:2 class ratio), the default $\tau = 0.50$ threshold artificially suppresses recall.
- Using 5-fold cross-validation on the training set, we mapped the precision-recall-accuracy trade-off curve across $\tau \in [0.20, 0.70]$.
- **Max $F_1$ Threshold ($\tau = 0.285$):** Optimizes positive-case capture, raising Recall to **86.63%**.
- **Balanced Ensemble Threshold ($\tau = 0.400$):** Yields the strongest overall performance: **80.55% accuracy**, **80.21% recall**, and **0.7335 $F_1$**.

### 3. Ordinal Classification Framework (Frank & Hall Method)
To respect the natural ordering $\text{Low} < \text{Medium} < \text{High}$, an ordinal decomposition was implemented using two binary estimators:
$$P(y \ge \text{Medium} \mid X) \quad \text{and} \quad P(y \ge \text{High} \mid X)$$
Enforcing monotonicity ($P(y \ge \text{High}) \le P(y \ge \text{Medium})$) dramatically reduces catastrophic errors:
- Under the ordinal framework, the probability of predicting `Low` when the true class is `High` drops to just **5.6%** (21 out of 373 households).

### 4. Soft-Voting Ensemble (Stacked Probabilistic Fusion)
Combining calibrated predictions from **Logistic Regression**, **Random Forest**, and **XGBoost** via soft voting reduces individual model variance, yielding the highest ROC-AUC (**0.8686**) and the lowest Brier calibration error (**0.1363**).

---

## 4. Visualizations & Diagnostic Plots

All plots are rendered at **300 DPI** and stored in `outputs/stage3_lite/figures/`:
1. **`roc_curve_comparison.png`:** Contrasts ROC curves for XGBoost vs. Soft-Voting Ensemble vs. Naive Baseline.
2. **`threshold_tuning_curve.png`:** Displays the cross-validated trade-off curve between accuracy and $F_1$-score as a function of decision threshold $\tau$.
3. **`confusion_matrices_comparison.png`:** Side-by-side comparison showing how threshold tuning cuts False Negatives from 125 down to 74.

---

## 5. Substantive Research Takeaway

Stage 3 Lite demonstrates that **targeted feature engineering and threshold calibration can push exposure-only prediction past 80% accuracy and 80% recall**. 

However, the fact that accuracy plateaus near **80.5%** (and 3-class accuracy plateaus near **61.8%**) confirms the fundamental scientific finding: **socioeconomic resilience cannot be reverse-engineered from satellite or weather station data alone.** While severe exposure strongly predicts vulnerability, household socioeconomic buffers remain the indispensable factor that prevents moderately exposed households from falling into severe poverty.
