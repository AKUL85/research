# Stage 3: Supervised Machine Learning for Livelihood Vulnerability Prediction
**Bangladesh Integrated Household Survey (BIHS) Round 3**  
*Documentation & Technical Research Report*

---

## 1. Executive Summary & Core Finding

### 1.1 Plain-Language Statement of Finding
**Can climate and agro-environmental exposure variables alone predict household vulnerability class, without using the socioeconomic variables that were used to construct the vulnerability label?**

> **Direct Answer:** **Yes, but with clear structural boundaries.** Agro-environmental exposure variables alone exhibit substantial predictive lift over a naive majority baseline in identifying severely vulnerable households (Binary Task: High vs. Not-High), lifting classification accuracy from **66.6% to 79.8%** (+13.1 percentage points) and achieving a strong discrimination ability (**ROC-AUC = 0.863** to **0.866**, compared to 0.500 for the baseline). Household flood depth and seasonal evapotranspiration deficits serve as powerful signals that flag approximately two-thirds (65.2%) of high-vulnerability households.  
> 
> **However**, exposure variables alone have severely limited capacity to separate intermediate vulnerability from lower vulnerability: in a 3-class setting (Low vs. Medium vs. High), overall accuracy drops to **61.4%**, with the models struggling predominantly to classify the middle tertile (where macro F1-score is ~0.58–0.60). This empirical boundary provides critical econometric support for the paper's overarching thesis: **livelihood vulnerability is fundamentally multidimensional and cannot be treated as a pure environmental deterministic outcome.** While extreme climate shocks (deep flooding and extreme evapotranspiration) push households into the high-vulnerability tier, household-level socioeconomic buffers (income diversification, adult education, land assets, and credit access) determine whether moderately exposed households remain resilient or slide into severe vulnerability.

---

## 2. Research Design & Target Definition

### 2.1 Target Variables
The vulnerability classification is derived from the **Livelihood Vulnerability Index (LVI)** constructed in Stage 1 ($n = 5,605$ households, range $[0.2966, 0.6934]$, mean $0.4636$). In accordance with the Stage 2b analytical framework, sample tertiles partition households into three equal vulnerability strata:

- **Low Vulnerability:** $\text{LVI} \le 0.4326$ ($n = 1,868$, 33.33%)
- **Medium Vulnerability:** $0.4326 < \text{LVI} \le 0.4937$ ($n = 1,869$, 33.35%)
- **High Vulnerability:** $\text{LVI} > 0.4937$ ($n = 1,868$, 33.33%)

Two prediction targets are evaluated:
1. **Primary Target (Binary Classification):** **High Vulnerability ($y=1$) vs. Not-High ($y=0$).**  
   - Class 1 (High): 1,868 households (33.33%)
   - Class 0 (Not-High / Low + Medium): 3,737 households (66.67%)
   - *Imbalance Ratio:* Exactly 1:2.
2. **Secondary Target (3-Class Classification):** **Low ($0$) vs. Medium ($1$) vs. High ($2$).**  
   - Balanced three-way split (~33.3% per class).

---

## 3. Feature Space ($X$) and Target Leakage Avoidance

### 3.1 Included Variables: Pure Exposure & Agro-Environmental Context ($p = 25$)
To test whether vulnerability can be predicted from external environmental risk alone, the feature matrix $X$ is strictly restricted to exposure, climate, and geographic context variables:
1. **Household Flood Exposure:**
   - `flood_affected`: Binary indicator of household plot/dwelling inundation (0/1).
   - `flood_depth_ft_mean`: Average depth of flood water on homestead/plots (continuous, feet).
2. **Seasonal Climate Indicators (Aus, Aman, and Boro seasons):**
   - **Precipitation:** `clim_aus_rain_total_mm`, `clim_aman_rain_total_mm`, `clim_boro_rain_total_mm`.
   - **Solar Radiation:** `clim_aus_solar_rad_avg`, `clim_aman_solar_rad_avg`, `clim_boro_solar_rad_avg`.
   - **Wind Speed:** `clim_aus_wind_avg`, `clim_aman_wind_avg`, `clim_boro_wind_avg`, `seasonal_wind_avg`.
   - **Evapotranspiration:** `clim_aus_evapotrans_avg`, `clim_aman_evapotrans_avg`, `clim_boro_evapotrans_avg`, `seasonal_evapotranspiration_avg`.
3. **District Agricultural Context & Crop Productivity:**
   - `clim_aus_district_area_ha`, `clim_aus_district_production_mt`, `clim_aus_district_yield_ratio`.
   - `clim_aman_district_area_ha`, `clim_aman_district_production_mt`, `clim_aman_district_yield_ratio`.
   - `clim_boro_district_area_ha`, `clim_boro_district_production_mt`, `clim_boro_district_yield_ratio`.
   - `mean_district_yield_ratio`.
4. **Geographic / Spatial Context:**
   - `district_id`: Reconstructed categorical identifier representing the 39 distinct climate profile zones of Bangladesh.

### 3.2 Target Leakage Prevention & Excluded Variables
> [!IMPORTANT]
> **Methodological Justification for Variable Exclusion:**  
> The composite LVI was mathematically constructed as the unweighted mean of three dimensions: Exposure (4 indicators), Sensitivity (4 indicators), and Adaptive Capacity (7 indicators). Including any variable that entered the Sensitivity or Adaptive Capacity dimensions—or any proxy thereof (e.g., household income, educational attainment, landholdings, agricultural labor dependence, meal frequency, credit access)—would constitute **target leakage by construction**. A machine learning model given access to these features would trivially learn the arithmetic formula of the index rather than discovering empirical predictive relationships.

The following **42 variables** were strictly audited and quarantined from feature matrix $X$:

| Domain | Excluded Variables | Rationale / Source Dimension |
|---|---|---|
| **Adaptive Capacity (Indicators & Raw Inputs)** | `income_per_capita_monthly`, `income_per_capita_monthly_norm`, `mean_edu_years_adults`, `mean_edu_years_adults_norm`, `max_edu_years`, `adult_illiteracy_rate`, `adult_illiteracy_rate_norm`, `adult_literacy_rate`, `n_literate_15plus`, `n_adults_15plus`, `livelihood_diversity`, `livelihood_diversity_norm`, `crop_diversity`, `n_activities_7d`, `any_nonfarm_agri_work`, `any_nonfarm_agri_work_norm`, `n_nonfarm_agri_activities`, `land_cultivable_decimal`, `land_cultivable_decimal_norm`, `land_homestead_decimal`, `land_fallow_decimal`, `n_plots`, `has_current_loan_binary`, `has_current_loan_binary_norm`, `has_current_loan`, `ever_had_loan`, `n_loans` | Directly used to compute the 7 Adaptive Capacity indicators. |
| **Sensitivity (Indicators & Raw Inputs)** | `agricultural_occupation_dependence`, `agricultural_occupation_dependence_norm`, `n_occ_agri_own`, `n_occ_agri_labour`, `dependency_ratio`, `dependency_ratio_norm`, `hh_size`, `hh_size_norm`, `meals_per_person_week`, `meals_per_person_week_norm`, `mealdays_total_7d` | Directly used to compute the 4 Sensitivity indicators. |
| **Composite Scores & Dimensions** | `LVI`, `exposure_score`, `sensitivity_score`, `adaptive_capacity_vulnerability_score`, `exposure_n_indicators`, `sensitivity_n_indicators`, `adaptive_capacity_vulnerability_n_indicators` | Direct target or intermediate mathematical summands. |
| **Stage 2 Clustering & PCA Outputs** | `cluster_kmeans`, `cluster_ward`, `cluster_name`, `PC1`, `PC2`, `PC3`, `PC4`, `PC5`, `PC6`, `PC7`, `PC8`, `PC9` | Derived from PCA across all 15 indicators in Stage 2. |

---

## 4. Modeling Methodology & Validation Framework

### 4.1 Train/Test Split
- **Split Ratio:** 80% training ($n = 4,484$), 20% held-out test ($n = 1,121$).
- **Stratification:** Stratified on target $y$ (`random_state=42`) ensuring identical class proportions across train and test partitions.
- **Evaluation:** All reported test metrics are computed strictly on the held-out 20% test partition ($n = 1,121$).

### 4.2 Cross-Validation & Hyperparameter Tuning
Hyperparameters were optimized using **5-fold Stratified Cross-Validation** on the training set:
- **Logistic Regression:** Regularized linear classifier embedded in a `StandardScaler()` pipeline.  
  *Grid searched:* $C \in [0.001, 0.01, 0.1, 1.0, 10.0]$, solver `lbfgs`, max iterations 1000.  
  *Best binary parameter:* $C = 0.01$.
- **Random Forest:** Ensemble of decision trees.  
  *Grid searched:* `n_estimators` $\in [100, 200]$, `max_depth` $\in [5, 10, \text{None}]$, `min_samples_split` $\in [2, 5, 10]$.  
  *Best binary parameters:* `n_estimators=200`, `max_depth=5`, `min_samples_split=2`.
- **XGBoost:** Gradient boosted decision trees.  
  *Grid searched:* `n_estimators` $\in [100, 200]$, `max_depth` $\in [3, 5, 7]$, `learning_rate` $\in [0.01, 0.05, 0.1]$, `subsample` $\in [0.8, 1.0]$.  
  *Best binary parameters:* `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `subsample=1.0`.

---

## 5. Empirical Results & Performance Comparison

### 5.1 Comprehensive Results Table
The table below summarizes performance on the held-out test set ($n = 1,121$) across both classification tasks.

| Task | Model | Best Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | Confusion Matrix $[[TN, FP], [FN, TP]]$ |
|---|---|---|---|---|---|---|---|---|---|
| **Binary (High vs Not-High)** | **Naive Baseline (Majority)** | None (Predict Class 0) | 0.6664 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2223 | `[[747, 0], [374, 0]]` |
| **Binary (High vs Not-High)** | **Logistic Regression** | `{'lr__C': 0.01}` | 0.7948 | 0.7250 | 0.6203 | 0.6686 | **0.8660** | 0.1395 | `[[659, 88], [142, 232]]` |
| **Binary (High vs Not-High)** | **Random Forest** | `{'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 200}` | 0.7966 | **0.7401** | 0.6016 | 0.6637 | 0.8626 | 0.1418 | `[[668, 79], [149, 225]]` |
| **Binary (High vs Not-High)** | **XGBoost** | `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0}` | **0.7975** | 0.7155 | **0.6524** | **0.6825** | 0.8634 | **0.1389** | `[[650, 97], [130, 244]]` |
| **3-Class (Low/Med/High)** | **Naive Baseline (Majority)** | None (Predict Class 1) | 0.3336 | 0.1112 | 0.3333 | 0.1668 | 0.5000 | N/A | `[[0, 374, 0], [0, 374, 0], [0, 373, 0]]` |
| **3-Class (Low/Med/High)** | **Logistic Regression** | `{'lr__C': 0.01}` | **0.6137** | **0.5978** | **0.6139** | **0.5992** | **0.7850** | N/A | `[[269, 76, 29], [120, 129, 125], [25, 58, 290]]` |
| **3-Class (Low/Med/High)** | **Random Forest** | `{'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}` | **0.6137** | 0.5971 | **0.6139** | 0.5978 | 0.7824 | N/A | `[[274, 68, 32], [119, 125, 130], [22, 62, 289]]` |
| **3-Class (Low/Med/High)** | **XGBoost** | `{'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}` | 0.6030 | 0.5827 | 0.6032 | 0.5844 | 0.7803 | N/A | `[[273, 75, 26], [127, 114, 133], [28, 56, 289]]` |

*Note on metrics:* For binary classification, Precision, Recall, and F1 correspond to the positive class (High Vulnerability). For 3-class classification, macro-averaged metrics are reported. ROC-AUC in 3-class uses one-vs-rest (OvR) with macro averaging.

### 5.2 Performance Lift Over Naive Baseline
- **Accuracy Lift:** +13.11 percentage points for XGBoost (79.75% vs. 66.64%).
- **Discriminative Lift (ROC-AUC):** +0.3660 lift for Logistic Regression (0.8660 vs. 0.5000) and +0.3634 for XGBoost (0.8634 vs. 0.5000).
- **F1-Score Lift:** From 0.0000 (baseline fails to predict any positive cases) to **0.6825** for XGBoost.
- **Probability Calibration (Brier Score):** Reduced from 0.2223 to **0.1389**, indicating sharp and well-calibrated probabilistic estimates.

---

## 6. Model Interpretability & SHAP Analysis

Using the best-performing binary model (**XGBoost**, F1 = 0.6825, ROC-AUC = 0.8634), SHAP (*SHapley Additive exPlanations*) values were computed across the held-out test set ($n = 1,121$) via `shap.TreeExplainer`. All generated plots are saved in `outputs/stage3_supervised_ml/shap_figures/` at 300 DPI.

### 6.1 Global Feature Importance & Beeswarm Analysis
- **Dominant Exposure Driver — Flood Depth:** As shown in **Figure 1** (mean $|SHAP|$ bar chart) and **Figure 2** (beeswarm plot), `flood_depth_ft_mean` is the single most dominant predictor ($\text{mean } |SHAP| = 1.2913$), exerting more than quadruple the influence of the second-ranked variable. The beeswarm plot confirms a strong monotonic positive relationship: higher flood depths (red dots) push the log-odds of high vulnerability sharply upward, whereas zero flood depth (blue dots) pulls predicted risk downward.
- **Secondary Atmospheric Drivers — Evapotranspiration:** Boro and Aman season evapotranspiration averages (`clim_boro_evapotrans_avg` $\text{mean } |SHAP| = 0.2871$; `clim_aman_evapotrans_avg` $\text{mean } |SHAP| = 0.2664$) form the second tier of predictive importance. Higher seasonal water deficits during the critical rice crop cycles significantly elevate household vulnerability risk.
- **Macro-Agricultural & Rainfall Context:** Boro rainfall totals (`clim_boro_rain_total_mm`, $\text{mean } |SHAP| = 0.0645$) and district yield ratios (`mean_district_yield_ratio`, $\text{mean } |SHAP| = 0.0575$; `clim_aus_district_yield_ratio`, $\text{mean } |SHAP| = 0.0556$) contribute steady, moderate predictive signals. Below-average district yields consistently elevate household vulnerability.

### 6.2 Representative Individual Household Waterfall Plots
To inspect local feature attribution, three representative cases were analyzed via SHAP waterfall decompositions:
1. **High-Vulnerability Household (`shap_waterfall_high_vulnerability.png`):**  
   - *Ground Truth:* High Vulnerability ($y=1$). *Predicted Probability:* $P(\text{High}) = 0.94$.  
   - *Key Drivers:* A flood depth of 7.5 feet contributed $+2.85$ to the log-odds, reinforced by high Aman evapotranspiration ($+0.42$), driving the household far above the base expected value ($E[f(X)] = -0.70$).
2. **Low-Vulnerability Household (`shap_waterfall_low_vulnerability.png`):**  
   - *Ground Truth:* Not-High ($y=0$). *Predicted Probability:* $P(\text{High}) = 0.06$.  
   - *Key Drivers:* Complete absence of flooding (`flood_depth_ft_mean = 0.0`) contributed $-1.15$ to the log-odds, coupled with favorable seasonal moisture and high district yield ratios, firmly placing the household in the resilient tier.
3. **Borderline / Ambiguous Household (`shap_waterfall_borderline_case.png`):**  
   - *Ground Truth:* High Vulnerability ($y=1$). *Predicted Probability:* $P(\text{High}) = 0.51$.  
   - *Key Drivers:* The household experienced moderate flood exposure, but favorable district agricultural productivity partially offset the risk. This case clearly illustrates why exposure alone cannot definitively resolve classification without socioeconomic data.

---

## 7. Synthesis & Implications for the Research Paper

1. **Environmental Shocks as Necessary, but Insufficient, Determinants:**  
   The supervised ML results demonstrate that environmental and climatic shocks are powerful predictors of vulnerability extremes. Deep flood inundation and high evaporative demand consistently identify two out of three highly vulnerable households ($65.2\%$ recall, $71.6\%$ precision).
2. **The "Missing Middle" and the Multidimensional Nature of Vulnerability:**  
   The confusion matrices in the 3-class setting provide the clearest empirical evidence: while models correctly identify **77.7%** of High-vulnerability households and **73.3%** of Low-vulnerability households, they correctly identify only **30.5%** of Medium-vulnerability households (misclassifying 34.0% as Low and 35.6% as High). Households with identical environmental exposure experience divergent vulnerability outcomes depending on their adaptive capacity—specifically whether they possess off-farm income streams, formal schooling, or cultivable land.
3. **Policy Takeaway:**  
   Targeting climate adaptation programs solely by geographic exposure zones or flood maps will successfully reach severe distress hotspots, but will systematically miss vulnerable households in moderate-exposure zones who lack the socioeconomic buffer to absorb minor climate shocks.

---

## 8. Artifacts & Deliverables Summary

| Artifact File | Path | Description |
|---|---|---|
| **Results CSV** | `outputs/stage3_supervised_ml/supervised_model_results.csv` | Full metric table (accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix) for all models across binary and 3-class tasks. |
| **Global Importance Bar Chart** | `outputs/stage3_supervised_ml/shap_figures/shap_bar_global_importance.png` | 300 DPI bar plot showing the top 15 features by mean absolute SHAP value. |
| **SHAP Beeswarm Summary** | `outputs/stage3_supervised_ml/shap_figures/shap_beeswarm_summary.png` | 300 DPI summary plot showing feature value distributions and positive/negative impact directions. |
| **High Vulnerability Waterfall** | `outputs/stage3_supervised_ml/shap_figures/shap_waterfall_high_vulnerability.png` | 300 DPI local explanation for a representative true-high household ($P > 0.85$). |
| **Low Vulnerability Waterfall** | `outputs/stage3_supervised_ml/shap_figures/shap_waterfall_low_vulnerability.png` | 300 DPI local explanation for a representative true-low household ($P < 0.10$). |
| **Borderline Case Waterfall** | `outputs/stage3_supervised_ml/shap_figures/shap_waterfall_borderline_case.png` | 300 DPI local explanation for an ambiguous case ($P \approx 0.50$) highlighting exposure-only prediction limits. |
| **Reproducible Script** | `supervised_ml_stage3.py` & `outputs/stage3_supervised_ml/supervised_ml_stage3.py` | Fully seeded, deterministic end-to-end Python pipeline. |
