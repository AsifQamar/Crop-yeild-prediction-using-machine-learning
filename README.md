# Crop Yield Prediction Using Machine Learning

---

## Chapter 1: Introduction

### 1.1 Introduction and Problem Statement

Agriculture is the backbone of the Indian economy, contributing significantly to GDP and employing a large portion of the population. However, crop yield remains unpredictable due to its dependence on numerous factors such as weather conditions, soil quality, fertilizer usage, pesticide application, and geographic location. Farmers and policymakers often lack data-driven tools to estimate yield in advance, leading to poor planning, resource wastage, and financial losses.

The problem this project addresses is: **How can we accurately predict crop yield given a set of agricultural and climatic parameters?**

Traditional methods of yield estimation rely on manual observation and historical averages, which fail to capture complex, non-linear interactions between input variables. Machine learning offers a way to model these relationships from historical data and provide actionable predictions.

### 1.2 Objectives of the Project

1. Clean and preprocess a real-world Indian crop yield dataset, removing data leakage (the `Production` column) and handling outliers.
2. Perform exploratory data analysis (EDA) to uncover patterns across crops, states, seasons, and climatic factors.
3. Train and evaluate three machine learning models — XGBoost, Random Forest, and K-Nearest Neighbors (KNN).
4. Build an automated model comparison algorithm that selects the best model based on R² Score, RMSE, and MAE.
5. Deploy the best model as a web application using Flask (backend) and HTML/CSS/JS (frontend) so users can input parameters and receive yield predictions.

### 1.3 Scope of the Project

- **Geographic coverage:** 30 Indian states.
- **Temporal coverage:** Crop data from 1997 to 2020.
- **Crop coverage:** 55 crop types across 6 seasons (Kharif, Rabi, Whole Year, Summer, Winter, Autumn).
- **Input features:** Area, annual rainfall, fertilizer, pesticide, average/max/min temperature, crop type, state, season, and year.
- **Output:** Predicted yield in tonnes per hectare.
- **Deployment:** Local Flask web application with a form-based interface.

The project does not cover real-time weather integration, satellite imagery, or soil-level microdata, which are areas for future enhancement.

---

## Chapter 2: Literature Survey

### 2.1 Review of Existing Work

| # | Authors / Source | Method | Key Findings |
|---|---|---|---|
| 1 | Liakos et al. (2018), *Sensors* | Review of ML in agriculture | Identified Random Forest, SVM, and ANN as the most widely used ML models for crop yield prediction. Highlighted the need for region-specific models. |
| 2 | Khaki & Wang (2019), *Frontiers in Plant Science* | Deep Neural Networks | Used county-level data in the US with weather and soil features. Deep learning outperformed traditional regression but required large datasets and compute. |
| 3 | Crane-Droesch (2018), *Environmental Research Letters* | Semi-parametric deep neural network | Combined econometric models with deep learning for US corn yield. Showed that non-linear models capture weather interactions better than linear approaches. |
| 4 | Jeong et al. (2016), *Remote Sensing* | Random Forest with satellite data | Used MODIS NDVI and climate variables. Random Forest achieved strong results but performance was tied to satellite data availability. |
| 5 | Gonzalez-Sanchez et al. (2014), *IJACSA* | Comparison of ML algorithms | Compared SVM, ANN, KNN, and regression trees on Mexican crop data. Found ensemble methods generally outperformed single models. |
| 6 | Rashid et al. (2021), *IEEE Access* | XGBoost for wheat yield | Applied XGBoost on Pakistani wheat data with weather features. Achieved R² > 0.85, demonstrating XGBoost's strength on tabular agricultural data. |

### 2.2 Summary of Literature Survey

The literature consistently shows that:
- **Ensemble tree-based methods** (Random Forest, XGBoost, Gradient Boosting) tend to outperform simple regression and distance-based methods (KNN) on structured agricultural data.
- Feature engineering — particularly removing leaky features and including climatic variables — is critical for model generalization.
- Most studies focus on single crops or single regions; multi-crop, multi-state models are less common.
- Web-based deployment of yield prediction models is rarely addressed in academic work, despite its practical value.

### 2.3 Research Gap

1. **Data leakage in existing projects:** Many publicly available crop yield prediction projects on GitHub include the `Production` column as a feature. Since `Yield = Production / Area`, this causes the model to memorize a formula rather than learn real patterns — producing misleadingly high accuracy that fails in practice.
2. **Lack of model comparison:** Most implementations train a single model without systematic comparison across algorithms using standardized metrics.
3. **No end-to-end deployment:** Academic papers rarely provide a usable interface. Farmers, students, and analysts need accessible tools, not just notebooks.

This project addresses all three gaps.

---

## Chapter 3: Dataset and Proposed Methodology

### 3.1 Dataset Description / Input Data

**Source:** Indian Crop Yield Dataset (CSV, ~2 MB)

| Column | Type | Description |
|---|---|---|
| Crop | Categorical | Name of the crop (55 unique types) |
| Crop_Year | Integer | Year of harvest (1997–2020) |
| Season | Categorical | Growing season (Kharif, Rabi, Whole Year, Summer, Winter, Autumn) |
| State | Categorical | Indian state (30 unique states) |
| Area | Float | Cultivated area in hectares |
| Production | Integer | Total production in tonnes — **removed (leakage)** |
| Annual_Rainfall | Float | Rainfall in millimeters |
| Fertilizer | Float | Fertilizer used in kilograms |
| Pesticide | Float | Pesticide used in kilograms |
| Yield | Float | **Target variable** — yield in tonnes/hectare |
| Avg_Temperature | Float | Average annual temperature in °C |
| Max_Temperature | Float | Maximum temperature in °C |
| Min_Temperature | Float | Minimum temperature in °C |

- **Total records:** 19,689 rows, 13 columns.
- **No null values** were found in the dataset.

### 3.2 Data Preprocessing / Input Preparation

The following preprocessing steps were applied:

1. **Production column dropped:** Since `Yield = Production / Area`, keeping `Production` as a feature would allow the model to reverse-engineer the target from a simple division, resulting in data leakage. This column was removed entirely.

2. **Whitespace stripping:** Categorical columns (`Crop`, `Season`, `State`) contained trailing whitespace (e.g., `"Kharif     "`). All string values were stripped.

3. **Duplicate removal:** Checked for and removed duplicate rows (0 found).

4. **Outlier removal:** Yield values beyond 3 standard deviations from the mean were removed, eliminating 164 extreme outlier rows. Final cleaned dataset: **19,525 rows, 12 columns**.

5. **Label encoding:** Categorical features (`Crop`, `Season`, `State`) were transformed to integer codes using `LabelEncoder` from scikit-learn.

6. **Feature scaling:** All 11 input features were standardized using `StandardScaler` (zero mean, unit variance) to ensure distance-based models (KNN) and gradient-based models (XGBoost) perform optimally.

7. **Train-test split:** 80% training (15,620 samples), 20% testing (3,905 samples), with `random_state=42` for reproducibility.

### 3.3 Algorithm / Model / Core Technique

Three regression algorithms were implemented and compared:

**1. XGBoost (Extreme Gradient Boosting)**
- Ensemble of decision trees trained sequentially, where each tree corrects the errors of the previous one.
- Hyperparameters: `n_estimators=300`, `max_depth=7`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.
- Strengths: Handles non-linear relationships well, built-in regularization, fast training.

**2. Random Forest**
- Ensemble of independently trained decision trees with bootstrap aggregation (bagging).
- Hyperparameters: `n_estimators=200`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`.
- Strengths: Robust to overfitting, handles mixed feature types, interpretable feature importance.

**3. K-Nearest Neighbors (KNN)**
- Instance-based learning — predicts yield as the weighted average of the K nearest training samples in feature space.
- Hyperparameters: `n_neighbors=7`, `weights='distance'`.
- Strengths: Simple, no training phase. Weakness: Sensitive to feature scaling and high-dimensional data.

**Automatic Model Selection Algorithm:**
After training all three models, the system compares them on R² Score, RMSE, and MAE. The model with the **highest R² Score** is automatically selected, saved to disk, and used for serving predictions.

### 3.4 Development, Testing and Validation

- **Development:** Python 3.12, scikit-learn, XGBoost, pandas, NumPy.
- **Testing:** The 20% held-out test set (3,905 samples) was used for evaluation. No test data was seen during training.
- **Validation:** Three standard regression metrics were computed on the test set:
  - **R² Score** — proportion of variance explained (higher is better, max = 1.0).
  - **RMSE** — root mean squared error in the same unit as yield (lower is better).
  - **MAE** — mean absolute error (lower is better).
- **Reproducibility:** All random operations use `random_state=42`.

### 3.5 Proposed Solution and System Workflow

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Raw CSV    │────▶│  Data Cleaning   │────▶│  Label Encoding  │
│   Dataset    │     │  & Preprocessing │     │  & Scaling       │
└──────────────┘     └──────────────────┘     └──────────────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │   Train 3 Models │
                                              │  XGBoost / RF /  │
                                              │       KNN        │
                                              └──────────────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Compare Metrics  │
                                              │  R², RMSE, MAE   │
                                              │ Select Best Model│
                                              └──────────────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  User enters │────▶│  Flask Backend   │────▶│  Return Yield    │
│  crop params │     │  (predict API)   │     │  (tonnes/hectare)│
└──────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Chapter 4: Implementation and Results

### 4.1 Development Environment and Tools

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Libraries | scikit-learn 1.6.1, XGBoost 3.4.1 |
| Data Libraries | pandas 2.2.3, NumPy 2.2.4 |
| Visualization | matplotlib 3.10.8, seaborn 0.13.2 |
| Backend | Flask 3.1.0 |
| Frontend | HTML5, CSS3, JavaScript (Vanilla) |
| Model Serialization | joblib 1.4.2 |
| OS | Windows |

### 4.2 Implementation

The project consists of the following files:

```
Crop_yeild_prediction/
├── Crop Yeild Data.csv          # Raw dataset
├── train_model.py               # Full ML pipeline (clean → EDA → train → compare → save)
├── app.py                       # Flask web server
├── templates/
│   └── index.html               # Frontend HTML
├── static/
│   ├── css/style.css            # Styling (light green + cream theme)
│   └── js/app.js                # Client-side logic
├── models/
│   ├── best_model.pkl           # Saved best model (XGBoost)
│   ├── best_model_name.pkl      # Name of the best model
│   ├── scaler.pkl               # Fitted StandardScaler
│   ├── label_encoders.pkl       # Fitted LabelEncoders
│   ├── feature_cols.pkl         # Ordered feature column names
│   ├── model_comparison.json    # Metrics for all 3 models
│   └── dropdown_data.json       # Crop/Season/State lists for UI
└── eda_plots/
    ├── yield_distribution.png
    ├── top_crops_yield.png
    ├── season_yield.png
    ├── state_yield.png
    ├── correlation_heatmap.png
    ├── yield_vs_features.png
    ├── yearly_trend.png
    └── model_comparison.png
```

**Key implementation details:**

- `train_model.py` runs the entire pipeline end-to-end: load → clean → EDA plots → encode → scale → split → train 3 models → compare → save best.
- `app.py` loads the saved model and artifacts at startup. It exposes REST APIs (`/api/predict`, `/api/dropdown-data`, `/api/model-info`, `/api/eda-plots`) and serves the frontend.
- The frontend has three tabs: **Predict** (input form), **Models** (comparison table and chart), and **Analysis** (EDA visualizations).

### 4.3 Experimental Results

All three models were trained on 15,620 samples and evaluated on 3,905 held-out test samples.

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **XGBoost** | **0.8533** | **4.5914** | **1.3242** |
| Random Forest | 0.8282 | 4.9683 | 1.3646 |
| KNN | 0.4586 | 8.8201 | 3.4515 |

**XGBoost** was automatically selected as the best model with the highest R² Score of 0.8533.

### 4.4 Performance Evaluation Metrics

Three standard regression metrics were used:

**R² Score (Coefficient of Determination):**
Measures the proportion of variance in yield that the model explains. A score of 0.8533 means XGBoost explains ~85.3% of the variance. A perfect model scores 1.0.

**RMSE (Root Mean Squared Error):**
The square root of the average squared difference between predicted and actual values. XGBoost achieved an RMSE of 4.5914 tonnes/hectare, meaning predictions are off by ~4.59 tonnes/hectare on average (penalizing large errors more).

**MAE (Mean Absolute Error):**
The average of absolute differences between predicted and actual values. XGBoost achieved an MAE of 1.3242 tonnes/hectare — the typical prediction error magnitude.

### 4.5 Results and Discussion

**Key observations:**

1. **XGBoost outperformed both alternatives.** Its sequential error-correction mechanism (boosting) captured complex interactions between features like crop type, region, rainfall, and fertilizer usage that Random Forest's independent trees could not fully exploit.

2. **Random Forest performed comparably** (R² = 0.8282), confirming that ensemble tree methods are well-suited for this type of structured agricultural data. The gap between XGBoost and Random Forest is modest (~2.5%), suggesting Random Forest is a viable alternative when interpretability is prioritized.

3. **KNN performed poorly** (R² = 0.4586). This is expected because KNN struggles with:
   - High-dimensional data (11 features after encoding).
   - Mixed feature types (encoded categoricals alongside continuous variables).
   - The distance metric treats all features equally, which may not reflect their true importance to yield.

4. **Removing the Production column was critical.** Without this step, all models would achieve R² > 0.99 by simply learning `Yield ≈ Production / Area` — a formula, not a pattern. By removing it, the models were forced to learn genuine relationships between agricultural inputs and yield.

5. **EDA insights:**
   - Coconut, Sugarcane, and Tapioca have the highest average yields.
   - Yield has gradually improved over the 1997–2020 period.
   - Rainfall and temperature show moderate correlation with yield.
   - Fertilizer usage shows weak direct correlation, suggesting its effect is crop-specific.

---

## Chapter 5: Conclusion and Future Scope

### 5.1 Conclusion

This project successfully built an end-to-end crop yield prediction system that:

- Cleaned and preprocessed a real-world Indian agricultural dataset of 19,689 records spanning 55 crops across 30 states over 24 years.
- Identified and removed the `Production` column to eliminate data leakage — a flaw present in many existing implementations.
- Generated 7 EDA visualizations revealing patterns in crop yield across regions, seasons, and years.
- Trained and compared three machine learning models (XGBoost, Random Forest, KNN) using a standardized evaluation pipeline.
- Automatically selected XGBoost as the best model (R² = 0.8533, RMSE = 4.5914, MAE = 1.3242).
- Deployed the model as a web application with a clean, minimalist interface where users can input crop parameters and receive yield predictions in tonnes per hectare.

The system demonstrates that gradient-boosted decision trees can effectively predict crop yield from standard agricultural and climatic variables, and that careful data preprocessing (particularly leakage removal) is essential for building trustworthy models.

### 5.2 Future Scope

1. **Real-time weather integration:** Connect to weather APIs (e.g., OpenWeatherMap, IMD) to auto-fill temperature and rainfall fields based on the selected state and year.
2. **Soil data incorporation:** Include soil type, pH, organic carbon, and nutrient levels as additional features for more accurate predictions.
3. **Satellite imagery:** Use NDVI (Normalized Difference Vegetation Index) from satellite data to capture real-time crop health.
4. **Deep learning models:** Experiment with LSTMs or Transformer-based models that can capture temporal patterns across multiple growing seasons.
5. **Hyperparameter tuning:** Apply Bayesian optimization or grid search to further improve model performance.
6. **Multi-output prediction:** Extend the system to predict both yield and optimal resource allocation (fertilizer, pesticide) simultaneously.
7. **Mobile deployment:** Package the Flask app as a Progressive Web App (PWA) or native mobile app for field use by farmers.
8. **Regional model specialization:** Train separate models per state or agro-climatic zone to capture local patterns more accurately.

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn flask joblib

# 2. Train the models (generates EDA plots + saves best model)
python train_model.py

# 3. Start the web application
python app.py

# 4. Open in browser
# http://127.0.0.1:5000
```

