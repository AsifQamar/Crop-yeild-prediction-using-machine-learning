"""
Crop Yield Prediction - Model Training Pipeline
=================================================
1. Data Cleaning & Preprocessing
2. EDA (saves plots to eda_plots/)
3. Model Training & Comparison (XGBoost, Random Forest, KNN)
4. Best model selection & saving
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR = os.path.join(BASE_DIR, 'eda_plots')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading & Cleaning Data")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_DIR, 'Crop Yeild Data.csv'))
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Drop 'Production' column - the target Yield = Production / Area
# Keeping it would let the model cheat by learning the formula
df.drop(columns=['Production'], inplace=True)
print("[OK] Dropped 'Production' column (data leakage prevention)")

# Strip whitespace from string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
print("[OK] Stripped whitespace from categorical columns")

# Check for and drop duplicates
dup_count = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"[OK] Removed {dup_count} duplicate rows")

# Check for nulls
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print(f"  Null values found:\n{null_counts[null_counts > 0]}")
    df.dropna(inplace=True)
    print("[OK] Dropped rows with null values")
else:
    print("[OK] No null values found")

# Remove outliers in Yield (beyond 3 standard deviations)
mean_yield = df['Yield'].mean()
std_yield = df['Yield'].std()
before = len(df)
df = df[(df['Yield'] >= mean_yield - 3 * std_yield) & (df['Yield'] <= mean_yield + 3 * std_yield)]
print(f"[OK] Removed {before - len(df)} yield outliers (3-sigma rule)")

print(f"Cleaned shape: {df.shape}")
print()

# ------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
print("=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

sns.set_theme(style='whitegrid', palette='muted')
plot_colors = ['#6BAF6B', '#F5E6CA', '#4A8C4A', '#D4A853', '#2E7D32']

# 2a. Yield Distribution
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['Yield'], bins=50, color=plot_colors[0], kde=True, ax=ax)
ax.set_title('Distribution of Crop Yield', fontsize=14, fontweight='bold')
ax.set_xlabel('Yield')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'yield_distribution.png'), dpi=150)
plt.close()
print("[OK] Saved yield_distribution.png")

# 2b. Top 15 crops by average yield
fig, ax = plt.subplots(figsize=(12, 6))
top_crops = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(15)
top_crops.plot(kind='barh', color=plot_colors[0], ax=ax)
ax.set_title('Top 15 Crops by Average Yield', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Yield')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'top_crops_yield.png'), dpi=150)
plt.close()
print("[OK] Saved top_crops_yield.png")

# 2c. Season-wise yield
fig, ax = plt.subplots(figsize=(10, 5))
season_yield = df.groupby('Season')['Yield'].mean().sort_values(ascending=False)
season_yield.plot(kind='bar', color=plot_colors[2], ax=ax)
ax.set_title('Average Yield by Season', fontsize=14, fontweight='bold')
ax.set_xlabel('Season')
ax.set_ylabel('Average Yield')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'season_yield.png'), dpi=150)
plt.close()
print("[OK] Saved season_yield.png")

# 2d. Correlation heatmap (numeric columns)
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGn', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("[OK] Saved correlation_heatmap.png")

# 2e. State-wise average yield (top 15)
fig, ax = plt.subplots(figsize=(12, 6))
state_yield = df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(15)
state_yield.plot(kind='barh', color=plot_colors[3], ax=ax)
ax.set_title('Top 15 States by Average Yield', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Yield')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'state_yield.png'), dpi=150)
plt.close()
print("[OK] Saved state_yield.png")

# 2f. Yield vs key features scatter
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
scatter_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Avg_Temperature']
for idx, feat in enumerate(scatter_features):
    ax = axes[idx // 2][idx % 2]
    ax.scatter(df[feat], df['Yield'], alpha=0.3, s=8, color=plot_colors[0])
    ax.set_xlabel(feat)
    ax.set_ylabel('Yield')
    ax.set_title(f'Yield vs {feat}', fontweight='bold')
plt.suptitle('Yield vs Key Features', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'yield_vs_features.png'), dpi=150)
plt.close()
print("[OK] Saved yield_vs_features.png")

# 2g. Yearly trend
fig, ax = plt.subplots(figsize=(12, 5))
yearly = df.groupby('Crop_Year')['Yield'].mean()
ax.plot(yearly.index, yearly.values, marker='o', color=plot_colors[2], linewidth=2)
ax.set_title('Average Yield Over Years', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Average Yield')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'yearly_trend.png'), dpi=150)
plt.close()
print("[OK] Saved yearly_trend.png")

print()

# ------------------------------------------------------------
# 3. FEATURE ENGINEERING & PREPROCESSING
# ------------------------------------------------------------
print("=" * 60)
print("STEP 3: Feature Engineering & Preprocessing")
print("=" * 60)

# Label encode categorical columns
label_encoders = {}
categorical_cols = ['Crop', 'Season', 'State']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"[OK] Label-encoded '{col}' ({len(le.classes_)} classes)")

# Save label encoders
joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
print("[OK] Saved label encoders")

# Save the lists of unique values for the frontend dropdowns
dropdown_data = {}
for col in categorical_cols:
    dropdown_data[col] = list(label_encoders[col].classes_)
with open(os.path.join(MODEL_DIR, 'dropdown_data.json'), 'w') as f:
    json.dump(dropdown_data, f)
print("[OK] Saved dropdown data for frontend")

# Features and target
feature_cols = ['Crop', 'Crop_Year', 'Season', 'State', 'Area',
                'Annual_Rainfall', 'Fertilizer', 'Pesticide',
                'Avg_Temperature', 'Max_Temperature', 'Min_Temperature']
X = df[feature_cols]
y = df['Yield']

print(f"\nFeatures: {feature_cols}")
print(f"Target: Yield")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
print("[OK] Fitted and saved StandardScaler")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"[OK] Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}")
print()

# Save feature columns for inference
joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.pkl'))

# ------------------------------------------------------------
# 4. MODEL TRAINING & COMPARISON
# ------------------------------------------------------------
print("=" * 60)
print("STEP 4: Training & Comparing Models")
print("=" * 60)

models = {
    'XGBoost': XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'KNN': KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n[>>] Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {
        'R2_Score': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'model': model
    }
    print(f"  [OK] {name} -> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# ------------------------------------------------------------
# 5. BEST MODEL SELECTION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Best Model Selection")
print("=" * 60)

# Choose the model with the highest R2 score
best_name = max(results, key=lambda k: results[k]['R2_Score'])
best_model = results[best_name]['model']

print(f"\n[BEST] Best Model: {best_name}")
print(f"   R2 Score : {results[best_name]['R2_Score']}")
print(f"   RMSE     : {results[best_name]['RMSE']}")
print(f"   MAE      : {results[best_name]['MAE']}")

# Save the best model
joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
joblib.dump(best_name, os.path.join(MODEL_DIR, 'best_model_name.pkl'))
print(f"[OK] Saved best model ({best_name}) to models/best_model.pkl")

# Save comparison results (without the model objects)
comparison = {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()}
comparison['best_model'] = best_name
with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'w') as f:
    json.dump(comparison, f, indent=2)
print("[OK] Saved model comparison to models/model_comparison.json")

# 5b. Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(results.keys())
metrics = ['R2_Score', 'RMSE', 'MAE']
colors_map = {'XGBoost': '#2E7D32', 'Random Forest': '#6BAF6B', 'KNN': '#D4A853'}

for idx, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    bars = axes[idx].bar(model_names, vals, color=[colors_map[m] for m in model_names])
    axes[idx].set_title(metric, fontsize=13, fontweight='bold')
    axes[idx].set_ylabel(metric)
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Model Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'model_comparison.png'), dpi=150)
plt.close()
print("[OK] Saved model_comparison.png")

print("\n" + "=" * 60)
print("[OK] TRAINING PIPELINE COMPLETE")
print("=" * 60)
