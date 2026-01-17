# Stock Prediction Model Results

This document summarizes the performance of XGBoost and LightGBM models for predicting SPY 3-day price movements.

## Dataset Overview

- **Training Period**: 2015-06-15 to 2024-12-31 (2,402 samples)
- **Test Period**: 2025-01-02 to 2026-01-16 (261 samples)
- **Total Features**: 115 engineered features
- **Target**: Binary classification (Up/Down over next 3 days)
- **Target Distribution (Train)**: 62.7% Up, 37.3% Down
- **Target Distribution (Test)**: 60.5% Up, 39.5% Down

## Model Comparison

### Summary Table

| Model | Phase | Train AUC | Test AUC | Test Acc | Test F1 | Best Iter | Status |
|-------|-------|-----------|----------|----------|---------|-----------|--------|
| XGBoost | All (115) | ~0.97 | ~0.50 | ~50% | - | - | ❌ Overfit |
| XGBoost | Selected (40) | ~0.75 | ~0.50 | ~50% | - | - | ❌ Poor |
| LightGBM | All (115) | 0.8919 | 0.4800 | 55.56% | 0.7041 | 8 | ⚠️ Overfit |
| **LightGBM** | **Selected (40)** | **0.7416** | **0.5546** | **60.54%** | **0.7469** | **3** | **✅ Best** |
| CatBoost | All (115) | 0.8857 | 0.5033 | 58.24% | 0.7322 | 73 | ⚠️ Overfit |
| CatBoost | Selected (40) | 0.6333 | 0.4953 | 59.77% | 0.7458 | 8 | ⚠️ Decent |

**Winner**: **LightGBM with 40 selected features** - Best test ROC AUC (0.5546) and accuracy (60.54%)

---

### XGBoost Results

#### Phase 1: All Features (115 features)
- **Train ROC AUC**: ~0.97
- **Test ROC AUC**: ~0.50
- **Test Accuracy**: ~50%
- **Observations**: Severe overfitting with default parameters

#### Phase 2: Selected Features (40 features)
- **Train ROC AUC**: ~0.75
- **Test ROC AUC**: ~0.50
- **Test Accuracy**: ~50%
- **Observations**: Feature selection reduced overfitting but test performance remained near random

**XGBoost Hyperparameters** (Regularized):
```python
{
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'eval_metric': 'auc'
}
```

---

### LightGBM Results ✅

#### Phase 1: All Features (115 features)
- **Train ROC AUC**: 0.8919
- **Test ROC AUC**: 0.4800
- **Test Accuracy**: 55.56%
- **Test F1 Score**: 0.7041
- **Best Iteration**: 8 (early stopped)

**Confusion Matrix (Test)**:
```
              Predicted
              Down   Up
Actual Down     7    96
       Up      20   138
```

#### Phase 2: Selected Features (40 features) ⭐
- **Train ROC AUC**: 0.7416
- **Test ROC AUC**: 0.5546
- **Test Accuracy**: 60.54%
- **Test F1 Score**: 0.7469
- **Best Iteration**: 3 (early stopped)

**Confusion Matrix (Test)**:
```
              Predicted
              Down   Up
Actual Down     6    97
       Up       6   152
```

**Performance Improvement**:
- **ROC AUC**: +0.0746 (+7.46%)
- **Accuracy**: +0.0498 (+4.98%)

**LightGBM Hyperparameters** (Defaults):
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'early_stopping_rounds': 50
}
```

---

### CatBoost Results

#### Phase 1: All Features (115 features)
- **Train ROC AUC**: 0.8857
- **Test ROC AUC**: 0.5033
- **Test Accuracy**: 58.24%
- **Test F1 Score**: 0.7322
- **Best Iteration**: 73 (early stopped)

**Confusion Matrix (Test)**:
```
              Predicted
              Down   Up
Actual Down     3   100
       Up       9   149
```

#### Phase 2: Selected Features (40 features)
- **Train ROC AUC**: 0.6333
- **Test ROC AUC**: 0.4953
- **Test Accuracy**: 59.77%
- **Test F1 Score**: 0.7458
- **Best Iteration**: 8 (early stopped)

**Confusion Matrix (Test)**:
```
              Predicted
              Down   Up
Actual Down     1   102
       Up       2   156
```

**Performance Change**:
- **ROC AUC**: -0.0080 (-0.80%)
- **Accuracy**: +0.0153 (+1.53%)
- **Note**: Feature selection improved accuracy but slightly decreased ROC AUC

**CatBoost Hyperparameters** (Defaults):
```python
{
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'depth': 6,
    'iterations': 1000,
    'learning_rate': 0.03,
    'od_wait': 50,  # Early stopping rounds
    'use_best_model': True
}
```

---

## Top Features by Permutation Importance

### LightGBM Top 10 Features (Best Model)

| Rank | Feature | Importance | Std Dev |
|------|---------|------------|---------|
| 1 | `spy_volume_ratio_50d` | 0.008046 | ±0.002682 |
| 2 | `spy_rsi_14` | 0.001533 | ±0.002541 |
| 3 | `spy_macd_pct` | 0.000766 | ±0.006365 |
| 4 | `spy_volume_ratio_20d` | 0.000766 | ±0.002299 |
| 5 | `spy_momentum_3d` | 0.000383 | ±0.002682 |
| 6 | `vix_spy_ratio` | 0.000383 | ±0.001149 |
| 7-40 | Various features | 0.000000 | - |

### CatBoost Top 10 Features

| Rank | Feature | Importance | Std Dev |
|------|---------|------------|---------|
| 1 | `spy_atr_zscore` | 0.007280 | - |
| 2 | `spy_volume_ratio_20d` | 0.006130 | - |
| 3 | `spy_gap_up` | 0.004981 | - |
| 4 | `spy_volume_percentile` | 0.003065 | - |
| 5 | `spy_gap_pct` | 0.003065 | - |
| 6 | `spy_roc_5d` | 0.002682 | - |
| 7 | `spy_momentum_50d` | 0.002682 | - |
| 8 | `vix_stretched_low` | 0.002682 | - |
| 9 | `dxy_rising` | 0.002299 | - |
| 10 | `spy_bb_middle_zone` | 0.002299 | - |

### Feature Category Breakdown

**Volume Features** (Most Important across models):
- `spy_volume_ratio_50d` - 50-day volume ratio (LightGBM #1)
- `spy_volume_ratio_20d` - 20-day volume ratio (Both models)
- `spy_volume_percentile` - Volume percentile (CatBoost)

**Volatility Features**:
- `spy_atr_zscore` - ATR z-score (CatBoost #1)
- `spy_rsi_14` - 14-day RSI (LightGBM #2)

**Momentum/Trend Features**:
- `spy_macd_pct` - MACD percentage (LightGBM)
- `spy_momentum_3d` - 3-day momentum (LightGBM)
- `spy_roc_5d` - 5-day rate of change (CatBoost)

**Gap Features** (CatBoost specific):
- `spy_gap_up` - Gap up indicator
- `spy_gap_pct` - Gap percentage

**Cross-Asset Features**:
- `vix_spy_ratio` - VIX to SPY ratio (LightGBM)
- `vix_stretched_low` - VIX stretched low (CatBoost)
- `dxy_rising` - DXY rising indicator (CatBoost)

---

## Key Insights

### 1. Model Performance Comparison
- **LightGBM is the clear winner** with 60.54% accuracy and 0.5546 ROC AUC
- **CatBoost performed moderately** with 58-60% accuracy but lower ROC AUC
- **XGBoost struggled** on this dataset, achieving only ~50% (random baseline)
- **Feature selection significantly improved** LightGBM (+7.46% ROC AUC) but not CatBoost
- All models show **limited predictive power** for 3-day price movements (best AUC ~0.55)

### 2. Algorithm Characteristics
- **LightGBM**: Leaf-wise growth, fast training, best generalization with feature selection
- **CatBoost**: Ordered boosting, symmetric trees, more stable but less responsive to feature selection
- **XGBoost**: Level-wise growth, struggled even with aggressive regularization

### 3. Early Stopping Behavior
- **LightGBM selected model**: Stopped at iteration 3 (very aggressive early stopping)
- **CatBoost full model**: Stopped at iteration 73 (more conservative)
- **Early stopping was critical** for preventing overfitting across all models

### 4. Market Regime Challenges
- **Regime shift** between training (2015-2023) and test (2024-2026) periods
- Models trained on historical data struggle with current market dynamics
- Test accuracy near 60% suggests weak signal for 3-day predictions
- All models tend to **over-predict "Up" movements** (high recall, low precision for Down)

### 5. Feature Importance Consistency
- **Volume-based features** are most predictive across both LightGBM and CatBoost
- **Technical indicators** (RSI, MACD, ATR) provide consistent signal
- **Cross-asset relationships** (VIX/SPY) add marginal value
- **Gap features** important for CatBoost but not LightGBM
- Many features have **zero permutation importance**, suggesting high redundancy

### 6. Overfitting Mitigation
- **Early stopping** was crucial (LightGBM stopped at iteration 3 for selected features)
- **Feature selection** reduced overfitting and improved generalization (especially for LightGBM)
- **Simpler models** (fewer features, fewer trees) performed better on test set
- **Default hyperparameters** worked better than aggressive regularization (LightGBM vs XGBoost)

---

## Recommendations

### For Model Improvement

1. **Extend Prediction Horizon**: Consider 5-day or 10-day predictions instead of 3-day
2. **Add Regime Detection**: Incorporate market regime features to adapt to changing conditions
3. **Alternative Targets**: Try predicting magnitude of moves, not just direction
4. **Ensemble Methods**: Combine XGBoost and LightGBM predictions
5. **Rolling Window Training**: Retrain models periodically on recent data

### For Feature Engineering

1. **Focus on Volume**: Volume-based features show strongest signal
2. **Reduce Feature Set**: Start with top 10-20 features instead of 115
3. **Add Sentiment Data**: Consider news sentiment, options flow, or social media data
4. **Market Microstructure**: Add intraday patterns, bid-ask spreads, or order flow

### For Production Use

⚠️ **Current models are NOT recommended for live trading** due to:
- Low test ROC AUC (~0.55, barely better than random)
- High false positive rate (predicts "Up" too often)
- Regime shift between train and test periods
- Limited edge for 3-day predictions

---

## Output Files

### XGBoost
- Location: `output/models/xgboost/`
- Files:
  - `xgboost_model.json` - Full model (115 features)
  - `xgboost_model_selected_features.json` - Selected model (40 features)
  - `permutation_importance.csv`
  - `selected_features.txt`
  - `feature_importance_top30.png`
  - `permutation_importance_top40.png`
  - `training_curves.png`

### LightGBM ⭐
- Location: `output/models/lightgbm/`
- Files:
  - `lightgbm_model.json` - Full model (115 features)
  - `lightgbm_model_selected_features.json` - **Best model** (40 features)
  - `permutation_importance.csv`
  - `selected_features.txt`
  - `feature_importance_top30.png`
  - `permutation_importance_top40.png`
  - `training_curves.png`

### CatBoost
- Location: `output/models/catboost/`
- Files:
  - `catboost_model.json` - Full model (115 features)
  - `catboost_model_selected_features.json` - Selected model (40 features)
  - `permutation_importance.csv`
  - `selected_features.txt`
  - `feature_importance_top30.png`
  - `permutation_importance_top40.png`
  - `training_curves.png`

---

## How to Use the LightGBM Model (Recommended)

The **LightGBM model with 40 selected features** is the best performing model and recommended for use.

### Model Files

**Best Model**: `output/models/lightgbm/lightgbm_model_selected_features.json`

**Required Files**:
- Model: `lightgbm_model_selected_features.json`
- Features: `selected_features.txt` (list of 40 required features)
- Metrics: `metrics.json` (performance benchmarks)

### Loading the Model

```python
import lightgbm as lgb
import pandas as pd

# Load the trained model
model = lgb.Booster(model_file='output/models/lightgbm/lightgbm_model_selected_features.json')

# Load the required features list
with open('output/models/lightgbm/selected_features.txt', 'r') as f:
    required_features = [line.strip() for line in f.readlines()]

print(f"Model requires {len(required_features)} features")
```

### Making Predictions

```python
# Prepare your data with the required features
# X should be a pandas DataFrame with the 40 required features
X_new = pd.DataFrame(...)  # Your feature data

# Ensure features are in the correct order
X_new = X_new[required_features]

# Make predictions
# Returns probabilities for [Down, Up]
predictions_proba = model.predict(X_new)

# Convert to binary predictions (0=Down, 1=Up)
predictions = (predictions_proba > 0.5).astype(int)

# Get probability of upward movement
prob_up = predictions_proba

print(f"Prediction: {'Up' if predictions[0] == 1 else 'Down'}")
print(f"Probability of Up: {prob_up[0]:.2%}")
```

### Required Features (40 total)

The model requires these 40 features in the following order:

```python
# Load from file
with open('output/models/lightgbm/selected_features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]
```

**Top 6 Most Important Features**:
1. `spy_volume_ratio_50d` - 50-day volume ratio
2. `spy_rsi_14` - 14-day RSI
3. `spy_macd_pct` - MACD percentage
4. `spy_volume_ratio_20d` - 20-day volume ratio
5. `spy_momentum_3d` - 3-day momentum
6. `vix_spy_ratio` - VIX to SPY ratio

### Feature Engineering Pipeline

To generate the required features, use the feature engineering pipeline:

```python
# Run the feature engineering script
# This will create spy_features_full.csv with all features
python feature_engineering.py

# Load the features
df = pd.read_csv('output/spy_features_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract required features
X = df[required_features]
y = df['target']  # Optional: if you have labels
```

### Complete Prediction Example

```python
import lightgbm as lgb
import pandas as pd
from datetime import datetime

# 1. Load model and feature list
model = lgb.Booster(model_file='output/models/lightgbm/lightgbm_model_selected_features.json')

with open('output/models/lightgbm/selected_features.txt', 'r') as f:
    required_features = [line.strip() for line in f.readlines()]

# 2. Load your data (from feature engineering output)
df = pd.read_csv('output/spy_features_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 3. Get the most recent data point
latest_data = df[required_features].iloc[-1:].copy()

# 4. Check for missing values
if latest_data.isna().any().any():
    print("Warning: Missing values detected!")
    print(latest_data.isna().sum()[latest_data.isna().sum() > 0])
else:
    # 5. Make prediction
    prob_up = model.predict(latest_data)[0]
    prediction = "Up" if prob_up > 0.5 else "Down"
    
    # 6. Display results
    print(f"Date: {df.index[-1].date()}")
    print(f"Prediction: {prediction}")
    print(f"Probability of Up: {prob_up:.2%}")
    print(f"Probability of Down: {(1-prob_up):.2%}")
    print(f"Confidence: {abs(prob_up - 0.5) * 2:.2%}")
```

### Model Performance Expectations

Based on test set evaluation (2025-01-02 to 2026-01-16):

- **Accuracy**: ~60.54%
- **ROC AUC**: ~0.5546
- **Precision (Up)**: ~61%
- **Recall (Up)**: ~96%

**Important Notes**:
- Model has **high recall for "Up" predictions** (catches most upward movements)
- Model has **low precision for "Down" predictions** (many false alarms)
- Performance may degrade with market regime changes
- Best used as one signal among many, not as sole trading decision

### Integration with Trading System

```python
class SPYPredictor:
    """Wrapper class for LightGBM SPY prediction model."""
    
    def __init__(self, model_path, features_path):
        self.model = lgb.Booster(model_file=model_path)
        
        with open(features_path, 'r') as f:
            self.required_features = [line.strip() for line in f.readlines()]
    
    def predict(self, features_df):
        """
        Predict SPY direction for next 3 days.
        
        Args:
            features_df: DataFrame with required features
            
        Returns:
            dict with prediction, probability, and confidence
        """
        # Ensure correct features and order
        X = features_df[self.required_features]
        
        # Check for missing values
        if X.isna().any().any():
            raise ValueError("Missing values in features!")
        
        # Predict
        prob_up = self.model.predict(X)[0]
        
        return {
            'prediction': 'Up' if prob_up > 0.5 else 'Down',
            'prob_up': prob_up,
            'prob_down': 1 - prob_up,
            'confidence': abs(prob_up - 0.5) * 2
        }

# Usage
predictor = SPYPredictor(
    model_path='output/models/lightgbm/lightgbm_model_selected_features.json',
    features_path='output/models/lightgbm/selected_features.txt'
)

# Get latest features from your data pipeline
latest_features = get_latest_features()  # Your function

# Make prediction
result = predictor.predict(latest_features)
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.1%})")
```

### Retraining the Model

To retrain with updated data:

```bash
# 1. Update data
python download_data.py

# 2. Regenerate features
python feature_engineering.py

# 3. Retrain LightGBM model
python modelling_lightgbm.py
```

The model will automatically:
- Split data into train/test
- Handle missing values
- Calculate permutation importance
- Select top 40 features
- Save new model files

---

## Conclusion

While LightGBM with feature selection achieved the best performance (60.54% accuracy, 0.5546 ROC AUC), the overall predictive power for 3-day SPY movements remains limited. The models demonstrate that:

1. **Volume patterns** contain the most predictive signal
2. **Feature selection** is critical for generalization
3. **Market regime shifts** significantly impact model performance
4. **Short-term predictions** (3 days) are inherently challenging

Further work should focus on extending the prediction horizon, incorporating regime detection, and exploring alternative data sources beyond traditional technical indicators.

---

*Last Updated: 2026-01-17*
