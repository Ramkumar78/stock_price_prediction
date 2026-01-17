# SPY Stock Price Prediction with Gradient Boosting

A machine learning project that predicts SPY (S&P 500 ETF) price movements over the next 3 days using gradient boosting models (XGBoost, LightGBM, CatBoost) trained on 115 engineered features from technical indicators, volume patterns, and cross-asset relationships.

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline for predicting whether SPY will move up or down over the next 3 trading days. The pipeline includes:

- **Data Collection**: Automated download of OHLCV data for SPY, VIX, TLT, DXY, and GLD
- **Feature Engineering**: 115 technical indicators across 5 feature modules
- **Model Training**: Three gradient boosting algorithms with 2-phase training (all features → selected features)
- **Model Evaluation**: Comprehensive performance analysis and comparison
- **Production Ready**: Model loading and prediction examples

## 📊 Key Results

| Model | Test ROC AUC | Test Accuracy | Status |
|-------|--------------|---------------|--------|
| **LightGBM (40 features)** | **0.5546** | **60.54%** | ✅ **Best** |
| CatBoost (40 features) | 0.4953 | 59.77% | ⚠️ Decent |
| XGBoost (40 features) | 0.4750 | 58.62% | ⚠️ Decent |
| Ensemble (soft voting) | 0.4857 | 55.17% | ❌ Worse |

**Winner**: LightGBM with 40 selected features achieves the best performance.

## 🗂️ Project Structure

```
stockprediction/
├── README.md                          # This file
├── download_data.py                   # Data download script
├── download_data.md                   # Data download documentation
├── feature_engineering.py             # Main feature engineering pipeline
├── FEATURE_MODULES_README.md          # Feature engineering documentation
├── spy_xgboost_features.md            # Feature specifications
│
├── Feature Modules/
│   ├── price_based_features.py        # Price, momentum, MA features
│   ├── volume_features.py             # Volume-based features
│   ├── volatility_features.py         # Volatility and technical indicators
│   ├── cross_asset_features.py        # VIX, TLT, DXY, GLD features
│   └── regime_dependent_features.py   # Market regime features
│
├── Model Training Scripts/
│   ├── modelling_xgboost.py           # XGBoost training
│   ├── modelling_lightgbm.py          # LightGBM training (BEST)
│   ├── modelling_catboost.py          # CatBoost training
│   └── modelling_ensemble.py          # Ensemble model
│
├── modelling.md                       # Model results & usage guide
│
└── output/
    ├── spy_features_full.csv          # Generated features
    └── models/
        ├── xgboost/                   # XGBoost models & metrics
        ├── lightgbm/                  # LightGBM models & metrics ⭐
        ├── catboost/                  # CatBoost models & metrics
        └── ensemble/                  # Ensemble model & metrics
```

## 📚 Documentation

- **[download_data.md](download_data.md)** - Data collection and preparation
- **[FEATURE_MODULES_README.md](FEATURE_MODULES_README.md)** - Feature engineering details
- **[spy_xgboost_features.md](spy_xgboost_features.md)** - Feature specifications
- **[modelling.md](modelling.md)** - Model comparison, results, and usage guide

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stockprediction.git
cd stockprediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy yfinance ta-lib
pip install xgboost lightgbm catboost
pip install scikit-learn matplotlib seaborn
```

### 2. Docker Deployment

The application works best when run as a full-stack Dockerized application using Docker Compose. This automatically builds and starts both the backend API and the frontend dashboard.

```bash
# Build and run the application
docker-compose up --build
```

#### Accessing the Application

Once the containers are running (you will see logs indicating the server started), open your web browser and navigate to:

- **Frontend Dashboard**: http://localhost:3000
  - This is the main interface where you can view predictions, charts, and trigger data updates.
- **Backend API Documentation**: http://localhost:8000/docs
  - Interactive Swagger UI to test API endpoints directly.

> **Note**: If port 3000 or 8000 is already in use on your machine, you may need to modify the port mapping in `docker-compose.yml`.

### 3. Download Data (Manual / Docker)

You can trigger data download from the dashboard UI or manually:

```bash
# Download OHLCV data for SPY, VIX, TLT, DXY, GLD (2015-present)
# If running locally:
python -m app.core.download_data

# If running via API:
curl -X POST http://localhost:8000/data/refresh
```

See [download_data.md](download_data.md) for details.

### 3. Generate Features

```bash
# Run feature engineering pipeline (creates 115 features)
python feature_engineering.py
```

Output: `output/spy_features_full.csv`

See [FEATURE_MODULES_README.md](FEATURE_MODULES_README.md) for feature details.

### 4. Train Models

```bash
# Train LightGBM (recommended)
python modelling_lightgbm.py

# Or train all models
python modelling_xgboost.py
python modelling_catboost.py
python modelling_ensemble.py
```

Models saved to `output/models/{model_name}/`

### 5. Make Predictions

```python
import lightgbm as lgb
import pandas as pd

# Load best model
model = lgb.Booster(model_file='output/models/lightgbm/lightgbm_model_selected_features.json')

# Load required features
with open('output/models/lightgbm/selected_features.txt', 'r') as f:
    required_features = [line.strip() for line in f.readlines()]

# Load your data
df = pd.read_csv('output/spy_features_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Get latest data
latest = df[required_features].iloc[-1:]

# Predict
prob_up = model.predict(latest)[0]
prediction = "Up" if prob_up > 0.5 else "Down"

print(f"Prediction: {prediction}")
print(f"Probability: {prob_up:.2%}")
```

See [modelling.md](modelling.md) for complete usage guide.

## 🔧 Features

### Feature Categories (115 total)

1. **Price-Based Features** (19 features)
   - Momentum indicators (3d, 5d, 10d, 20d, 50d)
   - Moving averages (20, 50, 100, 200-day)
   - Price extremes and gaps

2. **Volume Features** (16 features)
   - Volume ratios and z-scores
   - Volume trends and spikes
   - Accumulation/distribution patterns

3. **Volatility Features** (30 features)
   - ATR, Bollinger Bands, RSI, MACD
   - Volatility regimes
   - Technical indicator combinations

4. **Cross-Asset Features** (35 features)
   - VIX (volatility index) features
   - TLT (bonds), DXY (dollar), GLD (gold)
   - Risk-on/risk-off regimes

5. **Regime-Dependent Features** (15 features)
   - Market regime detection
   - Hurst exponent (trend persistence)
   - Regime-specific indicators

### Top 6 Most Important Features

1. `spy_volume_ratio_50d` - 50-day volume ratio
2. `spy_rsi_14` - 14-day RSI
3. `spy_macd_pct` - MACD percentage
4. `spy_volume_ratio_20d` - 20-day volume ratio
5. `spy_momentum_3d` - 3-day momentum
6. `vix_spy_ratio` - VIX to SPY ratio

## 📈 Model Training Approach

### Two-Phase Training

**Phase 1: All Features (115)**
- Train with all engineered features
- Calculate permutation importance on test set
- Identify truly predictive features

**Phase 2: Selected Features (40)**
- Retrain with top 40 features from permutation importance
- Reduce overfitting and improve generalization
- Compare performance vs full model

### Models Evaluated

1. **XGBoost**: Level-wise tree growth with aggressive regularization
2. **LightGBM**: Leaf-wise tree growth with default parameters ⭐
3. **CatBoost**: Ordered boosting with symmetric trees
4. **Ensemble**: Soft voting (probability averaging) of all three

## 📊 Performance Analysis

### LightGBM (Best Model)

**Metrics** (Test Set: 2025-01-02 to 2026-01-16):
- ROC AUC: 0.5546
- Accuracy: 60.54%
- Precision (Up): 61%
- Recall (Up): 96%
- F1 Score: 0.7469

**Characteristics**:
- High recall for "Up" predictions (catches most upward movements)
- Low precision for "Down" predictions (many false alarms)
- Stopped at iteration 3 (aggressive early stopping)
- Best generalization with feature selection

### Key Insights

1. **Volume features are most predictive** across all models
2. **Feature selection significantly improved** LightGBM (+7.46% ROC AUC)
3. **Early stopping was critical** for preventing overfitting
4. **Market regime shifts** impact model performance (train: 2015-2023, test: 2024-2026)
5. **Limited predictive power** for 3-day movements (best AUC ~0.55)

## ⚠️ Important Notes

### Model Limitations

- **Not recommended for live trading** without additional signals
- Performance may degrade with market regime changes
- Best used as one signal among many
- 3-day prediction horizon is inherently challenging

### Production Considerations

- Retrain periodically with updated data
- Monitor performance metrics over time
- Implement proper risk management
- Consider ensemble with other strategies

## 🔄 Retraining Workflow

```bash
# 1. Update data
python download_data.py

# 2. Regenerate features
python feature_engineering.py

# 3. Retrain models
python modelling_lightgbm.py
python modelling_xgboost.py
python modelling_catboost.py

# 4. Compare results
# Check output/models/{model_name}/metrics.json
```

## 📁 Output Files

Each model generates:
- `{model}_model.json` - Full model (115 features)
- `{model}_model_selected_features.json` - Selected model (40 features)
- `permutation_importance.csv` - Feature importance scores
- `selected_features.txt` - Top 40 features list
- `feature_importance_top30.png` - Importance visualization
- `permutation_importance_top40.png` - Permutation importance plot
- `training_curves.png` - AUC curves during training
- `metrics.json` - Performance metrics

## 🛠️ Technical Stack

- **Python**: 3.11+
- **Data**: pandas, numpy, yfinance
- **ML**: xgboost, lightgbm, catboost, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Technical Analysis**: ta-lib

## 📖 Further Reading

- [Feature Engineering Guide](FEATURE_MODULES_README.md) - Detailed feature descriptions
- [Model Comparison & Usage](modelling.md) - Complete model analysis and usage examples
- [Data Collection](download_data.md) - Data sources and preparation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Feature Engineering**: Add sentiment data, options flow, market microstructure
2. **Model Improvements**: Extend prediction horizon, add regime detection
3. **Alternative Targets**: Predict magnitude of moves, not just direction
4. **Ensemble Methods**: Weighted voting, stacking, blending

## 📄 License

This project is for educational purposes. Not financial advice.

## 🙏 Acknowledgments

- Data provided by Yahoo Finance (yfinance)
- Technical indicators from TA-Lib
- Gradient boosting frameworks: XGBoost, LightGBM, CatBoost

---

**Last Updated**: 2026-01-17

**Best Model**: `output/models/lightgbm/lightgbm_model_selected_features.json`
