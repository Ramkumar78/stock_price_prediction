# Feature Engineering Modules for SPY XGBoost Model

This directory contains modular feature engineering code for building an XGBoost price prediction model for SPY and other assets.

## 📁 Module Overview

### Core Feature Modules

| Module | Features | Description |
|--------|----------|-------------|
| **`price_based_features.py`** | 47 features | Momentum, moving averages, gaps, cross-asset relationships (VIX, TLT, DXY, GLD) |
| **`volume_features.py`** | 14 features | Volume ratios, percentiles, price-volume confirmation |
| **`volatility_features.py`** | 10 features | ATR (normalized), volatility regimes, intraday range |
| **`technical_features.py`** | 24 features | Bollinger Bands, RSI, MACD, Hurst Exponent |
| **`regime_features.py`** | 12 features | Momentum regimes, consecutive patterns, combined setups |
| **`regime_dependent_features.py`** | 10 features | Interaction terms combining indicators with market regime |

### Master Module

**`feature_engineering.py`** - Combines all modules into a single interface with helper functions.

---

## 🚀 Quick Start

### Basic Usage

```python
from feature_engineering import create_all_features, create_target_variable
import pandas as pd

# Load your OHLC data
spy = pd.read_csv('spy_data.csv', index_col='Date', parse_dates=True)
vix = pd.read_csv('vix_data.csv', index_col='Date', parse_dates=True)

# Create all features
features = create_all_features(
    spy=spy,
    vix=vix,
    include_regime_dependent=True
)

# Create target variable (3-day forward direction)
y = create_target_variable(spy, forward_days=3)

# Ready for modeling!
print(f"Features: {features.shape}")
print(f"Target: {y.shape}")
```

### Using Individual Modules

```python
from price_based_features import create_price_based_features
from volume_features import create_volume_features
from technical_features import create_technical_features

# Create only specific feature types
price_features = create_price_based_features(spy=spy, vix=vix)
volume_features = create_volume_features(spy)
technical_features = create_technical_features(spy)

# Combine manually
import pandas as pd
all_features = pd.concat([price_features, volume_features, technical_features], axis=1)
```

---

## 📊 Feature Categories

### 1. Price-Based Features (47 features)

**SPY Features:**
- **Momentum** (5): Log returns over 3d, 5d, 10d, 20d, 50d windows
- **Rate of Change** (2): ROC and momentum acceleration
- **Moving Averages** (6): Binary above/below 20/50/100/200 MA, golden cross, all MAs aligned
- **High/Low** (5): New highs/lows, near highs/lows (within 2%)
- **Gaps** (4): Gap percentage, gap up/down, gap filled

**Cross-Asset Features:**
- **VIX** (13): Level regimes, changes, mean reversion, velocity, SPY-VIX relationship
- **TLT** (3): Momentum, MA position, SPY-TLT correlation
- **DXY** (3): Momentum, MA position, rising trend
- **GLD** (3): Momentum, MA position, outperformance vs SPY
- **Risk Regimes** (2): Risk-on, risk-off indicators

### 2. Volume Features (14 features)

- **Ratios** (3): Volume relative to 10d, 20d, 50d averages
- **Statistical** (2): Volume percentile, z-score
- **Binary** (4): Above average, heavy volume, spike, increasing trend
- **Price-Volume** (4): Accumulation, distribution, confirmation, divergence

### 3. Volatility Features (10 features)

- **ATR** (3): Normalized ATR percentage, percentile, z-score
- **Binary** (4): High/low volatility regimes, expanding/contracting
- **Intraday** (3): Daily range percentage, wide/narrow range days

### 4. Technical Indicators (24 features)

- **Bollinger Bands** (9): Position, band width, squeeze, quartiles, walking upper band
- **RSI** (6): RSI-14, oversold/overbought levels, extreme levels, above 50
- **MACD** (6): Normalized MACD, histogram, ratio, crosses
- **Hurst Exponent** (4): Hurst-50d, trending/mean-reverting/strong trend regimes

### 5. Binary Regime Features (12 features)

- **Momentum Regimes** (5): 5d/20d/50d positive, all aligned, reversal
- **Consecutive Moves** (2): Three up/down days
- **Combined Setups** (5): Strong bull, overbought/oversold setups, breakout confirmed

### 6. Regime-Dependent Features (10 features)

- **Interaction Terms** (6): RSI/MACD/BB/Volume in trending vs mean-reverting markets
- **Strategies** (3): Trend continuation, mean reversion setup, regime shift detection

---

## 🎯 Design Principles

### Scale-Invariant Features ✅

All features are designed to be scale-invariant, making them suitable for different ETFs at various price levels:

- **Log returns** instead of absolute price changes
- **Ratios and percentages** instead of absolute values
- **Percentiles and z-scores** for normalization
- **Binary indicators** (0/1)

### Features to NEVER Use ❌

- Absolute ATR
- Absolute volume
- Absolute MACD
- Raw price levels (except for specific indicators like VIX level)

---

## 📈 Model Training Workflow

### Step 1: Create Features

```python
from feature_engineering import create_all_features, create_target_variable

features = create_all_features(spy=spy, vix=vix, tlt=tlt, dxy=dxy, gld=gld)
y = create_target_variable(spy, forward_days=3)
```

### Step 2: Handle Missing Values

```python
# Remove rows with NaN (from rolling windows)
valid_idx = features.dropna().index
X = features.loc[valid_idx]
y = y.loc[valid_idx]
```

### Step 3: Remove Highly Correlated Features

```python
from feature_engineering import remove_correlated_features

X_filtered = remove_correlated_features(X, threshold=0.95)
```

### Step 4: Train Initial Model

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(
    max_depth=3,
    learning_rate=0.01,
    n_estimators=500,
    random_state=42
)
model.fit(X_train, y_train)
```

### Step 5: Feature Selection via Permutation Importance

```python
from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(
    model, 
    X_test, 
    y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1
)

# Select top 30-40 features
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std
}).sort_values('importance', ascending=False)

top_features = importance_df.head(35)['feature'].tolist()
```

### Step 6: Retrain Final Model

```python
X_train_final = X_train[top_features]
X_test_final = X_test[top_features]

model_final = XGBClassifier(
    max_depth=3,
    learning_rate=0.01,
    n_estimators=500,
    random_state=42
)
model_final.fit(X_train_final, y_train)
```

---

## ⚠️ Performance Notes

### Computational Complexity

- **Fast** (~1 second): Price, volume, volatility features
- **Medium** (~5 seconds): Bollinger Bands, RSI, MACD
- **Slow** (~30-60 seconds): **Hurst Exponent** (rolling 50-day window)

### Recommendations

1. **For rapid prototyping**: Set `include_regime_dependent=False` to skip Hurst calculation
2. **For production**: Calculate Hurst once and cache the results
3. **For backtesting**: Pre-calculate all features and save to CSV/parquet

---

## 📝 Expected Feature Importance

Based on the specification, these features are expected to be most important:

1. **Hurst Exponent** (`spy_hurst_50d`) - Market regime detection
2. **BB Position** (`spy_bb_position`) - Overbought/oversold
3. **VIX Features** - Fear gauge and mean reversion
4. **Momentum** (20d, 50d) - Trend strength
5. **RSI** - Momentum oscillator
6. **Volume Confirmation** - Price-volume relationships
7. **Regime-Dependent Features** - Context-aware signals

---

## 🔧 Customization

### Adding New Features

To add custom features, create a new module following this pattern:

```python
# custom_features.py
import pandas as pd
import numpy as np

def create_custom_features(spy: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=spy.index)
    
    # Your custom feature logic here
    features['spy_custom_feature'] = ...
    
    return features
```

Then import and use in `feature_engineering.py`:

```python
from custom_features import create_custom_features

# Add to create_all_features function
custom_features_df = create_custom_features(spy)
all_features = pd.concat([all_features, custom_features_df], axis=1)
```

---

## 📚 References

- Feature specifications: `spy_xgboost_features.md`
- Target: 3-day forward direction (binary classification)
- Expected total features: ~70-80 before selection
- Final model features: ~30-40 after permutation importance

---

## ✅ Checklist

- [x] All features are scale-invariant
- [x] Proper asset name prefixes (e.g., `spy_`, `vix_`)
- [x] Binary features use `.astype(int)`
- [x] Continuous features are normalized
- [x] Cross-asset features handle index alignment
- [x] Hurst exponent has error handling
- [x] Example usage in each module
- [x] Master module combines everything

---

## 🎓 Next Steps

1. Load your actual OHLC data for SPY, VIX, TLT, DXY, GLD
2. Run `feature_engineering.py` to generate all features
3. Train XGBoost model with initial feature set
4. Use permutation importance to select top features
5. Validate on out-of-sample data
6. Test on multiple ETFs to ensure generalizability
