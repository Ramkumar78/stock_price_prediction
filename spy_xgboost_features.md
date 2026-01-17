# SPY XGBoost Model - Recommended Features

**Target:** 3-day forward direction (up/down classification)

**Total Recommended Features:** ~70-80 features
- SPY features: ~35-40
- Cross-asset features: ~30-35
- Then use permutation importance to select top 30-40 for final model

---

## Feature Categories

1. [Price-Based Features](#price-based-features)
2. [Volume Features](#volume-features)
3. [Volatility Features](#volatility-features)
4. [Technical Indicators](#technical-indicators)
5. [Binary Regime Features](#binary-regime-features)
6. [Cross-Asset Features (VIX)](#cross-asset-features-vix)
7. [Cross-Asset Features (TLT, DXY, GLD)](#cross-asset-features-tlt-dxy-gld)
8. [Regime-Dependent Features](#regime-dependent-features)

---

## Price-Based Features

### Momentum (Log Returns) - Multiple Windows
```python
# Short, medium, long momentum
momentum_3d = np.log(spy['Close'] / spy['Close'].shift(3))
momentum_5d = np.log(spy['Close'] / spy['Close'].shift(5))
momentum_10d = np.log(spy['Close'] / spy['Close'].shift(10))
momentum_20d = np.log(spy['Close'] / spy['Close'].shift(20))
momentum_50d = np.log(spy['Close'] / spy['Close'].shift(50))
```

### Rate of Change (Momentum Acceleration)
```python
# ROC: change in momentum
mom_10d = np.log(spy['Close'] / spy['Close'].shift(10))
roc_5d = mom_10d - mom_10d.shift(5)

# Alternative: recent vs older momentum
recent_mom = np.log(spy['Close'] / spy['Close'].shift(5))
older_mom = np.log(spy['Close'].shift(5) / spy['Close'].shift(10))
mom_acceleration = recent_mom - older_mom
```

### Moving Average Features
```python
# Binary: Above/Below key MAs
spy_above_20ma = (spy['Close'] > spy['Close'].rolling(20).mean()).astype(int)
spy_above_50ma = (spy['Close'] > spy['Close'].rolling(50).mean()).astype(int)
spy_above_100ma = (spy['Close'] > spy['Close'].rolling(100).mean()).astype(int)
spy_above_200ma = (spy['Close'] > spy['Close'].rolling(200).mean()).astype(int)

# Golden/Death Cross
ma_50 = spy['Close'].rolling(50).mean()
ma_200 = spy['Close'].rolling(200).mean()
ma_50_above_200 = (ma_50 > ma_200).astype(int)

# All MAs aligned (strong trend)
all_mas_bullish = ((spy['Close'] > spy['Close'].rolling(20).mean()) & 
                   (spy['Close'] > spy['Close'].rolling(50).mean()) &
                   (spy['Close'] > spy['Close'].rolling(200).mean())).astype(int)
```

### High/Low Features
```python
# New highs/lows
new_high_20d = (spy['Close'] == spy['High'].rolling(20).max()).astype(int)
new_high_50d = (spy['Close'] == spy['High'].rolling(50).max()).astype(int)
new_low_20d = (spy['Close'] == spy['Low'].rolling(20).min()).astype(int)

# Near highs/lows (within 2%)
near_high_20d = (spy['Close'] > spy['High'].rolling(20).max() * 0.98).astype(int)
near_low_20d = (spy['Close'] < spy['Low'].rolling(20).min() * 1.02).astype(int)
```

### Gap Features
```python
# Gap as percentage (scale-invariant)
gap_pct = (spy['Open'] - spy['Close'].shift(1)) / spy['Close'].shift(1)

# Binary gap indicators
gap_up = (spy['Open'] > spy['Close'].shift(1) * 1.002).astype(int)  # >0.2% gap
gap_down = (spy['Open'] < spy['Close'].shift(1) * 0.998).astype(int)

# Gap filled
gap_filled = ((gap_up == 1) & (spy['Low'] <= spy['Close'].shift(1))).astype(int)
```

---

## Volume Features

### Volume Ratios (Scale-Invariant)
```python
# Volume relative to averages
volume_ratio_10d = spy['Volume'] / spy['Volume'].rolling(10).mean()
volume_ratio_20d = spy['Volume'] / spy['Volume'].rolling(20).mean()
volume_ratio_50d = spy['Volume'] / spy['Volume'].rolling(50).mean()

# Volume percentile
volume_percentile = spy['Volume'].rolling(50).rank(pct=True)

# Volume z-score
volume_zscore = (spy['Volume'] - spy['Volume'].rolling(50).mean()) / \
                spy['Volume'].rolling(50).std()
```

### Volume Binary Features
```python
# Volume above average
volume_above_avg = (spy['Volume'] > spy['Volume'].rolling(20).mean()).astype(int)

# Heavy volume (top 20%)
heavy_volume = (spy['Volume'] > spy['Volume'].rolling(50).quantile(0.8)).astype(int)

# Volume spike (>2x average)
volume_spike = (spy['Volume'] > spy['Volume'].rolling(20).mean() * 2).astype(int)

# Volume increasing trend
volume_increasing = ((spy['Volume'] > spy['Volume'].shift(1)) & 
                     (spy['Volume'].shift(1) > spy['Volume'].shift(2))).astype(int)
```

### Price-Volume Confirmation
```python
# Accumulation (up day + high volume)
accumulation = ((spy['Close'] > spy['Open']) & 
                (spy['Volume'] > spy['Volume'].rolling(20).mean())).astype(int)

# Distribution (down day + high volume)
distribution = ((spy['Close'] < spy['Open']) & 
                (spy['Volume'] > spy['Volume'].rolling(20).mean())).astype(int)

# Price-volume confirmation
price_vol_confirm = ((spy['Close'] > spy['Close'].shift(1)) & 
                     (spy['Volume'] > spy['Volume'].shift(1))).astype(int)

# Price-volume divergence (weak rally)
price_vol_divergence = ((spy['Close'] > spy['Close'].shift(1)) & 
                        (spy['Volume'] < spy['Volume'].shift(1))).astype(int)
```

---

## Volatility Features

### ATR (Normalized - Scale-Invariant)
```python
# Calculate ATR
high_low = spy['High'] - spy['Low']
high_close = np.abs(spy['High'] - spy['Close'].shift(1))
low_close = np.abs(spy['Low'] - spy['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr_14 = true_range.rolling(14).mean()

# Normalize ATR by price (scale-invariant)
atr_pct = atr_14 / spy['Close']

# ATR percentile (relative to history)
atr_percentile = atr_14.rolling(100).rank(pct=True)

# ATR z-score
atr_zscore = (atr_14 - atr_14.rolling(50).mean()) / atr_14.rolling(50).std()
```

### ATR Binary Features
```python
# High/low volatility regimes
high_volatility = (atr_percentile > 0.8).astype(int)
low_volatility = (atr_percentile < 0.2).astype(int)

# Volatility expanding/contracting
vol_expanding = (atr_14 > atr_14.shift(5)).astype(int)
vol_contracting = (atr_14 < atr_14.shift(5)).astype(int)
```

### Intraday Range
```python
# Daily range as percentage
daily_range_pct = (spy['High'] - spy['Low']) / spy['Close']

# Wide/narrow range days
range_percentile = daily_range_pct.rolling(20).rank(pct=True)
wide_range = (range_percentile > 0.8).astype(int)
narrow_range = (range_percentile < 0.2).astype(int)
```

---

## Technical Indicators

### Bollinger Bands
```python
# Calculate BB
sma_20 = spy['Close'].rolling(20).mean()
std_20 = spy['Close'].rolling(20).std()
upper_band = sma_20 + (2 * std_20)
lower_band = sma_20 - (2 * std_20)

# BB Position (0 to 1) - CRITICAL FEATURE
bb_position = (spy['Close'] - lower_band) / (upper_band - lower_band)

# Normalized band width (scale-invariant)
band_width_pct = (upper_band - lower_band) / sma_20

# BB Squeeze (low volatility)
bb_squeeze = (band_width_pct < band_width_pct.rolling(100).quantile(0.2)).astype(int)
```

### BB Binary Features
```python
# Above/below bands
above_upper_bb = (spy['Close'] > upper_band).astype(int)
below_lower_bb = (spy['Close'] < lower_band).astype(int)

# BB quartiles
bb_upper_quartile = (bb_position > 0.75).astype(int)
bb_lower_quartile = (bb_position < 0.25).astype(int)
bb_middle_zone = ((bb_position >= 0.4) & (bb_position <= 0.6)).astype(int)

# Walking the band (riding upper band = strong trend)
bb_walking_upper = ((bb_position > 0.8) & 
                    (bb_position.shift(1) > 0.8) &
                    (bb_position.shift(2) > 0.8)).astype(int)
```

### RSI
```python
# Calculate RSI
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi_14 = calculate_rsi(spy['Close'], 14)

# Binary RSI features
rsi_oversold = (rsi_14 < 30).astype(int)
rsi_overbought = (rsi_14 > 70).astype(int)
rsi_extreme_oversold = (rsi_14 < 20).astype(int)
rsi_extreme_overbought = (rsi_14 > 80).astype(int)
rsi_above_50 = (rsi_14 > 50).astype(int)
```

### MACD (Normalized)
```python
# Calculate MACD
ema_12 = spy['Close'].ewm(span=12, adjust=False).mean()
ema_26 = spy['Close'].ewm(span=26, adjust=False).mean()
macd_line = ema_12 - ema_26
signal_line = macd_line.ewm(span=9, adjust=False).mean()

# Normalize by price (scale-invariant)
macd_pct = (ema_12 - ema_26) / spy['Close']
macd_hist_pct = (macd_line - signal_line) / spy['Close']

# Alternative: MACD ratio
macd_ratio = ema_12 / ema_26 - 1

# Binary MACD features
macd_positive = (macd_line > 0).astype(int)
macd_bullish_cross = ((macd_line > signal_line) & 
                      (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
macd_bearish_cross = ((macd_line < signal_line) & 
                      (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
```

### Hurst Exponent (Trending vs Mean-Reverting)
```python
# Calculate Hurst exponent
def hurst_exponent(ts, max_lag=20):
    """
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# Rolling Hurst over 50-day window
hurst_50d = spy['Close'].rolling(50).apply(
    lambda x: hurst_exponent(x.values, max_lag=20), 
    raw=False
)

# Binary regime features - CRITICAL
trending_market = (hurst_50d > 0.6).astype(int)
mean_reverting_market = (hurst_50d < 0.4).astype(int)
strong_trend = (hurst_50d > 0.65).astype(int)
```

---

## Binary Regime Features

### Momentum Regimes
```python
# Momentum direction
mom_5d_positive = (momentum_5d > 0).astype(int)
mom_20d_positive = (momentum_20d > 0).astype(int)
mom_50d_positive = (momentum_50d > 0).astype(int)

# All momentum windows aligned
all_mom_positive = ((mom_5d_positive == 1) & 
                    (mom_20d_positive == 1) &
                    (mom_50d_positive == 1)).astype(int)

# Momentum reversal (bounce in downtrend)
mom_reversal = ((mom_5d_positive == 1) & 
                (mom_20d_positive == 0)).astype(int)
```

### Consecutive Moves
```python
# Consecutive up/down days
three_up_days = ((spy['Close'] > spy['Close'].shift(1)) & 
                 (spy['Close'].shift(1) > spy['Close'].shift(2)) &
                 (spy['Close'].shift(2) > spy['Close'].shift(3))).astype(int)

three_down_days = ((spy['Close'] < spy['Close'].shift(1)) & 
                   (spy['Close'].shift(1) < spy['Close'].shift(2)) &
                   (spy['Close'].shift(2) < spy['Close'].shift(3))).astype(int)
```

### Combined Regimes
```python
# Strong bullish regime
strong_bull = ((spy_above_200ma == 1) & 
               (spy_above_50ma == 1) &
               (mom_20d_positive == 1) &
               (volume_above_avg == 1)).astype(int)

# Overbought setup (possible reversal)
overbought_setup = ((above_upper_bb == 1) & 
                    (heavy_volume == 1) &
                    (rsi_overbought == 1)).astype(int)

# Oversold bounce setup
oversold_setup = ((below_lower_bb == 1) & 
                  (rsi_oversold == 1) &
                  (new_low_20d == 1)).astype(int)

# Breakout with confirmation
breakout_confirmed = ((spy['Close'] > spy['High'].rolling(20).max().shift(1)) &
                      (volume_spike == 1)).astype(int)
```

---

## Cross-Asset Features (VIX)

### VIX Level Features
```python
# VIX absolute level (regime indicator)
vix_level = vix['Close']

# VIX binary regimes
vix_low = (vix['Close'] < 15).astype(int)  # Complacency
vix_elevated = (vix['Close'] > 20).astype(int)  # Fear
vix_panic = (vix['Close'] > 30).astype(int)  # Panic
```

### VIX Change Features (NOT momentum - VIX mean-reverts)
```python
# Absolute changes (VIX spikes/collapses)
vix_change_1d = vix['Close'] - vix['Close'].shift(1)
vix_change_5d = vix['Close'] - vix['Close'].shift(5)

# Percentage change
vix_pct_change_1d = vix['Close'] / vix['Close'].shift(1) - 1

# VIX spike detection
vix_spike = (vix['Close'] > vix['Close'].shift(1) * 1.1).astype(int)  # 10% jump
```

### VIX Mean Reversion Features
```python
# VIX z-score (stretched = likely to revert)
vix_sma_20 = vix['Close'].rolling(20).mean()
vix_zscore = (vix['Close'] - vix_sma_20) / vix['Close'].rolling(20).std()

# VIX stretched (mean reversion signal)
vix_stretched_high = (vix_zscore > 2).astype(int)
vix_stretched_low = (vix_zscore < -1).astype(int)
```

### VIX Velocity
```python
# Rate of change of VIX changes (acceleration/deceleration)
vix_velocity = vix_change_1d - vix_change_1d.shift(1)
```

### SPY-VIX Relationship
```python
# SPY return
spy_return_1d = np.log(spy['Close'] / spy['Close'].shift(1))

# Decoupling (SPY down but VIX NOT rising = bullish resilience)
spy_vix_decoupling = ((spy_return_1d < 0) & (vix_change_1d < 0)).astype(int)

# VIX/SPY ratio (normalized fear)
vix_spy_ratio = vix['Close'] / spy['Close']
```

---

## Cross-Asset Features (TLT, DXY, GLD)

### TLT (Bond) Features
```python
# TLT momentum
tlt_momentum_20d = np.log(tlt['Close'] / tlt['Close'].shift(20))

# TLT above/below MA (rates falling/rising)
tlt_above_50ma = (tlt['Close'] > tlt['Close'].rolling(50).mean()).astype(int)

# SPY-TLT correlation (regime detector)
spy_returns = np.log(spy['Close'] / spy['Close'].shift(1))
tlt_returns = np.log(tlt['Close'] / tlt['Close'].shift(1))
spy_tlt_corr = spy_returns.rolling(20).corr(tlt_returns)
```

### DXY (Dollar) Features
```python
# DXY momentum (strong dollar = headwind for equities)
dxy_momentum_20d = np.log(dxy['Close'] / dxy['Close'].shift(20))

# DXY above MA (dollar strength)
dxy_above_50ma = (dxy['Close'] > dxy['Close'].rolling(50).mean()).astype(int)

# DXY trend
dxy_rising = (dxy_momentum_20d > 0).astype(int)
```

### GLD (Gold) Features
```python
# GLD momentum
gld_momentum_20d = np.log(gld['Close'] / gld['Close'].shift(20))

# GLD outperforming SPY (flight to safety)
gld_outperform = (gld_momentum_20d > momentum_20d).astype(int)

# GLD above MA
gld_above_50ma = (gld['Close'] > gld['Close'].rolling(50).mean()).astype(int)
```

### Risk-On/Risk-Off Indicators
```python
# Risk-on: SPY up + VIX down
risk_on = ((spy['Close'] > spy['Close'].shift(1)) & 
           (vix['Close'] < vix['Close'].shift(1))).astype(int)

# Risk-off: SPY down + VIX up + GLD up
risk_off = ((spy['Close'] < spy['Close'].shift(1)) & 
            (vix['Close'] > vix['Close'].shift(1)) &
            (gld['Close'] > gld['Close'].shift(1))).astype(int)
```

---

## Regime-Dependent Features

### Interaction Terms (Hurst × Other Features)
```python
# RSI behavior differs in trending vs mean-reverting markets
rsi_in_trending = rsi_14 * trending_market
rsi_in_mr = rsi_14 * mean_reverting_market

# MACD in trending markets
macd_trending = macd_hist_pct * trending_market

# BB position in mean-reverting markets
bb_pos_mr = bb_position * mean_reverting_market

# Volume spike interpretation depends on regime
vol_spike_trending = volume_spike * trending_market
vol_spike_mr = volume_spike * mean_reverting_market
```

### Regime-Aware Strategies
```python
# Trend continuation (in trending market)
trend_continuation = ((trending_market == 1) & 
                     (mom_20d_positive == 1) &
                     (spy_above_50ma == 1)).astype(int)

# Mean reversion setup (in mean-reverting market)
mean_reversion_setup = ((mean_reverting_market == 1) & 
                        (rsi_overbought == 1) &
                        (bb_position > 0.8)).astype(int)

# Regime shift detection
regime_shift_to_mr = ((trending_market.shift(5) == 1) & 
                      (mean_reverting_market == 1)).astype(int)
```

---

## Feature Summary by Type

### Continuous Features (~35-40)
- Momentum: 5 windows (3d, 5d, 10d, 20d, 50d)
- ROC/Acceleration: 2 features
- Volume ratios: 3 features
- ATR: atr_pct, atr_percentile, atr_zscore
- BB: bb_position, band_width_pct
- RSI: rsi_14
- MACD: macd_pct, macd_hist_pct
- Hurst: hurst_50d
- VIX: vix_level, vix_change_1d, vix_change_5d, vix_zscore, vix_spy_ratio
- TLT: tlt_momentum_20d, spy_tlt_corr
- DXY: dxy_momentum_20d
- GLD: gld_momentum_20d
- Volume: volume_percentile, volume_zscore
- Gap: gap_pct
- Interaction terms: 5-6 features

### Binary Features (~35-40)
- MA: 4 features (20/50/100/200ma)
- Highs/Lows: 5 features
- BB: 6 features
- RSI: 5 features
- MACD: 3 features
- Hurst regimes: 3 features
- Volume: 4 features
- VIX: 5 features
- Cross-asset: 6 features
- Combined regimes: 5 features
- Momentum: 5 features
- ATR: 2 features

---

## Feature Selection Workflow

### Step 1: Remove Highly Correlated Features (>0.95)
```python
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_filtered = X.drop(columns=to_drop)
```

### Step 2: Train Initial Model
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth=3,
    learning_rate=0.01,
    n_estimators=500,
    random_state=42
)
model.fit(X_train, y_train)
```

### Step 3: Permutation Importance
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

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std
}).sort_values('importance', ascending=False)

# Select top 30-40 features
top_features = importance_df.head(35)['feature'].tolist()
```

### Step 4: Retrain Final Model
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

## Critical Reminders

### Scale-Invariant Features (Safe for different ETFs)
✅ All returns/momentum (log returns)
✅ RSI, Hurst, BB position
✅ All binary features
✅ Normalized ratios (volume_ratio, atr_pct, etc.)
✅ Percentiles and z-scores

### Features to Normalize
❌ Never use: Absolute ATR, absolute volume, absolute MACD
✅ Always normalize by price or use ratios

### VIX-Specific Rules
- VIX is about **CHANGE and LEVEL**, not momentum trends
- VIX mean-reverts, so use changes and z-scores
- Don't use long-term momentum on VIX (meaningless)

### Hurst Exponent Power
- Enables regime-aware feature engineering
- Combine with other features for context-dependent signals
- Computationally expensive - consider caching

---

## Target Variable

```python
# 3-day forward log return
target_return = np.log(spy['Close'].shift(-3) / spy['Close'])

# Binary classification: 1 if up, 0 if down
y = (target_return > 0).astype(int)
```

---

## Notes

- **Total features before selection:** ~70-80
- **After correlation filter:** ~60-70
- **After permutation importance:** ~30-40 (final model)
- **Expected feature importance:** Hurst, BB position, VIX features, and momentum likely to be top performers
- **Computation time:** Hurst exponent will be the slowest to calculate
- **Testing:** Validate on multiple ETFs at different price levels to ensure generalizability
