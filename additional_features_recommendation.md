# Additional Feature Recommendations for 3-Day Market Prediction

This document outlines recommended additional features to enhance the XGBoost model for predicting 3-day SPY market direction.

---

## 🎯 Priority Summary

| Priority | Category | Effort | Expected Lift |
|----------|----------|--------|---------------|
| 🥇 | Calendar/Seasonality | Low | Medium-High |
| 🥇 | VIX Term Structure | Low | High |
| 🥈 | Put/Call Ratio | Medium | High |
| 🥈 | Pattern Recognition | Low | Medium |
| 🥉 | Market Breadth | High | Medium |
| 🥉 | Cross-Asset Expansion | Medium | Medium |

---

## 1. Calendar & Seasonality Features (High Priority)

These are often overlooked but can be surprisingly predictive for short-term movements.

### Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `day_of_week` | 0=Monday, 4=Friday | Mondays often gap, Fridays see position squaring |
| `is_monday` | Binary | Monday effect (historically weaker) |
| `is_friday` | Binary | Weekend risk avoidance |
| `is_month_end` | Last 3 trading days | Rebalancing flows from funds |
| `is_month_start` | First 3 trading days | Fresh capital deployment |
| `is_opex_week` | Options expiration week | 3rd Friday = high volatility regime |
| `days_to_opex` | Days until next OpEx | Event risk countdown |
| `is_quarter_end` | Mar, Jun, Sep, Dec end | Window dressing by funds |
| `week_of_month` | 1-5 | First week often bullish |
| `trading_day_of_year` | 1-252 | Seasonal patterns (Jan effect, May sell-off) |

### Implementation

```python
def create_calendar_features(spy: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=spy.index)
    
    # Basic day features
    features['day_of_week'] = spy.index.dayofweek
    features['is_monday'] = (spy.index.dayofweek == 0).astype(int)
    features['is_friday'] = (spy.index.dayofweek == 4).astype(int)
    
    # Month position
    features['is_month_end'] = (
        spy.index.to_series().groupby(spy.index.to_period('M')).transform(
            lambda x: x >= x.iloc[-3]
        )
    ).astype(int)
    
    features['is_month_start'] = (
        spy.index.to_series().groupby(spy.index.to_period('M')).transform(
            lambda x: x <= x.iloc[2]
        )
    ).astype(int)
    
    # Quarter end
    features['is_quarter_end'] = (
        (spy.index.month.isin([3, 6, 9, 12])) & 
        (features['is_month_end'] == 1)
    ).astype(int)
    
    # OpEx detection (3rd Friday of each month)
    def is_opex(date):
        third_friday = pd.date_range(
            start=date.replace(day=1), 
            end=date.replace(day=28),
            freq='WOM-3FRI'
        )
        if len(third_friday) > 0:
            return abs((date - third_friday[0]).days) <= 2
        return False
    
    features['is_opex_week'] = spy.index.to_series().apply(is_opex).astype(int)
    
    return features
```

---

## 2. VIX Term Structure Features (High Priority)

The shape of the VIX curve is highly predictive of short-term market direction.

### Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `vix_term_spread` | VIX - VIX3M | Contango = complacency, backwardation = fear |
| `vix_term_ratio` | VIX / VIX3M | Normalized spread |
| `vix_in_backwardation` | Binary: VIX > VIX3M | Stress indicator |
| `vix_term_zscore` | Z-score of term spread | Extreme readings |
| `vvix_level` | VIX of VIX | Volatility uncertainty |
| `vvix_vix_ratio` | VVIX / VIX | Normalized fear of fear |

### Data Required

- **VIX3M** (3-month VIX): CBOE ticker `^VIX3M`
- **VVIX** (VIX of VIX): CBOE ticker `^VVIX`

### Implementation

```python
def create_vix_term_features(vix: pd.DataFrame, vix3m: pd.DataFrame, vvix: pd.DataFrame = None) -> pd.DataFrame:
    features = pd.DataFrame(index=vix.index)
    
    # Term structure
    features['vix_term_spread'] = vix['Close'] - vix3m['Close']
    features['vix_term_ratio'] = vix['Close'] / vix3m['Close']
    features['vix_in_backwardation'] = (features['vix_term_ratio'] > 1).astype(int)
    
    # Z-score of term spread
    features['vix_term_zscore'] = (
        (features['vix_term_spread'] - features['vix_term_spread'].rolling(50).mean()) /
        features['vix_term_spread'].rolling(50).std()
    )
    
    # Extreme backwardation (stress)
    features['vix_extreme_backwardation'] = (features['vix_term_ratio'] > 1.1).astype(int)
    
    # VVIX features (if available)
    if vvix is not None:
        features['vvix_level'] = vvix['Close']
        features['vvix_vix_ratio'] = vvix['Close'] / vix['Close']
        features['vvix_elevated'] = (vvix['Close'] > 100).astype(int)
    
    return features
```

---

## 3. Options Sentiment Features (Medium-High Priority)

Options market data provides forward-looking sentiment.

### Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `put_call_ratio` | Put volume / Call volume | Contrarian: high = bullish, low = bearish |
| `put_call_oi_ratio` | Put OI / Call OI | Open interest ratio |
| `pcr_zscore` | Z-score of P/C ratio | Extreme sentiment |
| `pcr_5d_change` | Change in P/C over 5 days | Sentiment momentum |
| `skew_index` | CBOE SKEW | Tail risk demand |

### Data Required

- **CBOE Put/Call Ratio**: Available from CBOE website or data providers
- **SKEW Index**: CBOE ticker `^SKEW`

### Implementation

```python
def create_options_sentiment_features(pcr: pd.DataFrame, skew: pd.DataFrame = None) -> pd.DataFrame:
    features = pd.DataFrame(index=pcr.index)
    
    # Put/Call ratio features
    features['put_call_ratio'] = pcr['Close']
    features['pcr_zscore'] = (
        (pcr['Close'] - pcr['Close'].rolling(20).mean()) /
        pcr['Close'].rolling(20).std()
    )
    features['pcr_5d_change'] = pcr['Close'] - pcr['Close'].shift(5)
    
    # Extreme sentiment (contrarian signals)
    features['pcr_extreme_fear'] = (features['pcr_zscore'] > 2).astype(int)  # Bullish
    features['pcr_extreme_greed'] = (features['pcr_zscore'] < -2).astype(int)  # Bearish
    
    # SKEW features
    if skew is not None:
        features['skew_index'] = skew['Close']
        features['skew_elevated'] = (skew['Close'] > 140).astype(int)
        features['skew_zscore'] = (
            (skew['Close'] - skew['Close'].rolling(50).mean()) /
            skew['Close'].rolling(50).std()
        )
    
    return features
```

---

## 4. Candlestick Pattern Features (Medium Priority)

Classic price patterns that can signal reversals or continuations.

### Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `inside_day` | Today's range within yesterday's | Consolidation before move |
| `outside_day` | Today's range engulfs yesterday's | Expansion/reversal |
| `doji` | Open ≈ Close | Indecision |
| `hammer` | Long lower wick, body near high | Bullish reversal |
| `shooting_star` | Long upper wick, body near low | Bearish reversal |
| `bullish_engulfing` | Green candle engulfs red | Bullish reversal |
| `bearish_engulfing` | Red candle engulfs green | Bearish reversal |
| `morning_star` | 3-candle bullish reversal | Strong bullish signal |
| `evening_star` | 3-candle bearish reversal | Strong bearish signal |

### Implementation

```python
def create_pattern_features(spy: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=spy.index)
    
    body = spy['Close'] - spy['Open']
    body_abs = body.abs()
    upper_wick = spy['High'] - spy[['Open', 'Close']].max(axis=1)
    lower_wick = spy[['Open', 'Close']].min(axis=1) - spy['Low']
    range_hl = spy['High'] - spy['Low']
    
    # Inside/Outside Day
    features['inside_day'] = (
        (spy['High'] < spy['High'].shift(1)) & 
        (spy['Low'] > spy['Low'].shift(1))
    ).astype(int)
    
    features['outside_day'] = (
        (spy['High'] > spy['High'].shift(1)) & 
        (spy['Low'] < spy['Low'].shift(1))
    ).astype(int)
    
    # Doji (body < 10% of range)
    features['doji'] = (body_abs < range_hl * 0.1).astype(int)
    
    # Hammer (lower wick > 2x body, small upper wick, body near high)
    features['hammer'] = (
        (lower_wick > 2 * body_abs) &
        (upper_wick < body_abs * 0.5) &
        (spy['Close'] > spy['Open'])  # Green hammer
    ).astype(int)
    
    # Shooting Star (upper wick > 2x body, small lower wick)
    features['shooting_star'] = (
        (upper_wick > 2 * body_abs) &
        (lower_wick < body_abs * 0.5) &
        (spy['Close'] < spy['Open'])  # Red shooting star
    ).astype(int)
    
    # Bullish Engulfing
    features['bullish_engulfing'] = (
        (spy['Close'].shift(1) < spy['Open'].shift(1)) &  # Yesterday was red
        (spy['Close'] > spy['Open']) &  # Today is green
        (spy['Open'] < spy['Close'].shift(1)) &  # Open below yesterday close
        (spy['Close'] > spy['Open'].shift(1))  # Close above yesterday open
    ).astype(int)
    
    # Bearish Engulfing
    features['bearish_engulfing'] = (
        (spy['Close'].shift(1) > spy['Open'].shift(1)) &  # Yesterday was green
        (spy['Close'] < spy['Open']) &  # Today is red
        (spy['Open'] > spy['Close'].shift(1)) &  # Open above yesterday close
        (spy['Close'] < spy['Open'].shift(1))  # Close below yesterday open
    ).astype(int)
    
    return features
```

---

## 5. Market Breadth Features (Medium Priority - Higher Effort)

Internal market health indicators. Requires additional data sources.

### Features

| Feature | Description | Data Source |
|---------|-------------|-------------|
| `pct_above_200ma` | % of S&P 500 stocks above 200-day MA | $MMTH or calculate |
| `pct_above_50ma` | % of S&P 500 stocks above 50-day MA | $MMFI or calculate |
| `advance_decline_ratio` | Advancing / Declining issues | $ADD |
| `advance_decline_line` | Cumulative A-D | Calculated from $ADD |
| `new_highs_lows_diff` | New 52-week highs - lows | NYSE data |
| `mcclellan_oscillator` | 19-day EMA - 39-day EMA of A-D ratio | Calculated |

### Implementation

```python
def create_breadth_features(pct_above_200ma: pd.Series, add: pd.Series = None) -> pd.DataFrame:
    features = pd.DataFrame(index=pct_above_200ma.index)
    
    # Percent above MA
    features['pct_above_200ma'] = pct_above_200ma
    features['breadth_oversold'] = (pct_above_200ma < 30).astype(int)
    features['breadth_overbought'] = (pct_above_200ma > 80).astype(int)
    
    # Breadth momentum
    features['breadth_momentum'] = pct_above_200ma - pct_above_200ma.shift(5)
    
    if add is not None:
        # Advance-Decline features
        features['ad_ratio'] = add
        features['ad_line'] = add.cumsum()
        
        # McClellan Oscillator
        ad_19ema = add.ewm(span=19, adjust=False).mean()
        ad_39ema = add.ewm(span=39, adjust=False).mean()
        features['mcclellan_osc'] = ad_19ema - ad_39ema
    
    return features
```

---

## 6. Cross-Asset Expansion (Medium Priority)

Additional cross-asset features to capture macro risk appetite.

### Features

| Feature | Ticker | Rationale |
|---------|--------|-----------|
| `hyg_lqd_spread` | HYG - LQD returns | Credit stress/risk appetite |
| `eem_momentum` | EEM 20d momentum | Emerging market risk |
| `iwm_spy_ratio` | IWM/SPY | Small cap leadership |
| `rsp_spy_ratio` | RSP/SPY | Concentration vs breadth |
| `copper_gold_ratio` | CPER/GLD | Economic optimism |
| `oil_momentum` | USO 10d momentum | Energy/inflation |

### Implementation

```python
def create_cross_asset_features(
    hyg: pd.DataFrame, lqd: pd.DataFrame,
    eem: pd.DataFrame, iwm: pd.DataFrame,
    spy: pd.DataFrame
) -> pd.DataFrame:
    features = pd.DataFrame(index=spy.index)
    
    # Credit spread proxy
    hyg_ret = np.log(hyg['Close'] / hyg['Close'].shift(1))
    lqd_ret = np.log(lqd['Close'] / lqd['Close'].shift(1))
    features['credit_spread_change'] = lqd_ret - hyg_ret  # Widening = risk-off
    features['credit_spread_20d'] = features['credit_spread_change'].rolling(20).sum()
    
    # Emerging markets
    features['eem_momentum_20d'] = np.log(eem['Close'] / eem['Close'].shift(20))
    features['eem_outperform'] = (
        features['eem_momentum_20d'] > np.log(spy['Close'] / spy['Close'].shift(20))
    ).astype(int)
    
    # Small cap vs large cap
    features['iwm_spy_ratio'] = iwm['Close'] / spy['Close']
    features['small_cap_leading'] = (
        features['iwm_spy_ratio'] > features['iwm_spy_ratio'].shift(10)
    ).astype(int)
    
    return features
```

---

## 7. 3-Day Horizon-Specific Features

Features specifically tuned to the 3-day prediction window.

### Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `momentum_3d_lagged` | 3d momentum from 3 days ago | Recent trend persistence |
| `reversal_rate_3d` | Rolling % of 3d reversals | Mean reversion likelihood |
| `range_3d` | 3-day high-low range | Short-term volatility |
| `autocorr_3d` | Returns autocorrelation at lag 3 | Trending vs mean-reverting |
| `streak_3d` | Consecutive 3-day moves | Momentum exhaustion |

### Implementation

```python
def create_horizon_features(spy: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    features = pd.DataFrame(index=spy.index)
    
    # 3-day momentum, lagged
    momentum_3d = np.log(spy['Close'] / spy['Close'].shift(horizon))
    features['momentum_3d_lagged'] = momentum_3d.shift(horizon)
    
    # 3-day reversal rate
    current_dir = (momentum_3d > 0).astype(int)
    prev_dir = (momentum_3d.shift(horizon) > 0).astype(int)
    reversal = (current_dir != prev_dir).astype(int)
    features['reversal_rate_3d'] = reversal.rolling(20).mean()
    
    # 3-day range
    features['range_3d'] = (
        (spy['High'].rolling(horizon).max() - spy['Low'].rolling(horizon).min()) / 
        spy['Close']
    )
    
    # Autocorrelation at lag 3
    daily_ret = np.log(spy['Close'] / spy['Close'].shift(1))
    features['autocorr_3d'] = daily_ret.rolling(60).apply(
        lambda x: x.autocorr(lag=horizon) if len(x) > horizon else np.nan, 
        raw=False
    )
    
    # Streak detection
    features['three_up_3d'] = (
        (momentum_3d > 0) & 
        (momentum_3d.shift(horizon) > 0)
    ).astype(int)
    
    return features
```

---

## 📋 Implementation Roadmap

### Phase 1 (Quick Wins - 1-2 hours)
1. ✅ `calendar_features.py` - No external data needed
2. ✅ `pattern_features.py` - Uses existing OHLC data
3. ✅ `horizon_features.py` - Uses existing OHLC data

### Phase 2 (Medium Effort - 1 day)
4. `vix_term_features.py` - Need VIX3M and VVIX data
5. `options_sentiment_features.py` - Need P/C ratio and SKEW data

### Phase 3 (Higher Effort - 2-3 days)
6. `breadth_features.py` - Need $MMTH, $ADD data
7. `cross_asset_features.py` - Need HYG, LQD, EEM, IWM data

---

## ⚠️ Data Sources

| Data | Free Sources | Paid Sources |
|------|--------------|--------------|
| VIX3M, VVIX, SKEW | Yahoo Finance (`^VIX3M`, `^VVIX`, `^SKEW`) | CBOE DataShop |
| Put/Call Ratio | CBOE website (daily download) | Quandl, CBOE |
| Market Breadth | StockCharts ($MMTH, $MMFI) | Bloomberg, Refinitiv |
| ETFs (HYG, LQD, EEM, IWM) | Yahoo Finance | Any data provider |

---

## 🔄 Integration with Existing Code

Add new modules to `feature_engineering.py`:

```python
# In create_all_features():
from calendar_features import create_calendar_features
from pattern_features import create_pattern_features
from vix_term_features import create_vix_term_features

# Add after existing features
print("  [7/9] Creating calendar features...")
calendar_features_df = create_calendar_features(spy)
all_features = pd.concat([all_features, calendar_features_df], axis=1)

print("  [8/9] Creating pattern features...")
pattern_features_df = create_pattern_features(spy)
all_features = pd.concat([all_features, pattern_features_df], axis=1)

# If VIX3M data is available
if vix3m is not None:
    print("  [9/9] Creating VIX term structure features...")
    vix_term_features_df = create_vix_term_features(vix, vix3m, vvix)
    all_features = pd.concat([all_features, vix_term_features_df], axis=1)
```

---

## Expected Impact

| Feature Category | Expected AUC Lift | Confidence |
|-----------------|-------------------|------------|
| Calendar | +0.01 - 0.02 | High |
| VIX Term Structure | +0.02 - 0.04 | High |
| Options Sentiment | +0.02 - 0.03 | Medium |
| Candlestick Patterns | +0.005 - 0.015 | Medium |
| Market Breadth | +0.01 - 0.025 | Medium |
| Cross-Asset | +0.01 - 0.02 | Medium |

**Note**: Actual improvement depends on feature interaction with existing features and market regime during test period.
