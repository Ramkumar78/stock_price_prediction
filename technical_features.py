"""
Technical Indicator Features for XGBoost Stock Prediction Model

This module generates technical indicator features from OHLC data.
Includes Bollinger Bands, RSI, MACD, and Hurst Exponent.

Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
"""

import pandas as pd
import numpy as np


def create_technical_features(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Create all technical indicator features for XGBoost model.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all technical indicator features, indexed same as input
    """
    features = pd.DataFrame(index=spy.index)
    
    # ========== BOLLINGER BANDS ==========
    # Calculate BB
    sma_20 = spy['Close'].rolling(20).mean()
    std_20 = spy['Close'].rolling(20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # BB Position (0 to 1) - CRITICAL FEATURE
    features['spy_bb_position'] = (spy['Close'] - lower_band) / (upper_band - lower_band)
    
    # Normalized band width (scale-invariant)
    features['spy_band_width_pct'] = (upper_band - lower_band) / sma_20
    
    # BB Squeeze (low volatility)
    features['spy_bb_squeeze'] = (
        features['spy_band_width_pct'] < features['spy_band_width_pct'].rolling(100).quantile(0.2)
    ).astype(int)
    
    # ========== BB BINARY FEATURES ==========
    # Above/below bands
    features['spy_above_upper_bb'] = (spy['Close'] > upper_band).astype(int)
    features['spy_below_lower_bb'] = (spy['Close'] < lower_band).astype(int)
    
    # BB quartiles
    features['spy_bb_upper_quartile'] = (features['spy_bb_position'] > 0.75).astype(int)
    features['spy_bb_lower_quartile'] = (features['spy_bb_position'] < 0.25).astype(int)
    features['spy_bb_middle_zone'] = (
        (features['spy_bb_position'] >= 0.4) & 
        (features['spy_bb_position'] <= 0.6)
    ).astype(int)
    
    # Walking the band (riding upper band = strong trend)
    features['spy_bb_walking_upper'] = (
        (features['spy_bb_position'] > 0.8) & 
        (features['spy_bb_position'].shift(1) > 0.8) &
        (features['spy_bb_position'].shift(2) > 0.8)
    ).astype(int)
    
    # ========== RSI ==========
    # Calculate RSI
    rsi_14 = _calculate_rsi(spy['Close'], 14)
    features['spy_rsi_14'] = rsi_14
    
    # Binary RSI features
    features['spy_rsi_oversold'] = (rsi_14 < 30).astype(int)
    features['spy_rsi_overbought'] = (rsi_14 > 70).astype(int)
    features['spy_rsi_extreme_oversold'] = (rsi_14 < 20).astype(int)
    features['spy_rsi_extreme_overbought'] = (rsi_14 > 80).astype(int)
    features['spy_rsi_above_50'] = (rsi_14 > 50).astype(int)
    
    # ========== MACD (Normalized) ==========
    # Calculate MACD
    ema_12 = spy['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = spy['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Normalize by price (scale-invariant)
    features['spy_macd_pct'] = (ema_12 - ema_26) / spy['Close']
    features['spy_macd_hist_pct'] = (macd_line - signal_line) / spy['Close']
    
    # Alternative: MACD ratio
    features['spy_macd_ratio'] = ema_12 / ema_26 - 1
    
    # Binary MACD features
    features['spy_macd_positive'] = (macd_line > 0).astype(int)
    features['spy_macd_bullish_cross'] = (
        (macd_line > signal_line) & 
        (macd_line.shift(1) <= signal_line.shift(1))
    ).astype(int)
    features['spy_macd_bearish_cross'] = (
        (macd_line < signal_line) & 
        (macd_line.shift(1) >= signal_line.shift(1))
    ).astype(int)
    
    # ========== HURST EXPONENT (Trending vs Mean-Reverting) ==========
    # Rolling Hurst over 50-day window
    hurst_50d = spy['Close'].rolling(50).apply(
        lambda x: _hurst_exponent(x.values, max_lag=20), 
        raw=False
    )
    features['spy_hurst_50d'] = hurst_50d
    
    # Binary regime features - CRITICAL
    features['spy_trending_market'] = (hurst_50d > 0.6).astype(int)
    features['spy_mean_reverting_market'] = (hurst_50d < 0.4).astype(int)
    features['spy_strong_trend'] = (hurst_50d > 0.65).astype(int)
    
    return features


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    period : int
        RSI period (default: 14)
    
    Returns:
    --------
    pd.Series
        RSI values
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent.
    
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    Parameters:
    -----------
    ts : np.ndarray
        Time series data
    max_lag : int
        Maximum lag for calculation
    
    Returns:
    --------
    float
        Hurst exponent value
    """
    if len(ts) < max_lag:
        return np.nan
    
    try:
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        
        # Filter out any zero or invalid values
        valid_indices = [i for i, t in enumerate(tau) if t > 0]
        if len(valid_indices) < 2:
            return np.nan
        
        valid_lags = [list(lags)[i] for i in valid_indices]
        valid_tau = [tau[i] for i in valid_indices]
        
        poly = np.polyfit(np.log(valid_lags), np.log(valid_tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    """
    Example usage of the technical indicator features function.
    """
    # Example: Create sample data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    spy_data = pd.DataFrame({
        'Open': np.random.randn(300).cumsum() + 300,
        'High': np.random.randn(300).cumsum() + 305,
        'Low': np.random.randn(300).cumsum() + 295,
        'Close': np.random.randn(300).cumsum() + 300,
        'Volume': np.random.randint(50000000, 100000000, 300)
    }, index=dates)
    
    # Create features
    print("Calculating technical features (this may take a moment due to Hurst exponent)...")
    features_df = create_technical_features(spy_data)
    
    print("\nTechnical Features Shape:", features_df.shape)
    print("\nFeature Names:")
    print(features_df.columns.tolist())
    print("\nFirst few rows:")
    print(features_df.head(60))
    print("\nFeature Statistics:")
    print(features_df.describe())
