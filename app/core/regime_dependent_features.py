"""
Regime-Dependent Features for XGBoost Stock Prediction Model

This module generates regime-dependent interaction features.
These features combine technical indicators with market regime (trending vs mean-reverting).

Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
"""

import pandas as pd
import numpy as np


def create_regime_dependent_features(
    spy: pd.DataFrame,
    rsi_14: pd.Series = None,
    bb_position: pd.Series = None,
    macd_hist_pct: pd.Series = None,
    hurst_50d: pd.Series = None
) -> pd.DataFrame:
    """
    Create regime-dependent interaction features for XGBoost model.
    
    These features combine technical indicators with market regime detection
    to create context-aware signals.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    rsi_14 : pd.Series, optional
        Pre-calculated RSI values (will calculate if not provided)
    bb_position : pd.Series, optional
        Pre-calculated Bollinger Band position (will calculate if not provided)
    macd_hist_pct : pd.Series, optional
        Pre-calculated MACD histogram percentage (will calculate if not provided)
    hurst_50d : pd.Series, optional
        Pre-calculated Hurst exponent (will calculate if not provided)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all regime-dependent features, indexed same as input
    """
    features = pd.DataFrame(index=spy.index)
    
    # Calculate Hurst if not provided
    if hurst_50d is None:
        hurst_50d = spy['Close'].rolling(50).apply(
            lambda x: _hurst_exponent(x.values, max_lag=20), 
            raw=False
        )
    
    # Define regimes
    trending_market = (hurst_50d > 0.6).astype(int)
    mean_reverting_market = (hurst_50d < 0.4).astype(int)
    
    # Calculate RSI if not provided
    if rsi_14 is None:
        rsi_14 = _calculate_rsi(spy['Close'], 14)
    
    # Calculate BB position if not provided
    if bb_position is None:
        sma_20 = spy['Close'].rolling(20).mean()
        std_20 = spy['Close'].rolling(20).std()
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        bb_position = (spy['Close'] - lower_band) / (upper_band - lower_band)
    
    # Calculate MACD if not provided
    if macd_hist_pct is None:
        ema_12 = spy['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = spy['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_pct = (macd_line - signal_line) / spy['Close']
    
    # ========== INTERACTION TERMS (Hurst × Other Features) ==========
    # RSI behavior differs in trending vs mean-reverting markets
    features['spy_rsi_in_trending'] = rsi_14 * trending_market
    features['spy_rsi_in_mr'] = rsi_14 * mean_reverting_market
    
    # MACD in trending markets
    features['spy_macd_trending'] = macd_hist_pct * trending_market
    
    # BB position in mean-reverting markets
    features['spy_bb_pos_mr'] = bb_position * mean_reverting_market
    
    # Volume spike interpretation depends on regime
    volume_spike = (spy['Volume'] > spy['Volume'].rolling(20).mean() * 2).astype(int)
    features['spy_vol_spike_trending'] = volume_spike * trending_market
    features['spy_vol_spike_mr'] = volume_spike * mean_reverting_market
    
    # ========== REGIME-AWARE STRATEGIES ==========
    # Calculate required components
    momentum_20d = np.log(spy['Close'] / spy['Close'].shift(20))
    mom_20d_positive = (momentum_20d > 0).astype(int)
    spy_above_50ma = (spy['Close'] > spy['Close'].rolling(50).mean()).astype(int)
    rsi_overbought = (rsi_14 > 70).astype(int)
    
    # Trend continuation (in trending market)
    features['spy_trend_continuation'] = (
        (trending_market == 1) & 
        (mom_20d_positive == 1) &
        (spy_above_50ma == 1)
    ).astype(int)
    
    # Mean reversion setup (in mean-reverting market)
    features['spy_mean_reversion_setup'] = (
        (mean_reverting_market == 1) & 
        (rsi_overbought == 1) &
        (bb_position > 0.8)
    ).astype(int)
    
    # Regime shift detection
    features['spy_regime_shift_to_mr'] = (
        (trending_market.shift(5) == 1) & 
        (mean_reverting_market == 1)
    ).astype(int)
    
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
    Example usage of the regime-dependent features function.
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
    print("Calculating regime-dependent features (this may take a moment)...")
    features_df = create_regime_dependent_features(spy_data)
    
    print("\nRegime-Dependent Features Shape:", features_df.shape)
    print("\nFeature Names:")
    print(features_df.columns.tolist())
    print("\nFirst few rows:")
    print(features_df.head(60))
    print("\nFeature Statistics:")
    print(features_df.describe())
