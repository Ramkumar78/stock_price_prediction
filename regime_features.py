"""
Binary Regime Features for XGBoost Stock Prediction Model

This module generates binary regime and pattern features from OHLC data.
Includes momentum regimes, consecutive moves, and combined regime indicators.

Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
"""

import pandas as pd
import numpy as np


def create_regime_features(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Create all binary regime features for XGBoost model.
    
    Note: This function requires some features from other modules for 
    combined regimes. You may need to pass additional parameters or 
    calculate them separately.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all regime features, indexed same as input
    """
    features = pd.DataFrame(index=spy.index)
    
    # ========== MOMENTUM REGIMES ==========
    # Calculate momentum for regime detection
    momentum_5d = np.log(spy['Close'] / spy['Close'].shift(5))
    momentum_20d = np.log(spy['Close'] / spy['Close'].shift(20))
    momentum_50d = np.log(spy['Close'] / spy['Close'].shift(50))
    
    # Momentum direction
    features['spy_mom_5d_positive'] = (momentum_5d > 0).astype(int)
    features['spy_mom_20d_positive'] = (momentum_20d > 0).astype(int)
    features['spy_mom_50d_positive'] = (momentum_50d > 0).astype(int)
    
    # All momentum windows aligned
    features['spy_all_mom_positive'] = (
        (features['spy_mom_5d_positive'] == 1) & 
        (features['spy_mom_20d_positive'] == 1) &
        (features['spy_mom_50d_positive'] == 1)
    ).astype(int)
    
    # Momentum reversal (bounce in downtrend)
    features['spy_mom_reversal'] = (
        (features['spy_mom_5d_positive'] == 1) & 
        (features['spy_mom_20d_positive'] == 0)
    ).astype(int)
    
    # ========== CONSECUTIVE MOVES ==========
    # Consecutive up/down days
    features['spy_three_up_days'] = (
        (spy['Close'] > spy['Close'].shift(1)) & 
        (spy['Close'].shift(1) > spy['Close'].shift(2)) &
        (spy['Close'].shift(2) > spy['Close'].shift(3))
    ).astype(int)
    
    features['spy_three_down_days'] = (
        (spy['Close'] < spy['Close'].shift(1)) & 
        (spy['Close'].shift(1) < spy['Close'].shift(2)) &
        (spy['Close'].shift(2) < spy['Close'].shift(3))
    ).astype(int)
    
    # ========== COMBINED REGIMES ==========
    # Calculate required components for combined regimes
    spy_above_200ma = (spy['Close'] > spy['Close'].rolling(200).mean()).astype(int)
    spy_above_50ma = (spy['Close'] > spy['Close'].rolling(50).mean()).astype(int)
    volume_above_avg = (spy['Volume'] > spy['Volume'].rolling(20).mean()).astype(int)
    
    # Strong bullish regime
    features['spy_strong_bull'] = (
        (spy_above_200ma == 1) & 
        (spy_above_50ma == 1) &
        (features['spy_mom_20d_positive'] == 1) &
        (volume_above_avg == 1)
    ).astype(int)
    
    # Calculate BB position for overbought/oversold setups
    sma_20 = spy['Close'].rolling(20).mean()
    std_20 = spy['Close'].rolling(20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    above_upper_bb = (spy['Close'] > upper_band).astype(int)
    below_lower_bb = (spy['Close'] < lower_band).astype(int)
    
    # Calculate RSI for setups
    rsi_14 = _calculate_rsi(spy['Close'], 14)
    rsi_overbought = (rsi_14 > 70).astype(int)
    rsi_oversold = (rsi_14 < 30).astype(int)
    
    # Heavy volume
    heavy_volume = (spy['Volume'] > spy['Volume'].rolling(50).quantile(0.8)).astype(int)
    
    # New lows
    new_low_20d = (spy['Close'] == spy['Low'].rolling(20).min()).astype(int)
    
    # Overbought setup (possible reversal)
    features['spy_overbought_setup'] = (
        (above_upper_bb == 1) & 
        (heavy_volume == 1) &
        (rsi_overbought == 1)
    ).astype(int)
    
    # Oversold bounce setup
    features['spy_oversold_setup'] = (
        (below_lower_bb == 1) & 
        (rsi_oversold == 1) &
        (new_low_20d == 1)
    ).astype(int)
    
    # Volume spike
    volume_spike = (spy['Volume'] > spy['Volume'].rolling(20).mean() * 2).astype(int)
    
    # Breakout with confirmation
    features['spy_breakout_confirmed'] = (
        (spy['Close'] > spy['High'].rolling(20).max().shift(1)) &
        (volume_spike == 1)
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


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    """
    Example usage of the regime features function.
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
    features_df = create_regime_features(spy_data)
    
    print("Regime Features Shape:", features_df.shape)
    print("\nFeature Names:")
    print(features_df.columns.tolist())
    print("\nFirst few rows:")
    print(features_df.head(60))
    print("\nFeature Statistics:")
    print(features_df.describe())
