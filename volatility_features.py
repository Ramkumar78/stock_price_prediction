"""
Volatility Features for XGBoost Stock Prediction Model

This module generates volatility-based features from OHLC data.
All features are normalized to be scale-invariant.

Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
"""

import pandas as pd
import numpy as np


def create_volatility_features(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Create all volatility-based features for XGBoost model.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all volatility features, indexed same as input
    """
    features = pd.DataFrame(index=spy.index)
    
    # ========== ATR (Normalized - Scale-Invariant) ==========
    # Calculate ATR
    high_low = spy['High'] - spy['Low']
    high_close = np.abs(spy['High'] - spy['Close'].shift(1))
    low_close = np.abs(spy['Low'] - spy['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    
    # Normalize ATR by price (scale-invariant)
    features['spy_atr_pct'] = atr_14 / spy['Close']
    
    # ATR percentile (relative to history)
    features['spy_atr_percentile'] = atr_14.rolling(100).rank(pct=True)
    
    # ATR z-score
    features['spy_atr_zscore'] = (
        (atr_14 - atr_14.rolling(50).mean()) / 
        atr_14.rolling(50).std()
    )
    
    # ========== ATR BINARY FEATURES ==========
    # High/low volatility regimes
    features['spy_high_volatility'] = (features['spy_atr_percentile'] > 0.8).astype(int)
    features['spy_low_volatility'] = (features['spy_atr_percentile'] < 0.2).astype(int)
    
    # Volatility expanding/contracting
    features['spy_vol_expanding'] = (atr_14 > atr_14.shift(5)).astype(int)
    features['spy_vol_contracting'] = (atr_14 < atr_14.shift(5)).astype(int)
    
    # ========== INTRADAY RANGE ==========
    # Daily range as percentage
    features['spy_daily_range_pct'] = (spy['High'] - spy['Low']) / spy['Close']
    
    # Wide/narrow range days
    range_percentile = features['spy_daily_range_pct'].rolling(20).rank(pct=True)
    features['spy_wide_range'] = (range_percentile > 0.8).astype(int)
    features['spy_narrow_range'] = (range_percentile < 0.2).astype(int)
    
    return features


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    """
    Example usage of the volatility features function.
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
    features_df = create_volatility_features(spy_data)
    
    print("Volatility Features Shape:", features_df.shape)
    print("\nFeature Names:")
    print(features_df.columns.tolist())
    print("\nFirst few rows:")
    print(features_df.head(20))
    print("\nFeature Statistics:")
    print(features_df.describe())
