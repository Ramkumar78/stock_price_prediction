"""
Volume Features for XGBoost Stock Prediction Model

This module generates volume-based features from OHLC data.
All features are scale-invariant using ratios, percentiles, and z-scores.

Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
"""

import pandas as pd
import numpy as np


def create_volume_features(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Create all volume-based features for XGBoost model.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all volume features, indexed same as input
    """
    features = pd.DataFrame(index=spy.index)
    
    # ========== VOLUME RATIOS (Scale-Invariant) ==========
    # Volume relative to averages
    features['spy_volume_ratio_10d'] = spy['Volume'] / spy['Volume'].rolling(10).mean()
    features['spy_volume_ratio_20d'] = spy['Volume'] / spy['Volume'].rolling(20).mean()
    features['spy_volume_ratio_50d'] = spy['Volume'] / spy['Volume'].rolling(50).mean()
    
    # Volume percentile
    features['spy_volume_percentile'] = spy['Volume'].rolling(50).rank(pct=True)
    
    # Volume z-score
    features['spy_volume_zscore'] = (
        (spy['Volume'] - spy['Volume'].rolling(50).mean()) / 
        spy['Volume'].rolling(50).std()
    )
    
    # ========== VOLUME BINARY FEATURES ==========
    # Volume above average
    features['spy_volume_above_avg'] = (
        spy['Volume'] > spy['Volume'].rolling(20).mean()
    ).astype(int)
    
    # Heavy volume (top 20%)
    features['spy_heavy_volume'] = (
        spy['Volume'] > spy['Volume'].rolling(50).quantile(0.8)
    ).astype(int)
    
    # Volume spike (>2x average)
    features['spy_volume_spike'] = (
        spy['Volume'] > spy['Volume'].rolling(20).mean() * 2
    ).astype(int)
    
    # Volume increasing trend
    features['spy_volume_increasing'] = (
        (spy['Volume'] > spy['Volume'].shift(1)) & 
        (spy['Volume'].shift(1) > spy['Volume'].shift(2))
    ).astype(int)
    
    # ========== PRICE-VOLUME CONFIRMATION ==========
    # Accumulation (up day + high volume)
    features['spy_accumulation'] = (
        (spy['Close'] > spy['Open']) & 
        (spy['Volume'] > spy['Volume'].rolling(20).mean())
    ).astype(int)
    
    # Distribution (down day + high volume)
    features['spy_distribution'] = (
        (spy['Close'] < spy['Open']) & 
        (spy['Volume'] > spy['Volume'].rolling(20).mean())
    ).astype(int)
    
    # Price-volume confirmation
    features['spy_price_vol_confirm'] = (
        (spy['Close'] > spy['Close'].shift(1)) & 
        (spy['Volume'] > spy['Volume'].shift(1))
    ).astype(int)
    
    # Price-volume divergence (weak rally)
    features['spy_price_vol_divergence'] = (
        (spy['Close'] > spy['Close'].shift(1)) & 
        (spy['Volume'] < spy['Volume'].shift(1))
    ).astype(int)
    
    return features


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    """
    Example usage of the volume features function.
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
    features_df = create_volume_features(spy_data)
    
    print("Volume Features Shape:", features_df.shape)
    print("\nFeature Names:")
    print(features_df.columns.tolist())
    print("\nFirst few rows:")
    print(features_df.head(10))
    print("\nFeature Statistics:")
    print(features_df.describe())
