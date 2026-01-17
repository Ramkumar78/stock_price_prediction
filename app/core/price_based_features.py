"""
Price-Based Features for XGBoost Stock Prediction Model

This module generates price-based features from OHLC data for SPY and other assets.
Features are designed to be scale-invariant and suitable for XGBoost classification.

Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def create_price_based_features(
    spy: pd.DataFrame,
    vix: Optional[pd.DataFrame] = None,
    tlt: Optional[pd.DataFrame] = None,
    dxy: Optional[pd.DataFrame] = None,
    gld: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create all price-based features for XGBoost model.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    vix : pd.DataFrame, optional
        VIX OHLC data
    tlt : pd.DataFrame, optional
        TLT (Bond ETF) OHLC data
    dxy : pd.DataFrame, optional
        DXY (Dollar Index) OHLC data
    gld : pd.DataFrame, optional
        GLD (Gold ETF) OHLC data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all price-based features, indexed same as input
    """
    features = pd.DataFrame(index=spy.index)
    
    # SPY Price-Based Features
    features = pd.concat([features, _create_spy_price_features(spy)], axis=1)
    
    # Cross-Asset Features
    if vix is not None:
        features = pd.concat([features, _create_vix_features(vix, spy)], axis=1)
    
    if tlt is not None:
        features = pd.concat([features, _create_tlt_features(tlt, spy)], axis=1)
    
    if dxy is not None:
        features = pd.concat([features, _create_dxy_features(dxy)], axis=1)
    
    if gld is not None:
        features = pd.concat([features, _create_gld_features(gld, spy)], axis=1)
    
    # Risk-On/Risk-Off Indicators (requires SPY, VIX, GLD)
    if vix is not None and gld is not None:
        features = pd.concat([features, _create_risk_regime_features(spy, vix, gld)], axis=1)
    
    return features


def _create_spy_price_features(spy: pd.DataFrame) -> pd.DataFrame:
    """Create SPY-specific price-based features."""
    features = pd.DataFrame(index=spy.index)
    
    # ========== MOMENTUM (Log Returns) - Multiple Windows ==========
    features['spy_momentum_3d'] = np.log(spy['Close'] / spy['Close'].shift(3))
    features['spy_momentum_5d'] = np.log(spy['Close'] / spy['Close'].shift(5))
    features['spy_momentum_10d'] = np.log(spy['Close'] / spy['Close'].shift(10))
    features['spy_momentum_20d'] = np.log(spy['Close'] / spy['Close'].shift(20))
    features['spy_momentum_50d'] = np.log(spy['Close'] / spy['Close'].shift(50))
    
    # ========== RATE OF CHANGE (Momentum Acceleration) ==========
    mom_10d = np.log(spy['Close'] / spy['Close'].shift(10))
    features['spy_roc_5d'] = mom_10d - mom_10d.shift(5)
    
    # Alternative: recent vs older momentum
    recent_mom = np.log(spy['Close'] / spy['Close'].shift(5))
    older_mom = np.log(spy['Close'].shift(5) / spy['Close'].shift(10))
    features['spy_mom_acceleration'] = recent_mom - older_mom
    
    # ========== MOVING AVERAGE FEATURES ==========
    # Binary: Above/Below key MAs
    features['spy_above_20ma'] = (spy['Close'] > spy['Close'].rolling(20).mean()).astype(int)
    features['spy_above_50ma'] = (spy['Close'] > spy['Close'].rolling(50).mean()).astype(int)
    features['spy_above_100ma'] = (spy['Close'] > spy['Close'].rolling(100).mean()).astype(int)
    features['spy_above_200ma'] = (spy['Close'] > spy['Close'].rolling(200).mean()).astype(int)
    
    # Golden/Death Cross
    ma_50 = spy['Close'].rolling(50).mean()
    ma_200 = spy['Close'].rolling(200).mean()
    features['spy_ma_50_above_200'] = (ma_50 > ma_200).astype(int)
    
    # All MAs aligned (strong trend)
    features['spy_all_mas_bullish'] = (
        (spy['Close'] > spy['Close'].rolling(20).mean()) & 
        (spy['Close'] > spy['Close'].rolling(50).mean()) &
        (spy['Close'] > spy['Close'].rolling(200).mean())
    ).astype(int)
    
    # ========== HIGH/LOW FEATURES ==========
    # New highs/lows
    features['spy_new_high_20d'] = (spy['Close'] == spy['High'].rolling(20).max()).astype(int)
    features['spy_new_high_50d'] = (spy['Close'] == spy['High'].rolling(50).max()).astype(int)
    features['spy_new_low_20d'] = (spy['Close'] == spy['Low'].rolling(20).min()).astype(int)
    
    # Near highs/lows (within 2%)
    features['spy_near_high_20d'] = (spy['Close'] > spy['High'].rolling(20).max() * 0.98).astype(int)
    features['spy_near_low_20d'] = (spy['Close'] < spy['Low'].rolling(20).min() * 1.02).astype(int)
    
    # ========== GAP FEATURES ==========
    # Gap as percentage (scale-invariant)
    features['spy_gap_pct'] = (spy['Open'] - spy['Close'].shift(1)) / spy['Close'].shift(1)
    
    # Binary gap indicators
    features['spy_gap_up'] = (spy['Open'] > spy['Close'].shift(1) * 1.002).astype(int)  # >0.2% gap
    features['spy_gap_down'] = (spy['Open'] < spy['Close'].shift(1) * 0.998).astype(int)
    
    # Gap filled
    gap_up = (spy['Open'] > spy['Close'].shift(1) * 1.002).astype(int)
    features['spy_gap_filled'] = ((gap_up == 1) & (spy['Low'] <= spy['Close'].shift(1))).astype(int)
    
    return features


def _create_vix_features(vix: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Create VIX-specific price-based features."""
    features = pd.DataFrame(index=vix.index)
    
    # ========== VIX LEVEL FEATURES ==========
    features['vix_level'] = vix['Close']
    
    # VIX binary regimes
    features['vix_low'] = (vix['Close'] < 15).astype(int)  # Complacency
    features['vix_elevated'] = (vix['Close'] > 20).astype(int)  # Fear
    features['vix_panic'] = (vix['Close'] > 30).astype(int)  # Panic
    
    # ========== VIX CHANGE FEATURES (NOT momentum - VIX mean-reverts) ==========
    # Absolute changes (VIX spikes/collapses)
    features['vix_change_1d'] = vix['Close'] - vix['Close'].shift(1)
    features['vix_change_5d'] = vix['Close'] - vix['Close'].shift(5)
    
    # Percentage change
    features['vix_pct_change_1d'] = vix['Close'] / vix['Close'].shift(1) - 1
    
    # VIX spike detection
    features['vix_spike'] = (vix['Close'] > vix['Close'].shift(1) * 1.1).astype(int)  # 10% jump
    
    # ========== VIX MEAN REVERSION FEATURES ==========
    # VIX z-score (stretched = likely to revert)
    vix_sma_20 = vix['Close'].rolling(20).mean()
    features['vix_zscore'] = (vix['Close'] - vix_sma_20) / vix['Close'].rolling(20).std()
    
    # VIX stretched (mean reversion signal)
    features['vix_stretched_high'] = (features['vix_zscore'] > 2).astype(int)
    features['vix_stretched_low'] = (features['vix_zscore'] < -1).astype(int)
    
    # ========== VIX VELOCITY ==========
    # Rate of change of VIX changes (acceleration/deceleration)
    features['vix_velocity'] = features['vix_change_1d'] - features['vix_change_1d'].shift(1)
    
    # ========== SPY-VIX RELATIONSHIP ==========
    # Align indices if needed
    if len(spy) == len(vix) and (spy.index == vix.index).all():
        # SPY return
        spy_return_1d = np.log(spy['Close'] / spy['Close'].shift(1))
        
        # Decoupling (SPY down but VIX NOT rising = bullish resilience)
        features['spy_vix_decoupling'] = (
            (spy_return_1d < 0) & (features['vix_change_1d'] < 0)
        ).astype(int)
        
        # VIX/SPY ratio (normalized fear)
        features['vix_spy_ratio'] = vix['Close'] / spy['Close']
    
    return features


def _create_tlt_features(tlt: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Create TLT (Bond ETF) price-based features."""
    features = pd.DataFrame(index=tlt.index)
    
    # ========== TLT MOMENTUM ==========
    features['tlt_momentum_20d'] = np.log(tlt['Close'] / tlt['Close'].shift(20))
    
    # ========== TLT MOVING AVERAGE ==========
    # TLT above/below MA (rates falling/rising)
    features['tlt_above_50ma'] = (tlt['Close'] > tlt['Close'].rolling(50).mean()).astype(int)
    
    # ========== SPY-TLT CORRELATION ==========
    # Align indices if needed
    if len(spy) == len(tlt) and (spy.index == tlt.index).all():
        spy_returns = np.log(spy['Close'] / spy['Close'].shift(1))
        tlt_returns = np.log(tlt['Close'] / tlt['Close'].shift(1))
        features['spy_tlt_corr'] = spy_returns.rolling(20).corr(tlt_returns)
    
    return features


def _create_dxy_features(dxy: pd.DataFrame) -> pd.DataFrame:
    """Create DXY (Dollar Index) price-based features."""
    features = pd.DataFrame(index=dxy.index)
    
    # ========== DXY MOMENTUM ==========
    # Strong dollar = headwind for equities
    features['dxy_momentum_20d'] = np.log(dxy['Close'] / dxy['Close'].shift(20))
    
    # ========== DXY MOVING AVERAGE ==========
    # DXY above MA (dollar strength)
    features['dxy_above_50ma'] = (dxy['Close'] > dxy['Close'].rolling(50).mean()).astype(int)
    
    # ========== DXY TREND ==========
    features['dxy_rising'] = (features['dxy_momentum_20d'] > 0).astype(int)
    
    return features


def _create_gld_features(gld: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Create GLD (Gold ETF) price-based features."""
    features = pd.DataFrame(index=gld.index)
    
    # ========== GLD MOMENTUM ==========
    features['gld_momentum_20d'] = np.log(gld['Close'] / gld['Close'].shift(20))
    
    # ========== GLD MOVING AVERAGE ==========
    features['gld_above_50ma'] = (gld['Close'] > gld['Close'].rolling(50).mean()).astype(int)
    
    # ========== GLD vs SPY ==========
    # Align indices if needed
    if len(spy) == len(gld) and (spy.index == gld.index).all():
        spy_momentum_20d = np.log(spy['Close'] / spy['Close'].shift(20))
        # GLD outperforming SPY (flight to safety)
        features['gld_outperform'] = (features['gld_momentum_20d'] > spy_momentum_20d).astype(int)
    
    return features


def _create_risk_regime_features(
    spy: pd.DataFrame, 
    vix: pd.DataFrame, 
    gld: pd.DataFrame
) -> pd.DataFrame:
    """Create risk-on/risk-off regime indicators."""
    features = pd.DataFrame(index=spy.index)
    
    # Ensure indices are aligned
    if len(spy) == len(vix) == len(gld) and \
       (spy.index == vix.index).all() and (spy.index == gld.index).all():
        
        # Risk-on: SPY up + VIX down
        features['risk_on'] = (
            (spy['Close'] > spy['Close'].shift(1)) & 
            (vix['Close'] < vix['Close'].shift(1))
        ).astype(int)
        
        # Risk-off: SPY down + VIX up + GLD up
        features['risk_off'] = (
            (spy['Close'] < spy['Close'].shift(1)) & 
            (vix['Close'] > vix['Close'].shift(1)) &
            (gld['Close'] > gld['Close'].shift(1))
        ).astype(int)
    
    return features


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    """
    Example usage of the price-based features function.
    """
    # Example: Create sample data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # Sample SPY data
    spy_data = pd.DataFrame({
        'Open': np.random.randn(300).cumsum() + 300,
        'High': np.random.randn(300).cumsum() + 305,
        'Low': np.random.randn(300).cumsum() + 295,
        'Close': np.random.randn(300).cumsum() + 300,
        'Volume': np.random.randint(50000000, 100000000, 300)
    }, index=dates)
    
    # Sample VIX data
    vix_data = pd.DataFrame({
        'Open': np.abs(np.random.randn(300)) + 15,
        'High': np.abs(np.random.randn(300)) + 16,
        'Low': np.abs(np.random.randn(300)) + 14,
        'Close': np.abs(np.random.randn(300)) + 15,
        'Volume': np.random.randint(1000000, 5000000, 300)
    }, index=dates)
    
    # Create features
    features_df = create_price_based_features(
        spy=spy_data,
        vix=vix_data
    )
    
    print("Price-Based Features Shape:", features_df.shape)
    print("\nFeature Names:")
    print(features_df.columns.tolist())
    print("\nFirst few rows:")
    print(features_df.head(10))
    print("\nFeature Statistics:")
    print(features_df.describe())
