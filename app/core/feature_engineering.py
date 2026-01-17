"""
Master Feature Engineering Module for XGBoost Stock Prediction Model

This module combines all feature engineering modules into a single interface.
It generates all features from OHLC data for SPY and cross-asset data.

Usage:
    from feature_engineering import create_all_features
    
    features = create_all_features(
        spy=spy_df,
        vix=vix_df,
        tlt=tlt_df,
        dxy=dxy_df,
        gld=gld_df
    )
"""

import pandas as pd
import os
import numpy as np
from typing import Optional

# Import all feature modules
from .price_based_features import create_price_based_features
from .volume_features import create_volume_features
from .volatility_features import create_volatility_features
from .technical_features import create_technical_features
from .regime_features import create_regime_features
from .regime_dependent_features import create_regime_dependent_features


def create_all_features(
    spy: pd.DataFrame,
    vix: Optional[pd.DataFrame] = None,
    tlt: Optional[pd.DataFrame] = None,
    dxy: Optional[pd.DataFrame] = None,
    gld: Optional[pd.DataFrame] = None,
    include_regime_dependent: bool = True
) -> pd.DataFrame:
    """
    Create all features for XGBoost stock prediction model.
    
    This function combines all feature engineering modules:
    - Price-based features (momentum, MAs, gaps, cross-asset)
    - Volume features (ratios, price-volume confirmation)
    - Volatility features (ATR, intraday range)
    - Technical indicators (BB, RSI, MACD, Hurst)
    - Binary regime features (momentum regimes, patterns)
    - Regime-dependent features (interaction terms)
    
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
    include_regime_dependent : bool, default=True
        Whether to include regime-dependent features (computationally expensive)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all features, indexed same as SPY input
        
    Notes:
    ------
    - Total features: ~70-80 before selection
    - Hurst exponent calculation is computationally expensive
    - All features are scale-invariant (safe for different ETFs)
    - Features will have NaN values for initial rows due to rolling windows
    """
    print("Starting feature engineering...")
    all_features = pd.DataFrame(index=spy.index)
    
    # 1. Price-Based Features
    print("  [1/6] Creating price-based features...")
    price_features = create_price_based_features(
        spy=spy,
        vix=vix,
        tlt=tlt,
        dxy=dxy,
        gld=gld
    )
    all_features = pd.concat([all_features, price_features], axis=1)
    print(f"        Added {len(price_features.columns)} price-based features")
    
    # 2. Volume Features
    print("  [2/6] Creating volume features...")
    volume_features_df = create_volume_features(spy)
    all_features = pd.concat([all_features, volume_features_df], axis=1)
    print(f"        Added {len(volume_features_df.columns)} volume features")
    
    # 3. Volatility Features
    print("  [3/6] Creating volatility features...")
    volatility_features_df = create_volatility_features(spy)
    all_features = pd.concat([all_features, volatility_features_df], axis=1)
    print(f"        Added {len(volatility_features_df.columns)} volatility features")
    
    # 4. Technical Indicator Features
    print("  [4/6] Creating technical indicator features (this may take a moment)...")
    technical_features_df = create_technical_features(spy)
    all_features = pd.concat([all_features, technical_features_df], axis=1)
    print(f"        Added {len(technical_features_df.columns)} technical features")
    
    # 5. Binary Regime Features
    print("  [5/6] Creating binary regime features...")
    regime_features_df = create_regime_features(spy)
    all_features = pd.concat([all_features, regime_features_df], axis=1)
    print(f"        Added {len(regime_features_df.columns)} regime features")
    
    # 6. Regime-Dependent Features (optional, computationally expensive)
    if include_regime_dependent:
        print("  [6/6] Creating regime-dependent features...")
        
        # Extract pre-calculated features to avoid redundant computation
        rsi_14 = technical_features_df['spy_rsi_14'] if 'spy_rsi_14' in technical_features_df.columns else None
        bb_position = technical_features_df['spy_bb_position'] if 'spy_bb_position' in technical_features_df.columns else None
        macd_hist_pct = technical_features_df['spy_macd_hist_pct'] if 'spy_macd_hist_pct' in technical_features_df.columns else None
        hurst_50d = technical_features_df['spy_hurst_50d'] if 'spy_hurst_50d' in technical_features_df.columns else None
        
        regime_dependent_features_df = create_regime_dependent_features(
            spy=spy,
            rsi_14=rsi_14,
            bb_position=bb_position,
            macd_hist_pct=macd_hist_pct,
            hurst_50d=hurst_50d
        )
        all_features = pd.concat([all_features, regime_dependent_features_df], axis=1)
        print(f"        Added {len(regime_dependent_features_df.columns)} regime-dependent features")
    else:
        print("  [6/6] Skipping regime-dependent features (disabled)")
    
    print(f"\nFeature engineering complete!")
    print(f"Total features created: {len(all_features.columns)}")
    print(f"Date range: {all_features.index[0]} to {all_features.index[-1]}")
    
    return all_features

def load_data_for_features(data_dir='data'):
    """
    Loads all downloaded CSVs and prepares them for the feature engineering module.
    """
    assets = ['spy', 'vix', 'tlt', 'dxy', 'gld']
    dfs = {}
    for asset in assets:
        path = os.path.join(data_dir, f"{asset}.csv")
        if os.path.exists(path):
            # Read CSV and parse the Date column
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Normalize to date-only format (remove any time components)
            # This handles both timezone-aware and timezone-naive datetimes
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
                
            dfs[asset] = df
            print(f"Loaded {asset.upper()}: {len(df)} rows")
        else:
            print(f"Warning: {path} not found.")
            dfs[asset] = None
    return dfs

def create_target_variable(spy: pd.DataFrame, forward_days: int = 3) -> pd.Series:
    """
    Create target variable for classification.
    
    Parameters:
    -----------
    spy : pd.DataFrame
        SPY OHLC data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    forward_days : int, default=3
        Number of days forward to predict
    
    Returns:
    --------
    pd.Series
        Binary target: 1 if price goes up, 0 if down
    """
    # Calculate forward log return
    target_return = np.log(spy['Close'].shift(-forward_days) / spy['Close'])
    
    # Binary classification: 1 if up, 0 if down
    y = (target_return > 0).astype(int)
    
    return y


def remove_correlated_features(
    X: pd.DataFrame, 
    threshold: float = 0.95
) -> pd.DataFrame:
    """
    Remove highly correlated features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=0.95
        Correlation threshold above which to drop features
    
    Returns:
    --------
    pd.DataFrame
        Feature matrix with highly correlated features removed
    """
    print(f"\nRemoving features with correlation > {threshold}...")
    print(f"Starting features: {len(X.columns)}")
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    X_filtered = X.drop(columns=to_drop)
    
    print(f"Dropped {len(to_drop)} correlated features")
    print(f"Remaining features: {len(X_filtered.columns)}")
    
    return X_filtered


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    """
    Example usage of the master feature engineering module.
    """
    # Example: Create sample data
    #dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # # Sample SPY data
    # spy_data = pd.DataFrame({
    #     'Open': np.random.randn(300).cumsum() + 300,
    #     'High': np.random.randn(300).cumsum() + 305,
    #     'Low': np.random.randn(300).cumsum() + 295,
    #     'Close': np.random.randn(300).cumsum() + 300,
    #     'Volume': np.random.randint(50000000, 100000000, 300)
    # }, index=dates)
    
    # # Sample VIX data
    # vix_data = pd.DataFrame({
    #     'Open': np.abs(np.random.randn(300)) + 15,
    #     'High': np.abs(np.random.randn(300)) + 16,
    #     'Low': np.abs(np.random.randn(300)) + 14,
    #     'Close': np.abs(np.random.randn(300)) + 15,
    #     'Volume': np.random.randint(1000000, 5000000, 300)
    # }, index=dates)

    data = load_data_for_features()
    
    # # Create all features
    # features = create_all_features(
    #     spy=spy_data,
    #     vix=vix_data,
    #     include_regime_dependent=True
    # )

        # 2. Generate Features
    # Note: vix, tlt, dxy, gld are optional in create_all_features
    features = create_all_features(
        spy=data['spy'],
        vix=data['vix'],
        tlt=data['tlt'],
        dxy=data['dxy'],
        gld=data['gld'], 
        include_regime_dependent=True
    )
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total features: {len(features.columns)}")
    print(f"Data shape: {features.shape}")
    print(f"\nFirst 10 feature names:")
    for i, col in enumerate(features.columns[:10], 1):
        print(f"  {i}. {col}")
    print("  ...")
    
    # Create target variable
    y = create_target_variable(data['spy'], forward_days=3)
    print(f"\nTarget variable created: {y.name if hasattr(y, 'name') else 'y'}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    
        # 4. Combine features and target into one dataframe
    # We rename the target column to 'target' for clarity
    final_df = pd.concat([features, y.rename('target')], axis=1)
    
    # 5. Save to output folder
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    output_path = os.path.join(output_dir, 'spy_features_full.csv')
    
    # We use index=True because the Date index is important for time-series splits later
    final_df.to_csv(output_path)

    # Remove highly correlated features (threshold=0.90)
    # 
    # We use 0.90 to remove only near-duplicate features while preserving diverse signals:
    # - Features with correlation > 0.90 are essentially redundant (e.g., momentum_5d vs roc_5d)
    # - Features with correlation 0.70-0.90 may be correlated but capture different market regimes
    # - More aggressive thresholds (0.80, 0.70) risk removing valuable complementary information
    # 
    # This conservative approach lets XGBoost's feature importance identify truly predictive features.
    # After initial training, use permutation importance to select the final top 30-40 features.
    features_filtered = remove_correlated_features(features, threshold=0.90)
    
    print("\n" + "="*60)
    print("READY FOR MODEL TRAINING")
    print("="*60)
    print(f"Features shape: {features_filtered.shape}")
    print(f"Target shape: {y.shape}")
    print("\nNext steps:")
    print("  1. Handle NaN values (dropna or forward fill)")
    print("  2. Split into train/test sets")
    print("  3. Train XGBoost model")
    print("  4. Use permutation importance to select top 30-40 features")
    print("  5. Retrain final model with selected features")
