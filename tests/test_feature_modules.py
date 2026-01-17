import pytest
import pandas as pd
import numpy as np
import talib
from app.core.technical_features import create_technical_features
from app.core.volume_features import create_volume_features
from app.core.volatility_features import create_volatility_features

@pytest.fixture
def sample_data():
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    # Create a trend to verify indicators
    close = np.linspace(100, 200, 100) + np.random.normal(0, 2, 100)
    data = pd.DataFrame({
        'Open': close + np.random.normal(0, 1, 100),
        'High': close + 2,
        'Low': close - 2,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

def test_technical_features(sample_data):
    features = create_technical_features(sample_data)

    # RSI
    assert 'spy_rsi_14' in features.columns
    # RSI should be between 0 and 100
    valid_rsi = features['spy_rsi_14'].dropna()
    assert ((valid_rsi >= 0) & (valid_rsi <= 100)).all()

    # MACD
    # The actual feature names are normalized versions or binary indicators
    # We should check for the features actually produced by create_technical_features
    assert 'spy_macd_pct' in features.columns
    assert 'spy_macd_hist_pct' in features.columns
    assert 'spy_macd_bullish_cross' in features.columns

    # Bollinger Bands
    # The function calculates upper/lower bands but only returns features, not raw bands
    assert 'spy_bb_position' in features.columns
    assert 'spy_band_width_pct' in features.columns
    assert 'spy_above_upper_bb' in features.columns

def test_volume_features(sample_data):
    features = create_volume_features(sample_data)

    # Volume Ratios
    # Actual feature names: 10d, 20d, 50d (not 5d)
    assert 'spy_volume_ratio_10d' in features.columns
    assert 'spy_volume_ratio_20d' in features.columns

    # Volume Trend
    assert 'spy_volume_increasing' in features.columns

    # Price-Volume
    assert 'spy_price_vol_confirm' in features.columns

def test_volatility_features(sample_data):
    features = create_volatility_features(sample_data)

    # ATR
    # Feature name is spy_atr_pct
    assert 'spy_atr_pct' in features.columns
    # ATR should be positive
    valid_atr = features['spy_atr_pct'].dropna()
    assert (valid_atr > 0).all()

    # Volatility Regimes
    assert 'spy_high_volatility' in features.columns
    unique_vals = features['spy_high_volatility'].dropna().unique()
    # Should be binary (0 or 1)
    assert set(unique_vals).issubset({0, 1})
