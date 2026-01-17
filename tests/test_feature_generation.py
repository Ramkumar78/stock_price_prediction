import pytest
import pandas as pd
import numpy as np
import os
from app.core.feature_engineering import create_all_features

def test_feature_generation():
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    spy_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 300,
        'High': np.random.randn(100).cumsum() + 305,
        'Low': np.random.randn(100).cumsum() + 295,
        'Close': np.random.randn(100).cumsum() + 300,
        'Volume': np.random.randint(50000000, 100000000, 100)
    }, index=dates)

    # We need other assets too for create_all_features to work without errors if it expects them,
    # though they are optional in signature, the code might assume they exist if not passed carefully.
    # Let's check signature: create_all_features(spy, vix=None, ...)
    # The code handles None, so we can just pass spy.

    # However, some internal functions might need VIX.
    # Let's create dummy VIX
    vix_data = pd.DataFrame({
        'Open': np.abs(np.random.randn(100)) + 15,
        'High': np.abs(np.random.randn(100)) + 16,
        'Low': np.abs(np.random.randn(100)) + 14,
        'Close': np.abs(np.random.randn(100)) + 15,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    # Run feature generation
    # include_regime_dependent=False to speed up test and avoid heavy computations like Hurst if not needed
    features = create_all_features(
        spy=spy_data,
        vix=vix_data,
        include_regime_dependent=False
    )

    # Assertions
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert len(features) == 100

    # Check for some expected columns
    expected_cols = ['spy_momentum_3d', 'spy_momentum_5d', 'vix_level']
    for col in expected_cols:
        assert col in features.columns

    # Check that we don't have ALL NaNs (initial rows will be NaN due to rolling windows)
    # But later rows should be populated
    assert not features.iloc[-1].isna().all()
