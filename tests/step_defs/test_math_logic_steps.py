import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import pandas as pd
import numpy as np
import os
from app.core import feature_engineering
from app.core.regime_dependent_features import _hurst_exponent, create_regime_dependent_features
from app.core.download_data import save_data

scenarios('../features/math_logic.feature')

@pytest.fixture
def sample_df():
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = {
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    }
    return pd.DataFrame(data, index=dates)

@given('a sample dataframe with price data', target_fixture='input_df')
def input_df(sample_df):
    return sample_df

@given('a sample dataframe with volume data', target_fixture='input_vol_df')
def input_vol_df(sample_df):
    return sample_df

@when('I calculate technical features', target_fixture='tech_features')
def calculate_tech(input_df):
    return feature_engineering.create_technical_features(input_df)

@when('I calculate volatility features', target_fixture='vol_features')
def calculate_vola(input_df):
    return feature_engineering.create_volatility_features(input_df)

@when('I calculate volume features', target_fixture='volume_features')
def calculate_volume(input_vol_df):
    return feature_engineering.create_volume_features(input_vol_df)

@then('the output should contain RSI')
def check_rsi(tech_features):
    assert any('rsi' in col for col in tech_features.columns)

@then('the output should contain MACD')
def check_macd(tech_features):
    assert any('macd' in col for col in tech_features.columns)

@then('the output should contain Bollinger Bands')
def check_bbands(tech_features):
    assert any('bb_' in col or 'band' in col for col in tech_features.columns)

@then('RSI should be between 0 and 100')
def check_rsi_values(tech_features):
    rsi_cols = [c for c in tech_features.columns if 'rsi' in c]
    for col in rsi_cols:
        assert tech_features[col].dropna().between(0, 100).all()

@then('the output should contain rolling volatility')
def check_rolling_vol(vol_features):
    assert any('std' in col or 'volatility' in col or 'atr' in col for col in vol_features.columns)

@then('the output should contain GARCH volatility if applicable')
def check_garch(vol_features):
    pass

@then('the output should contain accumulation')
def check_accumulation(volume_features):
    assert any('accumulation' in col for col in volume_features.columns)

@then('the output should contain volume change features')
def check_vol_change(volume_features):
    assert any('volume_ratio' in col or 'spike' in col for col in volume_features.columns)

# --- New Steps for Additional Coverage ---

@given('a random walk time series', target_fixture='random_walk')
def random_walk():
    np.random.seed(42)
    return np.cumsum(np.random.randn(1000))

@when('I calculate the Hurst exponent', target_fixture='hurst_val')
def calculate_hurst(random_walk):
    return _hurst_exponent(random_walk, max_lag=20)

@then('the result should be close to 0.5')
def check_hurst_val(hurst_val):
    assert 0.3 < hurst_val < 0.7

@then('the result should not be NaN')
def check_not_nan(hurst_val):
    assert not np.isnan(hurst_val)

@when('I calculate regime dependent features', target_fixture='regime_features')
def calculate_regime_features(input_df):
    return create_regime_dependent_features(input_df)

@then(parsers.parse('the output should contain "{col_name}"'))
def check_col_exists(regime_features, col_name):
    assert col_name in regime_features.columns

@given('a dataframe to save', target_fixture='df_to_save')
def df_to_save():
    return pd.DataFrame({"col": [1, 2]})

@when(parsers.parse('I save the data to "{name}"'))
def save_test_data(df_to_save, name):
    save_data(df_to_save, name)

@then(parsers.parse('the file "{filepath}" should exist'))
def check_file_exists(filepath):
    assert os.path.exists(filepath)

@then(parsers.parse('I cleanup the file "{filepath}"'))
def cleanup_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
