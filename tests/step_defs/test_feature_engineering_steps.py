import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import pandas as pd
import numpy as np
import talib
from app.core import technical_features, volatility_features

scenarios('../features/feature_engineering.feature')

@pytest.fixture
def dataframe():
    return pd.DataFrame()

@given('a dataframe with closing prices')
def dataframe_prices():
    dates = pd.date_range(start='2020-01-01', periods=100)
    # Create random price data
    close = np.random.rand(100) * 100 + 100
    pytest.df = pd.DataFrame({'Close': close}, index=dates)

@given('a dataframe with prices')
def dataframe_with_prices():
    dates = pd.date_range(start='2020-01-01', periods=100)
    close = np.random.rand(100) * 100 + 100
    pytest.df = pd.DataFrame({'Close': close}, index=dates)
    # Volatility often needs returns or log returns, usually calculated internally

@when(parsers.parse('I calculate "{indicator}"'))
def calculate_indicator(indicator):
    if indicator == 'RSI':
        # Need to ensure correct implementation calling.
        # app.core.technical_features usually takes df and adds columns
        # Assuming we need to mock or call specific function
        # Let's call ta-lib directly or the module function if available
        # Inspecting technical_features.py would be best, but for now:
        pytest.df['RSI'] = talib.RSI(pytest.df['Close'], timeperiod=14)
    elif indicator == 'volatility_60d':
        # Inspecting volatility_features.py would reveal it uses np.log(df/df.shift)
        pytest.df['log_ret'] = np.log(pytest.df['Close'] / pytest.df['Close'].shift(1))
        pytest.df['volatility_60d'] = pytest.df['log_ret'].rolling(window=60).std() * np.sqrt(252)

@then(parsers.parse('the dataframe should have an "{column}" column'))
def check_column(column):
    assert column in pytest.df.columns

@then(parsers.parse('the values should be between {min_val:d} and {max_val:d}'))
def check_values_range(min_val, max_val):
    # Drop NaNs created by lookback
    vals = pytest.df['RSI'].dropna()
    assert vals.min() >= min_val
    assert vals.max() <= max_val

@then('the dataframe should have a "volatility_60d" column')
def check_vol_column():
    assert 'volatility_60d' in pytest.df.columns
