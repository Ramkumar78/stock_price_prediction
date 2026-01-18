import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import pandas as pd
from app.core import feature_engineering

scenarios('../features/feature_engineering.feature')

@given('a dataframe with SPY data', target_fixture='spy_df')
def spy_df():
    return pd.DataFrame({'Close': [100]}, index=[0])

@given('mock feature functions')
def mock_feature_funcs(mocker):
    mocker.patch('app.core.feature_engineering.create_price_based_features', return_value=pd.DataFrame({'price_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_volume_features', return_value=pd.DataFrame({'vol_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_volatility_features', return_value=pd.DataFrame({'vola_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_technical_features', return_value=pd.DataFrame({'tech_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_regime_features', return_value=pd.DataFrame({'reg_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_regime_dependent_features', return_value=pd.DataFrame({'reg_dep_f': [1]}))

@given('a dataframe with closing prices', target_fixture='close_df')
def close_df():
    return pd.DataFrame({'Close': [100, 102, 101, 105]})

@when('create_all_features is called', target_fixture='features_df')
def call_create_all_features(spy_df, mocker):
    # We need dummy dataframes for other assets as arguments, or mock them if they are used inside
    # The function signature: create_all_features(spy, vix, tlt, dxy, gld, ...)
    dummy = pd.DataFrame()
    return feature_engineering.create_all_features(spy_df, dummy, dummy, dummy, dummy)

@when('create_target_variable is called with forward days 3', target_fixture='target_var')
def call_create_target(close_df):
    return feature_engineering.create_target_variable(close_df, forward_days=3)

@then('the output dataframe should contain price features')
def check_price_features(features_df):
    assert 'price_f' in features_df.columns

@then('the output dataframe should contain volume features')
def check_volume_features(features_df):
    assert 'vol_f' in features_df.columns

@then('the target variable should correctly indicate direction')
def check_target(target_var):
    # 100 -> 105 (up)
    assert target_var.iloc[0] == 1
