import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import pandas as pd
from app.core import download_data

scenarios('../features/data_download.feature')

@given('a mocked yfinance ticker')
def mock_yf_ticker(mocker):
    mock_ticker = mocker.Mock()
    mock_df = pd.DataFrame({'Open': [1], 'High': [2], 'Low': [1], 'Close': [2], 'Volume': [100]})
    mock_df.index = pd.to_datetime(['2020-01-01'])
    mock_ticker.history.return_value = mock_df
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

@when(parsers.parse('I download asset data for "{ticker}"'), target_fixture='downloaded_df')
def download_data_func(ticker):
    return download_data.download_asset_data(ticker, ticker, '2020-01-01')

@then('the returned data should not be None')
def check_data_not_none(downloaded_df):
    assert downloaded_df is not None

@then('the returned data should have 1 row')
def check_data_len(downloaded_df):
    assert len(downloaded_df) == 1
