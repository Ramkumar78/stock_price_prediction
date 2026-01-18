import pytest
from pytest_bdd import scenarios, given, when, then
import pandas as pd
from app.core import custom_pipeline

scenarios('../features/custom_pipeline.feature')

@given('mock download data function')
def mock_download(mocker):
    mocker.patch('app.core.download_data.download_asset_data', return_value=pd.DataFrame())

@given('mock save data function')
def mock_save(mocker):
    mocker.patch('app.core.download_data.save_data')

@when('I verify the existence of run_custom_pipeline')
def check_pipeline_existence():
    pass

@then('the function should be callable')
def check_callable():
    assert callable(custom_pipeline.run_custom_pipeline)
