import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from fastapi.testclient import TestClient
from app.api.main import app
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

scenarios('../features/prediction.feature')

@pytest.fixture
def client():
    return TestClient(app)

@given('the model is trained and available')
def model_available(mocker):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({'Date': ['2020-01-01'], 'Close': [100], 'f1': [1]}))
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.3, 0.7]])
    mocker.patch('xgboost.XGBClassifier', return_value=mock_xgb)
    mocker.patch('builtins.open', mocker.mock_open(read_data="f1"))

@given('the model is not available')
def model_not_available(mocker):
    mocker.patch('os.path.exists', return_value=False)

@when(parsers.parse('I request a prediction for "{ticker}"'))
def request_prediction(client, ticker):
    pytest.response = client.get(f"/predict?ticker={ticker}&model=xgboost")

@then('I should receive a JSON response')
def check_json_response():
    assert pytest.response.headers['content-type'] == 'application/json'

@then(parsers.parse('the response should contain a "{field}" field'))
def check_field(field):
    json_data = pytest.response.json()
    assert field in json_data

@then('the probability should be between 0 and 1')
def check_probability():
    json_data = pytest.response.json()
    prob = json_data.get('probability')
    assert 0 <= prob <= 1

@then('I should receive an error response')
def check_error_response():
    # 404 is an error response
    assert pytest.response.status_code >= 400

@then(parsers.parse('the status code should be {code1:d} or {code2:d}'))
def check_status_code(code1, code2):
    assert pytest.response.status_code in [code1, code2]
