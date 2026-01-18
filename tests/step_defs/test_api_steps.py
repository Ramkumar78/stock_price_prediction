import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from fastapi.testclient import TestClient
from app.api.main import app, custom_jobs
import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

scenarios('../features/api.feature')

# Use a context object to store state instead of module global
class ApiContext:
    response = None

@pytest.fixture
def context():
    return ApiContext()

@pytest.fixture
def client():
    # Clear custom jobs for isolation if needed
    custom_jobs.clear()
    return TestClient(app)

@given('the API client is ready')
def api_client(client):
    return client

@given('the metrics file does not exist')
def metrics_file_missing(mocker):
    mocker.patch('os.path.exists', return_value=False)

@given(parsers.parse('the metrics file exists for model "{model_name}"'))
def metrics_file_exists(mocker, model_name):
    mocker.patch('os.path.exists', return_value=True)
    mock_metrics_data = {"accuracy": 0.85, "train": {}, "test": {}}
    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(mock_metrics_data)))

@given(parsers.parse('the model file exists for "{model_name}"'))
def model_file_exists(mocker, model_name):
    # Patch exists to return True
    mocker.patch('os.path.exists', return_value=True)

    # We need to ensure the features file read works and returns features that exist in our mocked data
    # The code reads 'selected_features.txt'
    # And it reads 'output/spy_features_full.csv'

    # We need to distinguish between opening a file for read (features list) or model loading
    # mocker.mock_open is tricky with side_effect if we want different contents for different files
    # simpler approach: just ensure the mocked open returns a file object that yields lines "feature1"
    mocker.patch('builtins.open', mocker.mock_open(read_data="feature1\nfeature2"))


@given('the feature data exists')
def feature_data_exists(mocker):
    mock_df = pd.DataFrame({
        'Date': ['2023-01-01'],
        'Close': [100.0],
        'feature1': [1.0],
        'feature2': [2.0]
    })
    # The code does pd.read_csv(data_path)
    # We must ensure this returns our mock_df
    mocker.patch('pandas.read_csv', return_value=mock_df)

@given('the model is loaded successfully')
def model_loaded(mocker):
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.4, 0.6]])
    mocker.patch('xgboost.XGBClassifier', return_value=mock_xgb)

@given('external data sources are mocked')
def mock_external_data(mocker):
    mock_download = mocker.patch('app.api.main.download_asset_data')
    mock_download.return_value = MagicMock()
    mocker.patch('app.api.main.save_data')

@given('data for features is available')
def mock_data_for_features(mocker):
    mocker.patch('app.api.main.load_data_for_features', return_value={
        'spy': MagicMock(),
        'vix': MagicMock(),
        'tlt': MagicMock(),
        'dxy': MagicMock(),
        'gld': MagicMock()
    })
    mock_features = MagicMock()
    mock_features.columns = ['feat1', 'feat2']
    mocker.patch('app.api.main.create_all_features', return_value=mock_features)

    mock_y = MagicMock()
    mock_y.rename.return_value = mock_y
    mocker.patch('app.api.main.create_target_variable', return_value=mock_y)

    mock_final_df = MagicMock()
    mock_final_df.shape = (100, 3)
    mocker.patch('pandas.concat', return_value=mock_final_df)

    mocker.patch('os.makedirs')
    mocker.patch('pandas.DataFrame.to_csv')

@given('background tasks are mocked')
def mock_bg_tasks(mocker):
    mocker.patch('app.api.main.run_custom_pipeline')

@given(parsers.parse('a custom job exists with ID "{job_id}"'))
def create_custom_job(job_id):
    custom_jobs[job_id] = {"status": "pending", "ticker": "AAPL", "model": "xgboost"}

@given(parsers.parse('no custom job exists with ID "{job_id}"'))
def ensure_no_job(job_id):
    if job_id in custom_jobs:
        del custom_jobs[job_id]

@given('the model loading raises an error')
def model_load_error(mocker):
    mock_xgb = MagicMock()
    mock_xgb.load_model.side_effect = Exception("Model corrupt")
    mocker.patch('xgboost.XGBClassifier', return_value=mock_xgb)
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({'Date':['2021-01-01'], 'Close':[100]}))
    # Must also patch builtins.open so it doesn't fail
    mocker.patch('builtins.open', mocker.mock_open(read_data="f1"))


@given(parsers.parse('the model file does not exist for "{model_name}"'))
def model_file_not_exist(mocker, model_name):
    # Patch exists to return False
    mocker.patch('os.path.exists', return_value=False)

# --- Steps ---

@when('I request the root endpoint "/"')
def request_root(client, context):
    context.response = client.get("/")

@when(parsers.parse('I request metrics for model "{model_name}"'))
def request_metrics(client, context, model_name):
    context.response = client.get(f"/metrics?model={model_name}")

@when(parsers.parse('I request prediction for model "{model_name}"'))
def request_prediction(client, context, model_name):
    context.response = client.get(f"/predict?model={model_name}")

@when('I request to refresh data')
def request_refresh_data(client, context):
    context.response = client.post("/data/refresh")

@when('I request to generate features')
def request_generate_features(client, context):
    context.response = client.post("/features/generate")

@when(parsers.parse('I request custom training for "{ticker}"'))
def request_custom_train(client, context, ticker):
    context.response = client.post("/custom/train", json={"ticker": ticker, "model": "xgboost"})

@when(parsers.parse('I request status for job "{job_id}"'))
def request_job_status(client, context, job_id):
    context.response = client.get(f"/custom/status/{job_id}")

@then(parsers.parse('the response status code should be {status_code:d}'))
def response_status_code(context, status_code):
    assert context.response.status_code == status_code

@then(parsers.parse('the response body should contain the model name "{model_name}"'))
def response_model_name(context, model_name):
    assert context.response.json()["model"] == model_name

@then(parsers.parse('the response should contain prediction "{pred1}" or "{pred2}"'))
def response_prediction_value(context, pred1, pred2):
    assert context.response.json()['prediction'] in [pred1, pred2]

@then('the response should contain probability')
def response_probability(context):
    assert 'probability' in context.response.json()

@then(parsers.parse('the model used should be "{model_name}"'))
def response_model_used(context, model_name):
    assert context.response.json()['model_used'] == model_name

@then(parsers.parse('the message should be "{message}"'))
def response_message(context, message):
    assert context.response.json()["message"] == message

@then('the response should contain a job_id')
def response_job_id(context):
    assert "job_id" in context.response.json()

@then(parsers.parse('the response status should be "{status}"'))
def response_status_value(context, status):
    assert context.response.json()["status"] == status

@then(parsers.parse('the job status should be "{status}"'))
def job_status_value(context, status):
    assert context.response.json()["status"] == status
