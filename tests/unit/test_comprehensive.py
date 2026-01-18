import pytest
from fastapi.testclient import TestClient
from app.api.main import app
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import joblib
import os
import json

# --- Fixtures ---
@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_joblib_load(mocker):
    return mocker.patch('joblib.load')

@pytest.fixture
def mock_yfinance_download(mocker):
    return mocker.patch('app.core.download_data.yf.Ticker')

@pytest.fixture
def mock_feature_engineering_create(mocker):
    return mocker.patch('app.api.main.create_all_features')

@pytest.fixture
def mock_load_data_for_features(mocker):
    return mocker.patch('app.api.main.load_data_for_features')

# --- API Tests ---

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 404

def test_get_metrics_success(client, mocker):
    mocker.patch('os.path.exists', return_value=True)
    mock_metrics_data = {"accuracy": 0.85, "train": {}, "test": {}}
    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(mock_metrics_data)))

    response = client.get("/metrics?model=xgboost")
    assert response.status_code == 200
    assert response.json() == {"model": "Xgboost", "metrics": mock_metrics_data}

def test_get_metrics_not_found_query(client, mocker):
    mocker.patch('os.path.exists', return_value=False)
    response = client.get("/metrics?model=unknown")
    assert response.status_code == 422

def test_get_metrics_not_found_file(client, mocker):
    mocker.patch('os.path.exists', return_value=False)
    response = client.get("/metrics?model=xgboost")
    assert response.status_code == 404

def test_predict_endpoint_success_xgboost(client, mocker):
    mocker.patch('os.path.exists', return_value=True)

    mock_df = pd.DataFrame({
        'Date': ['2023-01-01'],
        'Close': [100.0],
        'feature1': [1.0],
        'feature2': [2.0]
    })
    mocker.patch('pandas.read_csv', return_value=mock_df)

    mock_file = mocker.mock_open(read_data="feature1\nfeature2")
    mocker.patch('builtins.open', mock_file)

    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.4, 0.6]])

    mocker.patch('xgboost.XGBClassifier', return_value=mock_xgb)

    response = client.get("/predict?model=xgboost")

    assert response.status_code == 200
    data = response.json()
    assert data['prediction'] == 'UP'
    assert data['probability'] == 0.6
    assert data['model_used'] == 'Xgboost'

def test_predict_endpoint_model_load_error(client, mocker):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({'Date':['2021-01-01']}))
    mocker.patch('builtins.open', mocker.mock_open(read_data="f1"))

    mock_xgb = MagicMock()
    mock_xgb.load_model.side_effect = Exception("Model corrupt")
    mocker.patch('xgboost.XGBClassifier', return_value=mock_xgb)

    response = client.get("/predict?model=xgboost")

    assert response.status_code == 500

def test_train_endpoint(client, mocker):
    mock_run = mocker.patch('subprocess.run')
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Training complete"

    response = client.post("/train/xgboost")
    assert response.status_code == 200
    assert "trained successfully" in response.json()['message']
    mock_run.assert_called_once()

# --- Custom Pipeline Tests ---
from app.core import custom_pipeline

def test_run_custom_pipeline_async(mocker):
    mock_dl = mocker.patch('app.core.download_data.download_asset_data', return_value=pd.DataFrame())
    mock_save = mocker.patch('app.core.download_data.save_data')
    assert hasattr(custom_pipeline, 'run_custom_pipeline')

# --- Feature Engineering Tests ---
from app.core import feature_engineering

def test_create_all_features(mocker):
    mocker.patch('app.core.feature_engineering.create_price_based_features', return_value=pd.DataFrame({'price_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_volume_features', return_value=pd.DataFrame({'vol_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_volatility_features', return_value=pd.DataFrame({'vola_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_technical_features', return_value=pd.DataFrame({'tech_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_regime_features', return_value=pd.DataFrame({'reg_f': [1]}))
    mocker.patch('app.core.feature_engineering.create_regime_dependent_features', return_value=pd.DataFrame({'reg_dep_f': [1]}))

    spy_df = pd.DataFrame({'Close': [100]}, index=[0])

    df = feature_engineering.create_all_features(spy_df)

    assert 'price_f' in df.columns
    assert 'vol_f' in df.columns

def test_create_target_variable():
    df = pd.DataFrame({'Close': [100, 102, 101, 105]})
    y = feature_engineering.create_target_variable(df, forward_days=3)
    assert y.iloc[0] == 1

# --- Download Data Tests ---
from app.core import download_data

def test_download_asset_data_success(mocker):
    mock_ticker = mocker.Mock()
    mock_df = pd.DataFrame({'Open': [1], 'High': [2], 'Low': [1], 'Close': [2], 'Volume': [100]})
    mock_df.index = pd.to_datetime(['2020-01-01'])
    mock_ticker.history.return_value = mock_df

    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    df = download_data.download_asset_data('SPY', 'SPY', '2020-01-01')
    assert df is not None
    assert len(df) == 1
