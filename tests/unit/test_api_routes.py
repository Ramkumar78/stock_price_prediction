from fastapi.testclient import TestClient
from app.api.main import app
import pytest
import os
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_read_main():
    # This endpoint doesn't exist in my plan, but FastAPI usually has docs
    response = client.get("/docs")
    assert response.status_code == 200

@patch('app.api.main.download_asset_data')
@patch('app.api.main.save_data')
def test_refresh_data(mock_save, mock_download):
    # Mock return value
    mock_download.return_value = MagicMock() # Return a dummy object (not None)

    response = client.post("/data/refresh")
    assert response.status_code == 200
    assert response.json()["message"] == "Data download complete."
    assert "SPY" in response.json()["downloaded"]

@patch('app.api.main.load_data_for_features')
@patch('app.api.main.create_all_features')
@patch('app.api.main.create_target_variable')
@patch('pandas.concat')
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
def test_generate_features(mock_to_csv, mock_makedirs, mock_concat, mock_target, mock_create_features, mock_load):
    # Mock data loading
    mock_load.return_value = {
        'spy': MagicMock(),
        'vix': MagicMock(),
        'tlt': MagicMock(),
        'dxy': MagicMock(),
        'gld': MagicMock()
    }

    # Mock feature creation
    mock_features = MagicMock()
    mock_features.columns = ['feat1', 'feat2']
    mock_create_features.return_value = mock_features

    # Mock target
    mock_y = MagicMock()
    mock_y.rename.return_value = mock_y
    mock_target.return_value = mock_y

    # Mock concat
    mock_final_df = MagicMock()
    mock_final_df.shape = (100, 3)
    mock_concat.return_value = mock_final_df

    response = client.post("/features/generate")
    assert response.status_code == 200
    assert response.json()["message"] == "Features generated successfully."
    assert response.json()["features_count"] == 2

def test_predict_endpoint_no_model():
    # Assuming no model is trained in the test environment initially
    # We patch os.path.exists only for the specific paths we check

    original_exists = os.path.exists

    def side_effect(path):
        if 'lightgbm_model_selected_features.json' in str(path) or 'selected_features.txt' in str(path):
            return False
        return original_exists(path)

    with patch('os.path.exists', side_effect=side_effect):
        response = client.get("/predict")
        assert response.status_code == 404
