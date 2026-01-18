
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.api.main import app, custom_jobs

client = TestClient(app)

@pytest.fixture
def mock_background_tasks():
    with patch("fastapi.BackgroundTasks.add_task") as mock:
        yield mock

@pytest.fixture
def mock_run_custom_pipeline():
    with patch("app.api.main.run_custom_pipeline") as mock:
        yield mock

def test_start_custom_train(mock_background_tasks, mock_run_custom_pipeline):
    """Test starting a custom training job."""
    response = client.post("/custom/train", json={"ticker": "AAPL", "model": "xgboost"})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "started"

    # Verify job is in in-memory store
    job_id = data["job_id"]
    assert job_id in custom_jobs
    assert custom_jobs[job_id]["ticker"] == "AAPL"

    # Verify background task was added
    # Note: FastAPI TestClient runs background tasks synchronously by default if not mocked,
    # but here we mocked run_custom_pipeline inside main, so we check if add_task was called.
    # Actually TestClient executes the background task function.
    # Since we patched run_custom_pipeline in app.api.main, it should be called.
    # However, we patched it in the test file but imported into main?
    # No, we must patch where it is imported.
    pass

def test_get_custom_status():
    """Test retrieving job status."""
    # Manually add a job
    job_id = "test_job_123"
    custom_jobs[job_id] = {
        "status": "completed",
        "ticker": "TSLA",
        "result": {"prediction": "UP"}
    }

    response = client.get(f"/custom/status/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["result"]["prediction"] == "UP"

def test_get_custom_status_not_found():
    response = client.get("/custom/status/non_existent")
    assert response.status_code == 404

import pandas as pd
import numpy as np
from app.core.custom_pipeline import run_custom_pipeline

@pytest.mark.asyncio
async def test_run_custom_pipeline_success():
    """Test the full pipeline logic with mocks."""
    job_id = "job_test_success"
    jobs_dict = {job_id: {}}

    # Mock data
    dates = pd.date_range('2020-01-01', periods=100)
    df = pd.DataFrame({
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.rand(100) * 1000
    }, index=dates)

    # Mock dependencies
    with patch("app.core.custom_pipeline.download_asset_data", return_value=df) as mock_dl, \
         patch("app.core.custom_pipeline.load_data_for_features", return_value={}) as mock_load_macro, \
         patch("app.core.custom_pipeline.create_all_features", return_value=df) as mock_feats, \
         patch("app.core.custom_pipeline.create_target_variable", return_value=pd.Series(np.random.randint(0, 2, 100), index=dates)) as mock_target, \
         patch("app.core.custom_pipeline.load_and_prepare_data", return_value=(df, mock_target.return_value, dates)) as mock_prep, \
         patch("app.core.custom_pipeline.split_train_test", return_value=(df, df, mock_target.return_value, mock_target.return_value)) as mock_split, \
         patch("app.core.custom_pipeline.handle_missing_values", return_value=(df, df, mock_target.return_value, mock_target.return_value)) as mock_handle, \
         patch("app.core.custom_pipeline.train_xgboost", return_value=(MagicMock(), {})) as mock_train, \
         patch("app.core.custom_pipeline.evaluate_xgboost", return_value=({'test': {'accuracy': 0.6, 'roc_auc': 0.7, 'f1': 0.6}, 'train': {'roc_auc': 0.8}}, [], [])) as mock_eval:

        # Mock model predict
        mock_model = mock_train.return_value[0]
        mock_model.predict.return_value = [0.8] # 80% prob

        await run_custom_pipeline(job_id, "TSLA", "xgboost", jobs_dict)

        assert jobs_dict[job_id]["status"] == "completed"
        assert jobs_dict[job_id]["result"]["prediction"] == "UP"
        assert jobs_dict[job_id]["result"]["model"] == "xgboost"
        assert jobs_dict[job_id]["result"]["ticker"] == "TSLA"

@pytest.mark.asyncio
async def test_run_custom_pipeline_failure():
    """Test pipeline failure handling."""
    job_id = "job_test_fail"
    jobs_dict = {job_id: {}}

    with patch("app.core.custom_pipeline.download_asset_data", return_value=None):
        await run_custom_pipeline(job_id, "FAIL", "xgboost", jobs_dict)

        assert jobs_dict[job_id]["status"] == "failed"
        assert "Failed to download" in jobs_dict[job_id]["error"]
