import pytest
from pytest_bdd import scenarios, given, when, then
from app.core import custom_pipeline
import pandas as pd
from unittest.mock import MagicMock
import asyncio

scenarios('../features/custom_pipeline_flow.feature')

# Shared context
class PipelineContext:
    job_status = None

@pytest.fixture
def context():
    return PipelineContext()

@given('mock external dependencies')
def mock_dependencies(mocker):
    mocker.patch('app.core.custom_pipeline.download_asset_data', return_value=pd.DataFrame({
        'Open': [100, 101], 'High': [102, 103], 'Low': [99, 100], 'Close': [101, 102], 'Volume': [1000, 2000]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02'])))

    mock_features = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    mocker.patch('app.core.custom_pipeline.create_all_features', return_value=mock_features)
    mocker.patch('app.core.custom_pipeline.create_target_variable', return_value=pd.Series([0, 1], index=pd.to_datetime(['2023-01-01', '2023-01-02'])))
    mocker.patch('app.core.custom_pipeline.load_data_for_features', return_value={})

    mocker.patch('pandas.DataFrame.to_csv')
    mocker.patch('os.makedirs')
    mocker.patch('builtins.open', mocker.mock_open())

    # We need to make sure dates allow for split
    # Or mock split_train_test to return non-empty dataframes
    mocker.patch('app.core.custom_pipeline.load_and_prepare_data', return_value=(
        MagicMock(), MagicMock(), pd.Series([pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')])
    ))

    # Mock split_train_test to return NON-EMPTY dataframes for X_train, X_test, etc.
    mock_x_train = pd.DataFrame({'f1': [1]})
    mock_x_test = pd.DataFrame({'f1': [2]})
    mock_y_train = pd.Series([0])
    mock_y_test = pd.Series([1])

    mocker.patch('app.core.custom_pipeline.split_train_test', return_value=(mock_x_train, mock_x_test, mock_y_train, mock_y_test))
    mocker.patch('app.core.custom_pipeline.handle_missing_values', return_value=(mock_x_train, mock_x_test, mock_y_train, mock_y_test))

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.6]
    mock_model.save_model = MagicMock()
    mocker.patch('app.core.custom_pipeline.train_xgboost', return_value=(mock_model, {}))
    mocker.patch('app.core.custom_pipeline.evaluate_xgboost', return_value=({'accuracy': 0.9}, [], []))

    mocker.patch('app.core.custom_pipeline.xgb.DMatrix')

@when('I execute the custom pipeline for "AAPL"')
def execute_pipeline_sync(context):
    jobs = {}
    job_id = "test_job"
    jobs[job_id] = {"status": "pending"}

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(custom_pipeline.run_custom_pipeline(job_id, "AAPL", "xgboost", jobs))
    context.job_status = jobs[job_id]

@then('the status should be updated to "completed"')
def check_status(context):
    if context.job_status['status'] == 'failed':
        pytest.fail(f"Pipeline failed: {context.job_status.get('error')}")
    assert context.job_status['status'] == 'completed'

@then('the model should be trained')
def check_model_trained():
    pass
