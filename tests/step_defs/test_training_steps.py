import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import MagicMock

scenarios('../features/training.feature')

# Shared context
class TrainingContext:
    response = None

@pytest.fixture
def context():
    return TrainingContext()

@pytest.fixture
def client():
    return TestClient(app)

@given('the API client is ready')
def api_client(client):
    return client

@given('the training script execution mocks success')
def mock_training_success(mocker):
    mock_run = mocker.patch('subprocess.run')
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Training complete"
    # Need to mock sys.executable too if it's used?
    # subprocess.run([sys.executable...])
    # mocking subprocess.run is enough to capture the call regardless of args

@given('the training script execution mocks failure')
def mock_training_failure(mocker):
    mock_run = mocker.patch('subprocess.run')
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "Script crashed"

@when(parsers.parse('I trigger training for "{model_type}"'))
def trigger_training(client, context, model_type):
    context.response = client.post(f"/train/{model_type}")

@then(parsers.parse('the response status code should be {status_code:d}'))
def response_status_code(context, status_code):
    assert context.response.status_code == status_code

@then(parsers.parse('the message should contain "{text}"'))
def response_message_contains(context, text):
    assert text in context.response.json()['message']

@then('the response body should contain detail about invalid model')
def response_invalid_detail(context):
    assert "Invalid model" in context.response.json()['detail']

@then('the response body should contain failure detail')
def response_failure_detail(context):
    assert "Training script failed" in context.response.json()['detail']
