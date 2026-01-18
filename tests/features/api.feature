Feature: API Endpoints
  As a user
  I want to access various API endpoints
  So that I can interact with the prediction system

  Scenario: Root endpoint returns 404
    Given the API client is ready
    When I request the root endpoint "/"
    Then the response status code should be 404

  Scenario: Get metrics for unknown model returns 404
    Given the API client is ready
    And the metrics file does not exist
    When I request metrics for model "unknown"
    Then the response status code should be 404

  Scenario: Get metrics for existing model
    Given the API client is ready
    And the metrics file exists for model "xgboost"
    When I request metrics for model "xgboost"
    Then the response status code should be 200
    And the response body should contain the model name "Xgboost"

  Scenario: Predict endpoint success for xgboost
    Given the API client is ready
    And the model file exists for "xgboost"
    And the feature data exists
    And the model is loaded successfully
    When I request prediction for model "xgboost"
    Then the response status code should be 200
    And the response should contain prediction "UP" or "DOWN"
    And the response should contain probability
    And the model used should be "Xgboost"

  Scenario: Refresh data endpoint success
    Given the API client is ready
    And external data sources are mocked
    When I request to refresh data
    Then the response status code should be 200
    And the message should be "Data download complete."

  Scenario: Generate features endpoint success
    Given the API client is ready
    And data for features is available
    When I request to generate features
    Then the response status code should be 200
    And the message should be "Features generated successfully."

  # --- New Scenarios for Missing Coverage ---

  Scenario: Custom Train Endpoint Success
    Given the API client is ready
    And background tasks are mocked
    When I request custom training for "AAPL"
    Then the response status code should be 200
    And the response should contain a job_id
    And the response status should be "started"

  Scenario: Custom Status Endpoint
    Given the API client is ready
    And a custom job exists with ID "job123"
    When I request status for job "job123"
    Then the response status code should be 200
    And the job status should be "pending"

  Scenario: Custom Status Endpoint Not Found
    Given the API client is ready
    And no custom job exists with ID "unknown_job"
    When I request status for job "unknown_job"
    Then the response status code should be 404

  Scenario: Predict Endpoint Model Load Error
    Given the API client is ready
    And the model file exists for "xgboost"
    But the model loading raises an error
    When I request prediction for model "xgboost"
    Then the response status code should be 500

  Scenario: Predict Endpoint No Model Found
    Given the API client is ready
    And the model file does not exist for "lightgbm"
    When I request prediction for model "lightgbm"
    Then the response status code should be 404
