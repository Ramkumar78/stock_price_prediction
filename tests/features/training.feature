Feature: Model Training API
  As a data scientist
  I want to trigger model training via API
  So that I can update the model with latest data

  Scenario: Trigger LightGBM training successfully
    Given the API client is ready
    And the training script execution mocks success
    When I trigger training for "lightgbm"
    Then the response status code should be 200
    And the message should contain "Lightgbm model trained successfully"

  Scenario: Trigger XGBoost training successfully
    Given the API client is ready
    And the training script execution mocks success
    When I trigger training for "xgboost"
    Then the response status code should be 200
    And the message should contain "Xgboost model trained successfully"

  Scenario: Trigger training with invalid model
    Given the API client is ready
    When I trigger training for "invalid_model"
    Then the response status code should be 400
    And the response body should contain detail about invalid model

  Scenario: Trigger training script failure
    Given the API client is ready
    And the training script execution mocks failure
    When I trigger training for "xgboost"
    Then the response status code should be 500
    And the response body should contain failure detail
