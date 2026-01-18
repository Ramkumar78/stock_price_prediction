Feature: Model Training
  As a data scientist
  I want to trigger model training
  So that I can update the model with latest data

  Scenario: Trigger LightGBM training
    Given the data is available
    When I trigger training for "lightgbm"
    Then I should receive a success response
    And the training process should start in background

  Scenario: Trigger Custom Asset Training
    Given I have a new asset ticker "AAPL"
    When I trigger custom training for "AAPL"
    Then I should receive a "Training started" message
    And the custom pipeline should be initiated
