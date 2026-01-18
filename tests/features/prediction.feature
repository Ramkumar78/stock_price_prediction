Feature: Model Prediction
  As a trader
  I want to get price predictions for SPY
  So that I can decide whether to buy or sell

  Scenario: Get prediction successfully
    Given the model is trained and available
    When I request a prediction for "SPY"
    Then I should receive a JSON response
    And the response should contain a "prediction" field
    And the response should contain a "probability" field
    And the probability should be between 0 and 1

  Scenario: Handle prediction error when model missing
    Given the model is not available
    When I request a prediction for "SPY"
    Then I should receive an error response
    And the status code should be 404 or 500
