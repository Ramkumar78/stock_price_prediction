Feature: Feature Engineering
  As a developer
  I want to generate technical indicators
  So that the model has features to learn from

  Scenario: Generate RSI
    Given a dataframe with closing prices
    When I calculate "RSI"
    Then the dataframe should have an "RSI" column
    And the values should be between 0 and 100

  Scenario: Generate Volatility
    Given a dataframe with prices
    When I calculate "volatility_60d"
    Then the dataframe should have a "volatility_60d" column
