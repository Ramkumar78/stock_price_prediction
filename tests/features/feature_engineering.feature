Feature: Feature Engineering
  As a data scientist
  I want to generate various financial features
  So that I can train accurate models

  Scenario: Create all features
    Given a dataframe with SPY data
    And mock feature functions
    When create_all_features is called
    Then the output dataframe should contain price features
    And the output dataframe should contain volume features

  Scenario: Create target variable
    Given a dataframe with closing prices
    When create_target_variable is called with forward days 3
    Then the target variable should correctly indicate direction
