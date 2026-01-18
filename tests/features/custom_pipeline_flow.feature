Feature: Custom Pipeline Flow
  As a developer
  I want to run the full custom pipeline
  So that I can verify the integration of download, features, and training

  Scenario: Run custom pipeline logic
    Given mock external dependencies
    When I execute the custom pipeline for "AAPL"
    Then the status should be updated to "completed"
    And the model should be trained
