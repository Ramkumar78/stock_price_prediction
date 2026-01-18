Feature: Custom Pipeline
  As a developer
  I want to run a custom asset pipeline
  So that I can train models on different assets

  Scenario: Run custom pipeline asynchronously
    Given mock download data function
    And mock save data function
    When I verify the existence of run_custom_pipeline
    Then the function should be callable
