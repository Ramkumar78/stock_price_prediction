Feature: Data Downloading
  As a data engineer
  I want to download asset data
  So that I can use it for training

  Scenario: Download asset data success
    Given a mocked yfinance ticker
    When I download asset data for "SPY"
    Then the returned data should not be None
    And the returned data should have 1 row
