Feature: Mathematical Logic and Feature Verification
  As a data scientist
  I want to verify the correctness of financial indicators
  So that the models receive accurate features

  Scenario: Verify Technical Indicators Calculation
    Given a sample dataframe with price data
    When I calculate technical features
    Then the output should contain RSI
    And the output should contain MACD
    And the output should contain Bollinger Bands
    And RSI should be between 0 and 100

  Scenario: Verify Volatility Features
    Given a sample dataframe with price data
    When I calculate volatility features
    Then the output should contain rolling volatility
    And the output should contain GARCH volatility if applicable

  Scenario: Verify Volume Features
    Given a sample dataframe with volume data
    When I calculate volume features
    Then the output should contain accumulation
    And the output should contain volume change features

  Scenario: Verify Hurst Exponent Calculation
    Given a random walk time series
    When I calculate the Hurst exponent
    Then the result should be close to 0.5
    And the result should not be NaN

  Scenario: Verify Regime Dependent Features
    Given a sample dataframe with price data
    When I calculate regime dependent features
    Then the output should contain "spy_rsi_in_trending"
    And the output should contain "spy_rsi_in_mr"
    And the output should contain "spy_trend_continuation"
    And the output should contain "spy_mean_reversion_setup"

  Scenario: Verify Data Saving
    Given a dataframe to save
    When I save the data to "test_asset"
    Then the file "data/test_asset.csv" should exist
    And I cleanup the file "data/test_asset.csv"
