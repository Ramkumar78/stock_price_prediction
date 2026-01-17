# Download Data Module

This module handles the retrieval of historical OHLCV data from Yahoo Finance for the XGBoost stock prediction model.

## Features

- **Automatic Adjustments**: Uses `auto_adjust=True` in `yfinance` to automatically handle stock splits and dividends for all price columns (Open, High, Low, Close). 
- **Volume Adjustment**: Historical volume is also split-adjusted to ensure consistency with adjusted prices.
- **Robustness**: Includes built-in retry logic (3 attempts) to handle potential network issues or Yahoo Finance API rate limits.
- **Caching**: Saves data to local CSV files in the `data/` directory for fast loading in feature engineering modules.

## Assets Covered

The script downloads daily data starting from **2018-01-01** for the following assets:

| Asset Name | Ticker Symbol | Description |
|------------|---------------|-------------|
| SPY | SPY | S&P 500 ETF (Primary Target) |
| VIX | ^VIX | Volatility Index (Fear Gauge) |
| TLT | TLT | 20+ Year Treasury Bond ETF |
| DXY | DX-Y.NYB | US Dollar Index |
| GLD | GLD | Gold ETF |

## Usage

Ensure you have the dependencies installed in your virtual environment:

```bash
pip install yfinance pandas
```

Run the script from the project root:

```bash
python download_data.py
```

## Output

The script creates a `data/` directory (if it doesn't exist) and saves the following files:
- `data/spy.csv`
- `data/vix.csv`
- `data/tlt.csv`
- `data/dxy.csv`
- `data/gld.csv`

Each file contains the columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

Since VIX and DXY are actual indexes and not ETFs they do no have volume data.
