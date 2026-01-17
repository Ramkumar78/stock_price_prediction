import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import time

# --- Configuration ---
TICKERS = {
    'SPY': 'SPY',
    'VIX': '^VIX',
    'TLT': 'TLT',
    'DXY': 'DX-Y.NYB',
    'GLD': 'GLD'
}
START_DATE = '2015-01-01'
DATA_DIR = 'data'

def download_asset_data(ticker_symbol, name, start_date):
    """
    Download daily OHLCV data for a given ticker.
    
    Args:
        ticker_symbol (str): The yfinance ticker symbol.
        name (str): Friendly name for the asset (used for filename).
        start_date (str): Start date in YYYY-MM-DD format.
        
    Returns:
        pd.DataFrame: Downloaded data or None if failed.
    """
    print(f"Downloading {name} ({ticker_symbol}) starting from {start_date}...")
    
    # Retry logic for robustness
    for attempt in range(3):
        try:
            # auto_adjust=True takes care of dividends and splits automatically
            # This adjusts Open, High, Low, Close based on corporate actions
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start_date, interval='1d', auto_adjust=True)
            
            if df.empty:
                print(f"Warning: No data found for {name} ({ticker_symbol}).")
                return None
            
            # Ensure index is datetime and sorted
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Normalize to date-only (remove timezone and time components)
            # This ensures clean date indices without time information
            df.index = df.index.normalize().tz_localize(None)
            
            # Keep only the columns we need for feature engineering
            # Note: volume column name might vary in capitalization depending on yfinance version/history call
            cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Map possible column names (yf.history usually returns capitalized)
            df = df[[col for col in df.columns if col in cols_to_keep]]
            
            print(f"Successfully downloaded {len(df)} rows for {name}.")
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {name}: {str(e)}")
            if attempt < 2:
                time.sleep(2) # Wait before retrying
            else:
                print(f"Error: All attempts failed for {name}.")
                return None

def save_data(df, name):
    """Save the dataframe to a CSV file in the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
        
    filepath = os.path.join(DATA_DIR, f"{name.lower()}.csv")
    df.to_csv(filepath)
    print(f"Saved {name} data to {filepath}")

def main():
    """Main execution function."""
    print("=== Data Download Script ===")
    print(f"Base Directory: {os.getcwd()}")
    
    for name, symbol in TICKERS.items():
        df = download_asset_data(symbol, name, START_DATE)
        if df is not None:
            save_data(df, name)
        print("-" * 30)

    print("Data download process complete.")

if __name__ == "__main__":
    main()
