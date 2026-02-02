import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker="BTC-USD", period="730d", interval="1h"):
    """
    Fetches historical data for the given ticker from yfinance.
    Uses hourly data ('1h') for the last 730 days (max available for hourly).
    
    Args:
        ticker (str): The ticker symbol.
        period (str): Data period (default '730d').
        interval (str): Data interval (default '1h').
        
    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    print(f"Fetching data for {ticker} (Period: {period}, Interval: {interval})...")
    try:
        # Fetch data
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            print("Warning: No data found.")
            return pd.DataFrame()
        
        # Ensure the index is a DatetimeIndex
        data.index = pd.to_datetime(data.index)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=1, drop_level=True)
            
        # [QUANT STANDARDS] Drop the most recent candle to avoid incomplete real-time data
        if not data.empty:
            data = data.iloc[:-1]
            
        # Ensure chronological order
        data = data.sort_index()
            
        print(f"Successfully fetched {len(data)} rows.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the loader
    df = fetch_data()
    print(df.head())
    print(df.tail())
