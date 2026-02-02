import pandas as pd

import ta

def add_features(df):
    """
    Adds technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame with 'Close', 'High', 'Low', 'Volume' columns.
        
    Returns:
        pd.DataFrame: The DataFrame with added features.
    """
    if df.empty:
        return df
    
    # Ensure columns are float
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # Copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Log Return
    # Requires numpy (ensure it is imported)
    import numpy as np
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # EMAs
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()
    
    # Volume Features
    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Simple Moving Average of Volume
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Target: Next day's Close (for training purposes, though usually we create x/y sequences)
    # We will handle target creation in the dataset creation step, 
    # but here we can add some lag features if needed.
    
    # [QUANT STANDARDS] Feature Safety Verification
    # All features above are Lag(0) or Lag(N).
    # RSI, MACD, EMA use strictly past data (rolling windows ending at t).
    # No shift(-k) operations are performed here.
    
    # Handle infinite values (e.g. from volume pct_change)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop NaNs created by indicators (Startup period)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # Test features
    from data_loader import fetch_data
    df = fetch_data()
    if not df.empty:
        df_features = add_features(df)
        print(df_features.head())
        print("Columns:", df_features.columns)
