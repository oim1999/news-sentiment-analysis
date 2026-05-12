"""
data_loader.py
Handles loading and cleaning of news and stock datasets.
"""

import os
import glob
import pandas as pd


def load_news_data(filepath: str) -> pd.DataFrame:
    """
    Load the news CSV and return a clean DataFrame.
    Keeps only: headline, date, stock
    """
    df = pd.read_csv(filepath)
    df = df[['headline', 'date', 'stock', 'publisher']].dropna()
    print(f"[✓] Loaded {len(df)} news headlines")
    return df


def parse_news_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse news 'date' column to datetime, strip time/timezone.
    """
    # errors='coerce' turns bad formats into NaT instead of crashing
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # Drop rows that failed to parse (should be very few)
    before = len(df)
    df = df.dropna(subset=['date'])
    after = len(df)
    if before != after:
        print(f"[!] Dropped {before - after} rows with unparseable dates")
    
    # Strip time and timezone
    df['date'] = df['date'].dt.normalize().dt.tz_localize(None)
    
    print(f"[✓] News dates parsed. Range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def load_stock_data(stock_data_dir: str) -> pd.DataFrame:
    """
    Load all stock CSV files into one DataFrame with columns:
    date, stock, Close
    """
    stock_files = glob.glob(os.path.join(stock_data_dir, "*.csv"))
    if not stock_files:
        raise FileNotFoundError(f"No CSV files found in: {stock_data_dir}")

    all_stocks = []
    for file in stock_files:
        ticker = os.path.basename(file).replace('.csv', '').upper()
        stock_df = pd.read_csv(file, parse_dates=['Date'])
        stock_df = stock_df[['Date', 'Close']].dropna()
        stock_df = stock_df.rename(columns={'Date': 'date'})
        stock_df['stock'] = ticker
        all_stocks.append(stock_df)

    combined = pd.concat(all_stocks, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    print(f"[✓] Loaded {combined['stock'].nunique()} stocks, {len(combined)} rows")
    return combined