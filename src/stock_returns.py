"""
stock_returns.py
Computes daily percentage returns from closing prices.
"""

import pandas as pd


def compute_daily_returns(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily % return: (Close_t - Close_{t-1}) / Close_{t-1} * 100
    Returns DataFrame with columns: date, stock, daily_return
    """
    df = stock_df.copy()
    df = df.sort_values(['stock', 'date'])
    df['daily_return'] = df.groupby('stock')['Close'].pct_change() * 100
    df = df[['date', 'stock', 'daily_return']].dropna()
    print(f"[✓] Computed returns: {len(df)} trading day records")
    return df