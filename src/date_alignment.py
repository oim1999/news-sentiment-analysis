"""
date_alignment.py
Aligns news dates to the next available trading day.
Handles weekends and holidays by shifting forward.
"""

import pandas as pd


def get_trading_calendar(stock_df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Extract unique trading dates from stock data.
    """
    dates = pd.to_datetime(stock_df['date']).drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def align_news_to_trading_day(news_df: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Shift each news date to the next available trading day.
    Articles on weekends/holidays move to the next trading day.
    """
    df = news_df.copy()
    trading_days = trading_calendar.normalize().unique()

    def next_trading_day(d):
        d = pd.Timestamp(d).normalize()
        if d in trading_days:
            return d
        future = trading_days[trading_days > d]
        if len(future) > 0:
            return future[0]
        return d

    df['date'] = df['date'].apply(next_trading_day)
    print(f"[✓] Aligned news dates to trading calendar")
    return df