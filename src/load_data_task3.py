import os
import pandas as pd
import glob

def load_news_data(filepath: str) -> pd.DataFrame:
    """
    Load the FNSPID news CSV and return a clean DataFrame.

    Parameters
    ----------
    filepath : str
        Path to your news CSV file.

    Returns
    -------
    pd.DataFrame with columns: headline, date, stock
    """
    df = pd.read_csv(filepath)

    # Keep only the columns we need
    df = df[['headline', 'date', 'stock']].dropna()

    return df

def normalize_news_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and normalize the 'date' column in the news DataFrame.

    Strips time and timezone info so dates can be matched
    against stock trading day dates cleanly.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'date' column.

    Returns
    -------
    df with 'date' as a tz-naive, time-free datetime column.
    """
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.normalize().dt.tz_localize(None)

    print(f"[✓] News dates normalized. Range: {df['date'].min().date()} → {df['date'].max().date()}")

    return df


def load_stock_returns(stock_data_dir: str) -> pd.DataFrame:
    """
    Load all stock CSV files and compute daily percentage returns.

    Daily return = (Close_today - Close_yesterday) / Close_yesterday * 100

    Parameters
    ----------
    stock_data_dir : str
        Path to the folder containing one CSV per stock ticker
        (e.g. 'data/raw/').

    Returns
    -------
    pd.DataFrame with columns: date, stock, daily_return
    """
    stock_files = glob.glob(os.path.join(stock_data_dir, "*.csv"))

    if not stock_files:
        raise FileNotFoundError(f"No CSV files found in: {stock_data_dir}")

    all_stocks = []

    for file in stock_files:
        ticker = os.path.basename(file).replace('.csv', '').upper()

        stock_df = pd.read_csv(file, parse_dates=['Date'])
        stock_df = stock_df[['Date', 'Close']].dropna()

        # Compute daily % return
        stock_df['daily_return'] = stock_df['Close'].pct_change() * 100
        stock_df['stock'] = ticker

        stock_df = stock_df.rename(columns={'Date': 'date'})
        stock_df['date'] = pd.to_datetime(stock_df['date'])

        all_stocks.append(stock_df[['date', 'stock', 'daily_return']])

    returns_df = pd.concat(all_stocks, ignore_index=True).dropna()

    print(f"[✓] Loaded stock returns: {returns_df['stock'].nunique()} stocks, {len(returns_df)} trading days")

    return returns_df
