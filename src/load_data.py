import pandas as pd


def load_data(filepath):
    """Load a CSV file and return a clean, date-indexed DataFrame."""
    df = pd.read_csv(filepath, index_col=0)  # use the first column as the index
    df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
    df['date'] = df['date'].dt.tz_localize(None)   # strip the -04:00 timezone
    df['date'] = df['date'].dt.normalize() 

    print(f"Loaded {len(df)} rows from '{filepath}'")
    return df


def remove_nulls(df):
    """Remove rows that have any missing values. Print a summary."""
    before = len(df)
    df = df.dropna()
    print(f"Removed {before - len(df)} row(s). {len(df)} rows remaining.")
    return df

