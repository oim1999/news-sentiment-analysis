import pandas as pd
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def compute_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a VADER compound sentiment score to each headline.

    Score ranges from -1 (very negative) to +1 (very positive).
    Scores near 0 are neutral.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'headline' column.

    Returns
    -------
    df with a new 'sentiment_score' column added.
    """
    sia = SentimentIntensityAnalyzer()

    df['sentiment_score'] = df['headline'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )

    print(f"[✓] Sentiment scored: {len(df)} headlines")
    print(df[['headline', 'sentiment_score']].head(5).to_string(index=False))

    return df
