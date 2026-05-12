"""
sentiment_analyzer.py
Sentiment scoring using NLTK VADER.
Justification: VADER is specifically designed for social media and financial
short text. It handles punctuation, capitalization, and degree modifiers
better than TextBlob for headline-length text.
"""

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)


def compute_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER compound sentiment score (-1 to +1) to each headline.
    """
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['headline'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    print(f"[✓] Scored {len(df)} headlines")
    print(df[['headline', 'sentiment_score']].head(5).to_string(index=False))
    return df


def label_sentiment_category(score: float) -> str:
    """
    Classify VADER compound score into Positive, Neutral, or Negative.
    Thresholds (standard): >=0.05 Positive, <=-0.05 Negative, else Neutral.
    """
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'