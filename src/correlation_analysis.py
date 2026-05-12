"""
correlation_analysis.py
Merging, aggregation, correlation computation, and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def merge_sentiment_and_returns(news_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join news sentiment with stock returns on date + stock ticker.
    """
    merged = pd.merge(news_df, returns_df, on=['date', 'stock'], how='inner')
    print(f"[✓] Merged: {len(merged)} rows across {merged['stock'].nunique()} stocks")
    if len(merged) == 0:
        print("[!] WARNING: Empty merge. Check ticker names match between datasets.")
    return merged


def aggregate_daily_sentiment(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average sentiment scores and returns per stock per day.
    """
    daily = merged_df.groupby(['date', 'stock']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        avg_return=('daily_return', 'mean')
    ).reset_index()
    print(f"[✓] Aggregated to {len(daily)} stock-day pairs")
    return daily


def compute_correlation(daily_df: pd.DataFrame) -> dict:
    """
    Pearson correlation between avg daily sentiment and avg daily return.
    """
    corr, pvalue = pearsonr(daily_df['avg_sentiment'], daily_df['avg_return'])

    if abs(corr) < 0.2:
        strength = "very weak"
    elif abs(corr) < 0.4:
        strength = "weak"
    elif abs(corr) < 0.6:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if corr > 0 else "negative"
    significance = "statistically significant" if pvalue < 0.05 else "not statistically significant"

    print("\n── Correlation Results ──────────────────────")
    print(f"  Pearson r  : {corr:.4f}")
    print(f"  P-value    : {pvalue:.4f}")
    print(f"  Strength   : {strength}")
    print(f"  Direction  : {direction}")
    print(f"  Result is  : {significance} (p < 0.05)")
    print("─────────────────────────────────────────────\n")

    return {
        "correlation": corr,
        "pvalue": pvalue,
        "strength": strength,
        "direction": direction,
        "significant": pvalue < 0.05
    }


def plot_sentiment_vs_return(daily_df: pd.DataFrame, corr_result: dict, save_path: str = None):
    """
    Scatter plot of sentiment vs return with Pearson r annotation.
    """
    corr = corr_result['correlation']
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        daily_df['avg_sentiment'],
        daily_df['avg_return'],
        alpha=0.4,
        color='steelblue',
        edgecolors='none',
        s=30
    )

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    ax.annotate(
        f'Pearson r = {corr:.3f}',
        xy=(0.05, 0.92),
        xycoords='axes fraction',
        fontsize=11,
        color='darkred'
    )

    ax.set_xlabel('Average Daily Sentiment Score (VADER)', fontsize=11)
    ax.set_ylabel('Average Daily Stock Return (%)', fontsize=11)
    ax.set_title('News Sentiment vs. Daily Stock Return', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Saved scatter plot: {save_path}")
    plt.show()


def plot_return_by_sentiment_category(daily_df: pd.DataFrame, save_path: str = None):
    """
    Bar chart of average return by sentiment category (Positive/Neutral/Negative).
    """
    def label_sentiment_category(score: float) -> str:
        """
        Classify VADER compound score into Positive, Neutral, or Negative.
        """
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df = daily_df.copy()
    df['sentiment_label'] = df['avg_sentiment'].apply(label_sentiment_category)
    avg_returns = df.groupby('sentiment_label')['avg_return'].mean()
    avg_returns = avg_returns.reindex(['Positive', 'Neutral', 'Negative'])

    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(avg_returns.index, avg_returns.values, color=colors, edgecolor='black', width=0.5)

    for bar, val in zip(bars, avg_returns.values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}%',
                ha='center',
                va='bottom',
                fontsize=10
            )

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Sentiment Category', fontsize=11)
    ax.set_ylabel('Average Daily Return (%)', fontsize=11)
    ax.set_title('Average Stock Return by Sentiment Category', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Saved bar chart: {save_path}")
    plt.show()

