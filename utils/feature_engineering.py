import pandas as pd
import numpy as np

def add_day_of_week(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df['DayOfWeek'] = df[timestamp_col].dt.dayofweek
    return df

def add_hour_of_day(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df['HourOfDay'] = df[timestamp_col].dt.hour
    return df

def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    return df

def add_month(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df['Month'] = df[timestamp_col].dt.month
    return df

def add_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    df['EngagementRate'] = (
        df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
    ) / df['Impressions']
    return df
def len_hashtags(x:str):
    return len(str(x).split())


def add_hashtag_count(df: pd.DataFrame) -> pd.DataFrame:
    df['HashtagCount'] = df['Hashtags'].apply(len_hashtags)
    return df
def add_hashtags_count(df: pd.DataFrame) -> pd.DataFrame:
    df['HashtagCount'] = df['Hashtags'].str.count('#')
    return df

def cap_len(x: str):
    return len(str(x).split())

def add_caption_length(df: pd.DataFrame) -> pd.DataFrame:
    df['CaptionLength'] = df['Caption'].apply(cap_len)
    return df

def add_hashtag_density(df: pd.DataFrame) -> pd.DataFrame:
    df['HashtagDensity'] = df['HashtagCount'] / (df['CaptionLength'] + 1)
    return df

def add_source_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df['FromHomeRatio'] = df['From Home'] / df['Impressions']
    df['FromHashtagsRatio'] = df['From Hashtags'] / df['Impressions']
    df['FromExploreRatio'] = df['From Explore'] / df['Impressions']
    df['FromOtherRatio'] = df['From Other'] / df['Impressions']
    return df
