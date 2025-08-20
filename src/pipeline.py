import os, sys
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector

#from column_functions import TfidfWrapper, ravel_values, DateTimeConverter








scale_features = [
    "Likes", "Comments", "Saves",
    "Shares", "Profile Visits",
    "CaptionLength", "HashtagCount", 
    "DayOfWeek", "HourOfDay", "Month"
]

passthrough_features = ["IsWeekend", "HashtagDensity"]

full_features = scale_features + passthrough_features

target = ["Impressions"]




preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), scale_features)
]

)