import os, sys
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from xgboost import XGBRegressor
from train import grid_search_train
from column_functions import TfidfWrapper, ravel_values, DateTimeConverter




flatten = FunctionTransformer(ravel_values, validate=False)
vectorizer = make_pipeline(flatten, TfidfVectorizer(preprocessor=None, lowercase=False))
scale = Pipeline([
    ("scaler", StandardScaler())
]) 

cols = ['Impressions', 'Likes', 'Profile Visits', 'CaptionLength',
       'DayOfWeek', 'IsWeekend', 'Month',
       'HourOfDay', 'NumOfHashtags']

scale_col = ['Impressions', 'Likes', 'Profile Visits', 
             'CaptionLength', 'DayOfWeek', 'Month',
            'HourOfDay', 'NumOfHashtags']

datetime = Pipeline([
    ("datetime", DateTimeConverter)
])

preprocessing = ColumnTransformer([
    ("vectorizer", vectorizer, ["Caption", "Hashtags"]),
    ("scaler", scale, scale_col),
    ("datetime", datetime, "Timestamp")
])


full_model_pipeline = Pipeline([
    ("preprocessing", preprocessing)
])