import os, sys
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from xgboost import XGBRegressor
from column_functions import TfidfWrapper, ravel_values, DateTimeConverter




flatten = FunctionTransformer(ravel_values, validate=False)
vectorizer = make_pipeline(flatten, TfidfVectorizer(preprocessor=None, lowercase=False))
scale = Pipeline([
    ("scaler", StandardScaler())
]) 

text_features = ['Caption', 'Hashtags']
scale_features = ['CaptionLength', 'DayOfWeek', 'Month',
                  'HourOfDay', 'HashtagCount']
passthrough_features = ['HashtagDensity', 'IsWeekend']
timestamp_feature = ['Timestamp']
cat_feature = ['Content Type']


preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'Caption'),  
        ('hashtags', TfidfVectorizer(), 'Hashtags'),  
        ('scale', StandardScaler(), scale_features),
        ('encode', OneHotEncoder(), cat_feature)
    ]
)
