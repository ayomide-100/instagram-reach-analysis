import sys, os
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from pipeline import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath('../data'))
sys.path.append(os.path.abspath('../models'))
df = pd.read_csv("data/processed/clean_data.csv")

xgbr = XGBRegressor(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)

rf  = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    oob_score=True
)








ridge_model = Ridge(
    alpha=10,          # regularization strength
    fit_intercept=True, 
    solver="auto",      # let sklearn choose the best one
    random_state=42
)

gbr = GradientBoostingRegressor(
    n_estimators=50,       # Reduced from 200
    learning_rate=0.05,    # Slightly increased to compensate for fewer estimators
    max_depth=3,           # Reduced from 5
    min_samples_leaf=5,    # Increased from 4
    subsample=0.8,
    random_state=42
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', gbr)
])

X = df.drop(columns=['Impressions', 'Timestamp', 'HashtagCount',
                    'Profile Visits', 'Saves', 'Shares',
                    'From Home', 'From Hashtags', 'From Explore',
                    'From Other', 'Comments', 'Follows', 'Likes', 
                    'FromHomeRatio', 'FromHashtagsRatio', 'EngagementRate',
                    'FromExploreRatio', 'FromOtherRatio', 'NumOfHashtags'])

y = df["Impressions"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train
pipeline.fit(X_train, y_train)
scores = cross_val_score(pipeline, X, y, cv=5, scoring= 'neg_mean_squared_error')

# Evaluate
y_pred = pipeline.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

rmse_scores = np.sqrt(-scores)
print(f"RMSE scores for each fold: {rmse_scores}")
print(f"Average RMSE: {rmse_scores.mean():.4f}")
print(f"Standard Deviation of RMSE: {rmse_scores.std():.4f}")
# Save trained model

joblib.dump(pipeline, "models/ml_models/impressions_model.pkl")
print("Model saved as impressions_model.pkl")
