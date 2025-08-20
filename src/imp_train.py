import sys, os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from pipeline import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../models')))

df = pd.read_csv("data/processed/clean_data.csv")

par = PassiveAggressiveRegressor(random_state=42)


gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
)

ridge = Ridge(alpha=7, random_state=42)


ensemble = VotingRegressor(estimators=[
    ('ridge', ridge),
    ('gbr', gbr)
], weights=[0.01, 0.99])



pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', gbr)
])


X = df.drop(columns=['Impressions', 'Timestamp', 'NumOfHashtags',
                     'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Follows',
                     'FromHomeRatio', 'FromHashtagsRatio', 'EngagementRate', 'FromExploreRatio', 
                     'FromOtherRatio', 'Content Type', 'Caption', 'Hashtags'])

y = np.log1p(df["Impressions"])  # log(1 + x)

bins = pd.qcut(df["Impressions"], q=5, duplicates="drop")
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')


rmse_scores = np.sqrt(-scores)


print(f"RMSE score for fold: {rmse_scores}")
print(f"Average RMSE: {rmse_scores.mean():.4f}")
print(f"Standard Deviation of RMSE: {rmse_scores.std():.4f}")

print(f"R2 score for folds: {r2_scores}")
print("Mean R2 score:", r2_scores.mean())
print("Standard deviation of R2 scores:", r2_scores.std())


