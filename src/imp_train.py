import sys, os
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from pipeline import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../models')))

df = pd.read_csv("data/processed/clean_data.csv")


df['Hashtags'] = df['Hashtags'].str.replace('#', ' ').fillna('')
df['Caption'] = df['Caption'].str.replace('#', ' ').fillna('') 


gbr = GradientBoostingRegressor(
    n_estimators=300,        # more trees, but shallow
    learning_rate=0.01,      # slower learning (safer with small data)
    max_depth=2,             # very shallow trees reduce variance
    min_samples_split=5,     # don’t split too aggressively
    min_samples_leaf=5,      # avoid tiny leaves
    subsample=0.8,           # stochastic boosting → helps with variance
    max_features="sqrt",     # random subset of features per split
    loss="squared_error",    # standard regression loss
    random_state=42
)

ridge = Ridge(alpha=10, random_state=42)
elasticnet = ElasticNet(
    alpha=0.05,        # overall regularization strength (slightly lower than Ridge)
    l1_ratio=0.2,      # 20% Lasso, 80% Ridge -> stable + some sparsity
    fit_intercept=True,
    max_iter=5000,
    tol=1e-4,
    random_state=42
)

lasso = Lasso(
    alpha=0.0005,   # tiny alpha so it doesn’t kill all text features
    fit_intercept=True,
    max_iter=5000,
    tol=1e-4,
    random_state=42
)


ensemble = VotingRegressor(estimators=[
    ('ridge', ridge),
    ('gbr', gbr)
], weights=[0.05, 0.95])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ensemble)
])


X = df.drop(columns=['Impressions', 'Timestamp', 'NumOfHashtags', 'Profile Visits', 'Saves', 'Shares',
                     'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Comments', 'Follows', 'Likes',
                     'FromHomeRatio', 'FromHashtagsRatio', 'EngagementRate', 'FromExploreRatio', 'FromOtherRatio'])

y = np.log1p(df["Impressions"])  # log(1 + x)

bins = pd.qcut(df["Impressions"], q=5, duplicates="drop")
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')


rmse_scores = np.sqrt(-scores)
print(f"RMSE scores for each fold: {rmse_scores}")
print(f"Average RMSE: {rmse_scores.mean():.4f}")
print(f"Standard Deviation of RMSE: {rmse_scores.std():.4f}")

print("R2 scores for each fold:", r2_scores)
print("Mean R2 score:", r2_scores.mean())
print("Standard deviation of R2 scores:", r2_scores.std())


# pipeline.fit(X, y)
# joblib.dump(pipeline, "models/ml_models/impressions_model.pkl")
# print("Model saved as impressions_model.pkl")