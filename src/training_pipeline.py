import sys, os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pipeline import full_features, preprocessor, target
from imp_train import gbr



sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../models')))

df = pd.read_csv("data/processed/clean_data.csv")


X = df[full_features]
y = np.log1p(df[target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                      random_state=42)


train_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gbr)
])


train_pipeline.fit(X_train, y_train)

y_pred = train_pipeline.predict(X_test)
print("Holdout RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Holdout R2:", r2_score(y_test, y_pred))


joblib.dump(train_pipeline, "models/ml_models/impression_model.pkl")
print("Model saved as impression_model.pkl")