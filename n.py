import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor

# ---------------------------
# Define function for modeling
# ---------------------------

def run_model(df, target):
    # Engagement-related columns (leakage sources)
    engagement_cols = [
        'Impressions', 'From Home', 'From Hashtags', 'From Explore',
        'From Other', 'Saves', 'Comments', 'Shares', 
        'Likes', 'Profile Visits', 'Follows',
        'FromHomeRatio', 'FromHashtagsRatio', 
        'FromExploreRatio', 'FromOtherRatio'
    ]
    
    # Drop target & leakage cols
    drop_cols = [col for col in engagement_cols if col != target]
    X = df.drop(columns=drop_cols)
    y = df[target]
    
    # Identify column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['Content Type'] if 'Content Type' in X.columns else []
    text_features = []
    if 'Caption' in X.columns: text_features.append('Caption')
    if 'Hashtags' in X.columns: text_features.append('Hashtags')
    
    # Preprocessing
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50, stop_words='english'))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            # Apply TF-IDF separately to each text column
            *[(f"text_{col}", text_transformer, col) for col in text_features]
        ],
        remainder='drop'
    )
    
    # Model
    model = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Cross-validation RMSE
    scores = -cross_val_score(model, X_train, y_train, 
                              cv=5, scoring='neg_root_mean_squared_error')
    
    print(f"\nTarget: {target}")
    print(f"Mean CV RMSE: {scores.mean():.2f}")
    print(f"Std CV RMSE: {scores.std():.2f}")
    
    # Fit final model
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Run for each target
# ---------------------------

df = pd.read_csv("your_dataset.csv")  # Replace with your actual dataset

targets = ['Impressions', 'Profile Visits', 'Likes']
models = {}

for target in targets:
    models[target] = run_model(df, target)
