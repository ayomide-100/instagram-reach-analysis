import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
def grid_search_train(X, y):
    model = XGBRegressor(objective = 'reg:squarederror',
                          random_state = 42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate':[0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3,
                                scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_

