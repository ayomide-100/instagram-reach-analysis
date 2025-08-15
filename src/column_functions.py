import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class TfidfWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        X = self._preprocess(X)
        self.vectorizer.fit(X)
        return self  # <--- MUST return self

    def transform(self, X):
        X = self._preprocess(X)
        return self.vectorizer.transform(X)
    
    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            # Flatten all values across columns into strings
            return X.astype(str).agg(' '.join, axis=1).tolist()
        elif isinstance(X, pd.Series):
            return X.astype(str).tolist()
        elif isinstance(X, list):
            return [str(text) if text is not None else "" for text in X]
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()




def ravel_values(x):
    return x.values.ravel()


class DateTimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.column] = pd.to_datetime(X[self.column], errors='coerce')
        return X
