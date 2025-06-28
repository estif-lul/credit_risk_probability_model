import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__ (self, time_column):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        # Convert time column to datetime
        X_[self.time_column] = pd.to_datetime(X_[self.time_column])
        X_['TransactionHour'] = X_[self.time_column].dt.hour
        X_['TransactionDay'] = X_[self.time_column].dt.dayofweek
        X_['TransactionMonth'] = X_[self.time_column].dt.month
        X_['TransactionYear'] = X_[self.time_column].dt.year

        return X_[['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']]