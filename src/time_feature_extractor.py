import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    TimeFeatureExtractor extracts temporal features from a specified datetime column in a DataFrame.
    Parameters
    ----------
    time_column : str
        The name of the column containing datetime information from which to extract features.
    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data. This implementation does nothing and is present for compatibility.
    transform(X)
        Transforms the input DataFrame by extracting the hour, day of the week, month, and year
        from the specified datetime column. Returns a DataFrame with these new features:
            - TransactionHour: Hour of the transaction (0-23)
            - TransactionDay: Day of the week (0=Monday, 6=Sunday)
            - TransactionMonth: Month of the transaction (1-12)
            - TransactionYear: Year of the transaction
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the extracted temporal features.
    """

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