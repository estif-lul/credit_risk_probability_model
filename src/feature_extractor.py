import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xverse.transformer import WOE

from time_feature_extractor import TimeFeatureExtractor


def preprocess_features(df):
    """
    Preprocesses the input DataFrame by applying feature engineering and transformation pipelines.
    This function performs the following steps:
    - Applies median imputation and standard scaling to numeric features ('Amount', 'Value').
    - Applies most frequent imputation and one-hot encoding to categorical features 
      ('ProductCategory', 'ProviderId', 'ProductId', 'PricingStrategy', 'ChannelId', 'FraudResult').
    - Extracts time-based features (hour, day, month, year) from the 'TransactionStartTime' column.
    - Combines all transformed features into a single DataFrame.
    - Optionally appends the 'is_high_risk' column to the processed DataFrame if it exists in the input.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing raw features.
    Returns:
        processed_df (pd.DataFrame): DataFrame containing preprocessed features and, if present, the 'is_high_risk' column.
        preprocessor (ColumnTransformer): Fitted preprocessor object for transforming new data.
    """


    numeric_features = ['Amount', 'Value']
    categorical_features = ['ProductCategory', 'ProviderId', 'ProductId', 'PricingStrategy', 'ChannelId', 'FraudResult']
    time_column = 'TransactionStartTime'

    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    time_pipeline = Pipeline(steps=[
        ('time_features', TimeFeatureExtractor(time_column=time_column))
    ])


    preprocessor = ColumnTransformer(transformers=[
        ('time', time_pipeline, [time_column]),
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    print(df.columns)
    features = preprocessor.fit_transform(df)
    feature_names = (
        numeric_features + 
        list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) +
        ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'])
    
    processed_df = pd.DataFrame(features.toarray(), columns=feature_names)
    processed_df = pd.concat([processed_df, df['is_high_risk'].reset_index(drop=True)], axis=1) if 'is_high_risk' in df.columns else pd.DataFrame()

    return processed_df, preprocessor

def aggregate_customer_features(df):
    """
    Aggregates transaction-level features into customer-level features.
    Groups the input DataFrame by 'CustomerId' and computes various aggregate statistics
    for each customer, including total and average transaction amounts, transaction count,
    total and average value, and diversity metrics for channels and product categories.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least the following columns:
            - 'CustomerId'
            - 'Amount'
            - 'TransactionId'
            - 'Value'
            - 'ChannelId'
            - 'ProductCategory'
    Returns:
        pd.DataFrame: DataFrame with one row per customer and the following columns:
            - 'CustomerId'
            - 'total_transaction_amount'
            - 'avg_transaction_amount'
            - 'std_transaction_amount'
            - 'transaction_count'
            - 'total_value'
            - 'avg_value'
            - 'channel_diversity'
            - 'product_diversity'
    """

    customer_df = df.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        std_transaction_amount=('Amount', 'std'),
        transaction_count=('TransactionId', 'count'),
        total_value=('Value', 'sum'),
        avg_value=('Value', 'mean'),
        channel_diversity=('ChannelId', pd.Series.nunique),
        product_diversity=('ProductCategory', pd.Series.nunique)
    ).reset_index()

    return customer_df


def iv_woe(data, target, bins=10, show_woe=False):
    """
    Calculates the Information Value (IV) and Weight of Evidence (WoE) for each feature in a DataFrame with respect to a binary target variable.
    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the features and the target variable.
    target : str
        The name of the target column in the DataFrame. The target should be binary (0/1).
    bins : int, optional (default=10)
        The number of bins to use for numeric features with more than 10 unique values. Binning is performed using quantiles.
    show_woe : bool, optional (default=False)
        If True, prints the WoE table for each feature.
    Returns
    -------
    newDF : pandas.DataFrame
        A DataFrame containing the Information Value (IV) for each feature.
    woeDF : pandas.DataFrame
        A DataFrame containing the WoE calculation details for each feature and bin/category.
    Notes
    -----
    - Features with more than 10 unique numeric values are binned using quantile-based discretization.
    - WoE and IV are commonly used in credit scoring and risk modeling to evaluate the predictive power of features.
    - Small event/non-event counts are floored at 0.5 to avoid division by zero and infinite WoE values.
    """
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF
