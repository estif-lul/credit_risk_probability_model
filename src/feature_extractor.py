import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xverse.transformer import WOE

from time_feature_extractor import TimeFeatureExtractor


def main(df):

    numeric_features = ['Amount', 'Value']
    categorical_features = ['ProductCategory', 'ProviderId', 'ProductId', 'PricingStrategy', 'ChannelId']
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
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
        ('time', time_pipeline, [time_column])
    ])

    features = preprocessor.fit_transform(df)
    feature_names = (
        numeric_features + 
        list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) +
        ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'])
    
    processed_df = pd.DataFrame(features.toarray(), columns=feature_names)
    processed_df.to_csv('data/processed/processed_features.csv', index=False)

# def preprocess_features(df, preprocessor):
#     features = preprocessor.fit_transform(df)
#     feature_names = (
#         numeric_features + 
#         list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) +
#         ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'])

def aggregate_customer_features(df):
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

if __name__ == "__main__":
    df = pd.read_csv('data/raw/data.csv')
    # main(df)
    # customer_features = aggregate_customer_features(df)
    # print(customer_features.head())
    # customer_features.to_csv('data/processed/customer_features.csv', index=False)
    n_df, w_df = iv_woe(df, 'FraudResult', bins=10, show_woe=True)