import pandas as pd
from src.time_feature_extractor import TimeFeatureExtractor

def test_extract_transaction_hour():
    df = pd.DataFrame({'TransactionStartTime': ['2025-01-01 08:30:00']})
    te = TimeFeatureExtractor('TransactionStartTime')
    result = te.transform(df)


    assert result['TransactionHour'].iloc[0] == 8