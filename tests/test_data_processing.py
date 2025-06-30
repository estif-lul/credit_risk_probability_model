import pandas as pd
from fastapi.testclient import TestClient

import sys
sys.path.insert(1, 'src/')
from time_feature_extractor import TimeFeatureExtractor
from api.main import predict_risk, ApplicationData, app

def test_extract_transaction_hour():
    df = pd.DataFrame({'TransactionStartTime': ['2025-01-01 08:30:00']})
    te = TimeFeatureExtractor('TransactionStartTime')
    result = te.transform(df)


    assert result['TransactionHour'].iloc[0] == 8

client = TestClient(app)

def test_prediction_response_format():
    payload = {
        "recency1":.826966,
        "frequency":-0.253459,
        "monetary":-0.059529,
        "total_transactions":-0.253459,
        "total_amount":-0.059529,
        "avg_amount":-0.034087,
        "std_amount":-0.140432,
        "total_value":-0.089524,
        "avg_value":-0.052297,
        "std_value":-0.131508,
        "unique_providers":-1.382737,
        "unique_products":-1.115175,
        "unique_channels":-1.404749,
        "fraud_count":-0.066617,
        "fraud_rate":-0.086096
    }
    

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 'risk_probability' in response.json()
    assert isinstance(response.json()['risk_probability'], float)