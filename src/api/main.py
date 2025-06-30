from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel
import mlflow
import pandas as pd

import sys
sys.path.insert(1, 'src/')
from get_best_model import GetBestModel
from api.pydantic_models import RiskPredictionResponse

app = FastAPI()

# Load the registered model
model = GetBestModel()

class ApplicationData(BaseModel):
    recency: float
    frequency: float
    monetary: int
    total_transactions: float
    total_amount: float
    avg_amount: int
    std_amount: float
    total_value: float
    avg_value: float
    std_value: float
    unique_providers: float
    unique_products: float
    unique_channels: float
    fraud_count: float
    fraud_rate: float

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_risk(input: ApplicationData):
    data = pd.DataFrame([input.model_dump()])
    proba = model.predict_proba(data)[0][1]
    return RiskPredictionResponse(risk_probability=round(proba, 4))