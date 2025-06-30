from pydantic import BaseModel, Field

class ApplicationData(BaseModel):
    recency: float = Field(..., description="Days since last transaction")
    frequency: float = Field(..., description="Number of transactions in the period")
    monetary: int = Field(..., description="Total monetary value of transactions")
    total_transactions: float = Field(..., description="Total number of transactions")
    total_amount: float = Field(..., description="Sum of all transaction amounts")
    avg_amount: int = Field(..., description="Average transaction amount")
    std_amount: float = Field(..., description="Standard deviation of transaction amounts")
    total_value: float = Field(..., description="Total value of all transactions")
    avg_value: float = Field(..., description="Average value per transaction")
    std_value: float = Field(..., description="Standard deviation of transaction values")
    unique_providers: float = Field(..., description="Number of unique providers used")
    unique_products: float = Field(..., description="Number of unique products purchased")
    unique_channels: float = Field(..., description="Number of unique channels used")
    fraud_count: float = Field(..., description="Number of fraudulent transactions")
    fraud_rate: float = Field(..., description="Rate of fraudulent transactions")
    

class RiskPredictionResponse(BaseModel):
    risk_probability: float = Field(..., description="Predicted probability of default (0.0 to 1.0)")