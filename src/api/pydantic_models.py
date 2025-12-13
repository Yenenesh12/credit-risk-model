"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class Transaction(BaseModel):
    """Transaction data model."""
    TransactionId: Optional[str] = Field(None, description="Unique transaction identifier")
    BatchId: Optional[int] = Field(None, description="Batch identifier")
    AccountId: Optional[int] = Field(None, description="Account identifier")
    SubscriptionId: Optional[int] = Field(None, description="Subscription identifier")
    CustomerId: Optional[str] = Field(None, description="Customer identifier")
    CurrencyCode: Optional[str] = Field(None, description="Currency code")
    CountryCode: Optional[int] = Field(None, description="Country code")
    ProviderId: Optional[str] = Field(None, description="Provider identifier")
    ProductId: Optional[str] = Field(None, description="Product identifier")
    ProductCategory: Optional[str] = Field(None, description="Product category")
    ChannelId: Optional[str] = Field(None, description="Channel identifier")
    Amount: float = Field(..., description="Transaction amount (positive for debit)")
    Value: Optional[float] = Field(None, description="Absolute transaction value")
    TransactionStartTime: Optional[str] = Field(None, description="Transaction timestamp")
    PricingStrategy: Optional[str] = Field(None, description="Pricing strategy")
    FraudResult: Optional[int] = Field(None, description="Fraud indicator (0 or 1)")
    
    @validator('Amount')
    def validate_amount(cls, v):
        """Validate amount is reasonable."""
        if abs(v) > 1000000:  # 1 million limit
            raise ValueError('Amount exceeds reasonable limit')
        return v
    
    @validator('FraudResult')
    def validate_fraud_result(cls, v):
        """Validate fraud result is 0 or 1."""
        if v is not None and v not in [0, 1]:
            raise ValueError('FraudResult must be 0 or 1')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "TransactionId": "TXN123456",
                "Amount": 150.75,
                "Value": 150.75,
                "ProductCategory": "Electronics",
                "ChannelId": "Web",
                "ProviderId": "P1",
                "CountryCode": 254,
                "CurrencyCode": "USD",
                "TransactionStartTime": "2023-10-01T14:30:00"
            }
        }

class PredictionResult(BaseModel):
    """Single prediction result."""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    prediction: int = Field(..., description="Binary prediction (0=low risk, 1=high risk)")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    threshold_used: float = Field(..., ge=0, le=1, description="Threshold used for classification")

class PredictionResponse(BaseModel):
    """Response for single prediction."""
    transaction_id: str = Field(..., description="Transaction identifier")
    prediction: int = Field(..., description="Binary prediction")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    risk_level: RiskLevel = Field(..., description="Risk level")
    threshold_used: float = Field(..., ge=0, le=1, description="Threshold used")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN123456",
                "prediction": 0,
                "probability": 0.23,
                "risk_level": "Low",
                "threshold_used": 0.5,
                "timestamp": "2023-10-01T14:30:00Z"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""
    transactions: List[Transaction] = Field(..., description="List of transactions")
    callback_url: Optional[str] = Field(None, description="Callback URL for async processing")

class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status (processing/completed/failed)")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Response timestamp")
    predictions: List[PredictionResult] = Field([], description="List of predictions")

class ModelInfo(BaseModel):
    """Model information."""
    model_type: str = Field(..., description="Type of model")
    threshold: float = Field(..., description="Current prediction threshold")
    metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    loaded_at: str = Field(..., description="When model was loaded")

class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    timestamp: str = Field(..., description="Check timestamp")

class ThresholdUpdate(BaseModel):
    """Threshold update request."""
    threshold: float = Field(..., gt=0, lt=1, description="New threshold value")
    
    class Config:
        schema_extra = {
            "example": {
                "threshold": 0.6
            }
        }

class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Amount exceeds reasonable limit",
                "timestamp": "2023-10-01T14:30:00Z"
            }
        }