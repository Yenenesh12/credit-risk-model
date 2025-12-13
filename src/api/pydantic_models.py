from pydantic import BaseModel, Field, validator, confloat, conint
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk categories for predictions."""
    LOW_RISK = "Low Risk"
    MEDIUM_RISK = "Medium Risk"
    HIGH_RISK = "High Risk"
    VERY_HIGH_RISK = "Very High Risk"


class Recommendation(str, Enum):
    """Recommendation categories."""
    APPROVE = "Approve"
    REVIEW = "Review"
    REQUIRES_DOCUMENTATION = "Requires Additional Documentation"
    DECLINE = "Decline"


class CreditApplication(BaseModel):
    """Credit application data model."""
    
    # Demographic information
    age: conint(ge=18, le=100) = Field(..., description="Applicant age")
    income: confloat(ge=0) = Field(..., description="Annual income")
    debt: confloat(ge=0) = Field(..., description="Total debt")
    savings: confloat(ge=0) = Field(..., description="Total savings")
    
    # Credit information
    credit_score: conint(ge=300, le=850) = Field(..., description="Credit score")
    credit_history_length: conint(ge=0) = Field(..., description="Credit history length in years")
    num_open_accounts: conint(ge=0) = Field(..., description="Number of open credit accounts")
    num_delinquent_accounts: conint(ge=0) = Field(..., description="Number of delinquent accounts")
    
    # Employment information
    employment_length: conint(ge=0) = Field(..., description="Employment length in years")
    employment_status: str = Field(..., description="Employment status")
    
    # Loan information
    loan_amount: confloat(ge=0) = Field(..., description="Requested loan amount")
    loan_term: conint(ge=1, le=30) = Field(..., description="Loan term in years")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    
    # Additional features
    debt_to_income_ratio: Optional[confloat(ge=0)] = Field(None, description="Debt to income ratio")
    credit_utilization: Optional[confloat(ge=0, le=1)] = Field(None, description="Credit utilization ratio")
    payment_history: Optional[confloat(ge=0, le=1)] = Field(None, description="Payment history score")
    
    @validator('debt_to_income_ratio', always=True)
    def calculate_dti(cls, v, values):
        """Calculate debt-to-income ratio if not provided."""
        if v is not None:
            return v
        
        if 'income' in values and 'debt' in values:
            if values['income'] > 0:
                return values['debt'] / values['income']
        
        return 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000,
                "debt": 10000,
                "savings": 20000,
                "credit_score": 720,
                "credit_history_length": 10,
                "num_open_accounts": 5,
                "num_delinquent_accounts": 0,
                "employment_length": 5,
                "employment_status": "employed",
                "loan_amount": 15000,
                "loan_term": 3,
                "loan_purpose": "home_improvement",
                "debt_to_income_ratio": 0.2,
                "credit_utilization": 0.3,
                "payment_history": 0.95
            }
        }


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    application_id: str = Field(..., description="Unique application identifier")
    application: CreditApplication = Field(..., description="Credit application data")
    
    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP123456",
                "application": CreditApplication.Config.schema_extra["example"]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    application_id: str = Field(..., description="Unique application identifier")
    probability_default: confloat(ge=0, le=1) = Field(..., description="Probability of default")
    prediction: int = Field(..., description="Binary prediction (0=approve, 1=decline)")
    risk_category: RiskCategory = Field(..., description="Risk category")
    recommendation: Recommendation = Field(..., description="Recommendation")
    confidence_interval: List[confloat(ge=0, le=1)] = Field(
        ..., 
        description="95% confidence interval for probability"
    )
    threshold_used: confloat(ge=0, le=1) = Field(..., description="Threshold used for binary prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP123456",
                "probability_default": 0.15,
                "prediction": 0,
                "risk_category": "Low Risk",
                "recommendation": "Approve",
                "confidence_interval": [0.12, 0.18],
                "threshold_used": 0.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictionItem(BaseModel):
    """Item for batch prediction request."""
    
    application_id: str = Field(..., description="Unique application identifier")
    application: CreditApplication = Field(..., description="Credit application data")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    applications: List[BatchPredictionItem] = Field(
        ..., 
        description="List of credit applications",
        max_items=1000  # Limit batch size
    )
    
    @validator('applications')
    def validate_unique_ids(cls, v):
        """Validate that all application IDs are unique."""
        ids = [item.application_id for item in v]
        if len(ids) != len(set(ids)):
            raise ValueError("All application IDs must be unique")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_type: str = Field(..., description="Type of model")
    threshold: confloat(ge=0, le=1) = Field(..., description="Current prediction threshold")
    n_features: int = Field(..., description="Number of features")
    feature_names: List[str] = Field(..., description="Feature names")
    prediction_stats: Dict[str, Any] = Field(..., description="Prediction statistics")
    deployment_time: str = Field(..., description="Model deployment timestamp")


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    message: str = Field(..., description="Error message")
    error: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")