"""
FastAPI application for credit risk prediction API.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import uuid
import logging
import asyncio

from .pydantic_models import (
    Transaction, BatchPredictionRequest, 
    PredictionResponse, BatchPredictionResponse,
    ModelInfo, HealthCheck
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk based on transaction data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_info = {}

class ModelManager:
    """Manages model loading and prediction."""
    
    @staticmethod
    def load_model(model_path: str = "../models/gradient_boosting_model.pkl"):
        """Load the trained model."""
        global model, model_info
        
        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            model_info = {
                'model_type': model_data['model_type'],
                'best_params': model_data['best_params'],
                'threshold': model_data.get('threshold', 0.5),
                'metrics': model_data.get('metrics', {}),
                'loaded_at': datetime.now().isoformat()
            }
            logger.info(f"Model loaded successfully: {model_info['model_type']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def load_preprocessor(preprocessor_path: str = "../data/processed/preprocessor.pkl"):
        """Load the preprocessor."""
        global preprocessor
        
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise
    
    @staticmethod
    def preprocess_transaction(transaction: Dict) -> pd.DataFrame:
        """Preprocess a single transaction."""
        global preprocessor
        
        if preprocessor is None:
            raise HTTPException(status_code=500, detail="Preprocessor not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Get config from preprocessor
        config = preprocessor.get('config', {})
        numerical_features = config.get('features', {}).get('numerical', [])
        categorical_features = config.get('features', {}).get('categorical', [])
        
        # Filter to available features
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # 1. Handle missing values
        imputer = preprocessor.get('imputer')
        if imputer and numerical_features:
            df[numerical_features] = imputer.transform(df[numerical_features])
        
        # 2. Scale numerical features
        scaler = preprocessor.get('scaler')
        if scaler and numerical_features:
            df[numerical_features] = scaler.transform(df[numerical_features])
        
        # 3. Encode categorical features
        encoder = preprocessor.get('encoder')
        if encoder and categorical_features:
            encoded_array = encoder.transform(df[categorical_features])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(categorical_features),
                index=df.index
            )
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)
        
        # Ensure columns match training
        feature_names = preprocessor.get('feature_names')
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]
        
        return df
    
    @staticmethod
    def predict_single(transaction: Dict) -> Dict[str, Any]:
        """Make prediction for a single transaction."""
        global model, model_info
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Preprocess
            df_processed = ModelManager.preprocess_transaction(transaction)
            
            # Predict
            threshold = model_info.get('threshold', 0.5)
            proba = model.predict_proba(df_processed)[0, 1]
            prediction = int(proba >= threshold)
            
            # Determine risk level
            if proba < 0.3:
                risk_level = "Low"
            elif proba < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                "prediction": prediction,
                "probability": float(proba),
                "risk_level": risk_level,
                "threshold_used": threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @staticmethod
    def predict_batch(transactions: List[Dict]) -> List[Dict[str, Any]]:
        """Make predictions for a batch of transactions."""
        global model, model_info
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Preprocess all transactions
            if preprocessor is None:
                raise HTTPException(status_code=500, detail="Preprocessor not loaded")
            
            # Get config
            config = preprocessor.get('config', {})
            numerical_features = config.get('features', {}).get('numerical', [])
            categorical_features = config.get('features', {}).get('categorical', [])
            
            # Filter to available features
            numerical_features = [f for f in numerical_features if f in df.columns]
            categorical_features = [f for f in categorical_features if f in df.columns]
            
            # Apply preprocessing
            imputer = preprocessor.get('imputer')
            if imputer and numerical_features:
                df[numerical_features] = imputer.transform(df[numerical_features])
            
            scaler = preprocessor.get('scaler')
            if scaler and numerical_features:
                df[numerical_features] = scaler.transform(df[numerical_features])
            
            encoder = preprocessor.get('encoder')
            if encoder and categorical_features:
                encoded_array = encoder.transform(df[categorical_features])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=encoder.get_feature_names_out(categorical_features),
                    index=df.index
                )
                df = df.drop(columns=categorical_features)
                df = pd.concat([df, encoded_df], axis=1)
            
            # Ensure columns match training
            feature_names = preprocessor.get('feature_names')
            if feature_names:
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0
                df = df[feature_names]
            
            # Make predictions
            threshold = model_info.get('threshold', 0.5)
            probas = model.predict_proba(df)[:, 1]
            predictions = (probas >= threshold).astype(int)
            
            # Prepare results
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, probas)):
                # Determine risk level
                if proba < 0.3:
                    risk_level = "Low"
                elif proba < 0.7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                # Add transaction ID if available
                transaction_id = None
                if i < len(transactions) and 'TransactionId' in transactions[i]:
                    transaction_id = transactions[i]['TransactionId']
                
                results.append({
                    "transaction_id": transaction_id,
                    "prediction": int(pred),
                    "probability": float(proba),
                    "risk_level": risk_level,
                    "threshold_used": threshold
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    try:
        ModelManager.load_model()
        ModelManager.load_preprocessor()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

# Health check endpoint
@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        timestamp=datetime.now().isoformat()
    )

# Model info endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return ModelInfo(
        model_type=model_info.get('model_type', 'unknown'),
        threshold=model_info.get('threshold', 0.5),
        metrics=model_info.get('metrics', {}),
        loaded_at=model_info.get('loaded_at', 'unknown')
    )

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Make a credit risk prediction for a single transaction.
    
    - **transaction**: Transaction data including amount, product category, etc.
    - Returns: Prediction with risk score and probability
    """
    logger.info(f"Received prediction request for transaction")
    
    # Convert transaction to dict
    transaction_dict = transaction.dict(exclude_unset=True)
    
    # Make prediction
    result = ModelManager.predict_single(transaction_dict)
    
    # Prepare response
    response = PredictionResponse(
        transaction_id=transaction.TransactionId if hasattr(transaction, 'TransactionId') else str(uuid.uuid4()),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        threshold_used=result["threshold_used"],
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"Prediction completed: {response}")
    return response

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks = None):
    """
    Make credit risk predictions for a batch of transactions.
    
    - **transactions**: List of transaction data
    - **callback_url**: Optional URL for async callback (for large batches)
    - Returns: Batch prediction results
    """
    logger.info(f"Received batch prediction request with {len(request.transactions)} transactions")
    
    # Check if batch is too large for synchronous processing
    if len(request.transactions) > 1000 and request.callback_url:
        # Process asynchronously
        if background_tasks:
            job_id = str(uuid.uuid4())
            background_tasks.add_task(
                process_batch_async,
                job_id=job_id,
                transactions=request.transactions,
                callback_url=request.callback_url
            )
            
            return BatchPredictionResponse(
                job_id=job_id,
                status="processing",
                message=f"Batch processing started. Results will be sent to {request.callback_url}",
                timestamp=datetime.now().isoformat(),
                predictions=[]
            )
    
    # Process synchronously
    try:
        # Convert transactions to dicts
        transactions_dict = [t.dict(exclude_unset=True) for t in request.transactions]
        
        # Make predictions
        predictions = ModelManager.predict_batch(transactions_dict)
        
        # Prepare response
        response = BatchPredictionResponse(
            job_id=str(uuid.uuid4()),
            status="completed",
            message=f"Successfully processed {len(predictions)} transactions",
            timestamp=datetime.now().isoformat(),
            predictions=predictions
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update threshold endpoint
@app.post("/model/threshold")
async def update_threshold(threshold: float = Field(..., gt=0, lt=1)):
    """
    Update the prediction threshold.
    
    - **threshold**: New threshold value (0 < threshold < 1)
    """
    global model_info
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    old_threshold = model_info.get('threshold', 0.5)
    model_info['threshold'] = threshold
    
    logger.info(f"Threshold updated from {old_threshold} to {threshold}")
    
    return JSONResponse(
        status_code=200,
        content={
            "message": f"Threshold updated successfully",
            "old_threshold": old_threshold,
            "new_threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
    )

# Async batch processing function
async def process_batch_async(job_id: str, transactions: List[Transaction], callback_url: str):
    """Process large batch asynchronously."""
    logger.info(f"Starting async batch processing for job {job_id}")
    
    try:
        # Convert transactions to dicts
        transactions_dict = [t.dict(exclude_unset=True) for t in transactions]
        
        # Make predictions
        predictions = ModelManager.predict_batch(transactions_dict)
        
        # Prepare result
        result = {
            "job_id": job_id,
            "status": "completed",
            "message": f"Successfully processed {len(predictions)} transactions",
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions
        }
        
        # Send callback (simplified - in production, use proper HTTP client)
        logger.info(f"Job {job_id} completed. Would send results to {callback_url}")
        
    except Exception as e:
        logger.error(f"Async batch processing failed for job {job_id}: {e}")
        
        # Send error callback
        error_result = {
            "job_id": job_id,
            "status": "failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "predictions": []
        }
        logger.info(f"Would send error results to {callback_url}")

# Model metrics endpoint
@app.get("/model/metrics")
async def get_model_metrics():
    """Get model performance metrics."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    metrics = model_info.get('metrics', {})
    
    return JSONResponse(
        status_code=200,
        content={
            "metrics": metrics,
            "model_type": model_info.get('model_type', 'unknown'),
            "threshold": model_info.get('threshold', 0.5),
            "timestamp": datetime.now().isoformat()
        }
    )

# Example data endpoint
@app.get("/example/transaction")
async def get_example_transaction():
    """Get example transaction data for testing."""
    example = {
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
    
    return JSONResponse(status_code=200, content=example)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)