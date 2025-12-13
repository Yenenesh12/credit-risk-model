"""
FastAPI application for credit risk model serving.
Provides REST API for model predictions.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from .pydantic_models import (
    CreditApplication, 
    PredictionRequest, 
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthCheck
)
from ..predict import create_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
predictor = None
prediction_history = []
MAX_HISTORY_SIZE = 10000

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting up Credit Risk Model API...")
    
    global predictor
    try:
        predictor = create_predictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Credit Risk Model API...")
    # Cleanup resources if needed

# Create FastAPI app
app = FastAPI(
    title="Credit Risk Model API",
    description="API for credit risk prediction using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests."""
    start_time = datetime.now()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Log request details
    logger.info(
        f"Method: {request.method} "
        f"Path: {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {process_time:.2f}ms"
    )
    
    return response

# Dependency to get predictor
def get_predictor():
    """Dependency to get the predictor instance."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return predictor

# Health check endpoint
@app.get("/health", response_model=HealthCheck, tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "service": "credit-risk-model-api",
        "version": "1.0.0"
    }

# Model information endpoint
@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(predictor_obj: CreditRiskPredictor = Depends(get_predictor)):
    """Get information about the deployed model."""
    
    try:
        # Get model type
        model_type = type(predictor_obj.model).__name__
        
        # Get feature names if available
        feature_names = []
        try:
            if hasattr(predictor_obj.preprocessor, 'get_feature_names'):
                feature_names = predictor_obj.preprocessor.get_feature_names()
        except:
            pass
        
        # Get prediction statistics
        stats = predictor_obj.get_prediction_stats()
        
        return {
            "model_type": model_type,
            "threshold": predictor_obj.threshold,
            "n_features": len(feature_names),
            "feature_names": feature_names[:10],  # Return first 10 features
            "prediction_stats": stats,
            "deployment_time": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    predictor_obj: CreditRiskPredictor = Depends(get_predictor)
):
    """Make a single credit risk prediction."""
    
    try:
        logger.info(f"Received prediction request for application ID: {request.application_id}")
        
        # Convert to dictionary for predictor
        application_data = request.application.dict()
        
        # Make prediction
        predictions = predictor_obj.predict_with_confidence(application_data)
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        prediction = predictions[0]
        
        # Create response
        response = PredictionResponse(
            application_id=request.application_id,
            probability_default=prediction['probability_default'],
            prediction=prediction['prediction'],
            risk_category=prediction['risk_category'],
            recommendation=prediction['recommendation'],
            confidence_interval=prediction['confidence_interval'],
            threshold_used=prediction['threshold_used'],
            timestamp=datetime.now().isoformat()
        )
        
        # Store in history (in background)
        background_tasks.add_task(
            store_prediction_history,
            request.application_id,
            prediction,
            request.application
        )
        
        logger.info(f"Prediction completed for {request.application_id}: {prediction['risk_category']}")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    predictor_obj: CreditRiskPredictor = Depends(get_predictor)
):
    """Make batch predictions."""
    
    try:
        logger.info(f"Received batch prediction request with {len(request.applications)} applications")
        
        # Convert to list of dictionaries
        applications_data = []
        application_ids = []
        
        for app_req in request.applications:
            applications_data.append(app_req.application.dict())
            application_ids.append(app_req.application_id)
        
        # Convert to DataFrame
        df = pd.DataFrame(applications_data)
        
        # Make predictions
        predictions = predictor_obj.predict_with_confidence(df)
        
        # Prepare responses
        responses = []
        for idx, (app_id, pred) in enumerate(zip(application_ids, predictions)):
            response = PredictionResponse(
                application_id=app_id,
                probability_default=pred['probability_default'],
                prediction=pred['prediction'],
                risk_category=pred['risk_category'],
                recommendation=pred['recommendation'],
                confidence_interval=pred['confidence_interval'],
                threshold_used=pred['threshold_used'],
                timestamp=datetime.now().isoformat()
            )
            responses.append(response)
            
            # Store in history (in background)
            background_tasks.add_task(
                store_prediction_history,
                app_id,
                pred,
                applications_data[idx]
            )
        
        logger.info(f"Batch prediction completed: {len(responses)} predictions")
        
        return BatchPredictionResponse(predictions=responses)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Threshold adjustment endpoint
@app.post("/model/threshold", tags=["Model"])
async def set_threshold(
    new_threshold: float = Field(0.5, ge=0.0, le=1.0, description="New prediction threshold"),
    predictor_obj: CreditRiskPredictor = Depends(get_predictor)
):
    """Adjust the prediction threshold."""
    
    if not 0 <= new_threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    
    old_threshold = predictor_obj.threshold
    predictor_obj.threshold = new_threshold
    
    logger.info(f"Threshold updated from {old_threshold} to {new_threshold}")
    
    return {
        "message": "Threshold updated successfully",
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "timestamp": datetime.now().isoformat()
    }

# Prediction history endpoint
@app.get("/predictions/history", tags=["Monitoring"])
async def get_prediction_history(
    limit: int = Field(100, ge=1, le=1000, description="Number of records to return"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get prediction history."""
    
    try:
        # Filter by date if provided
        filtered_history = prediction_history
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            filtered_history = [h for h in filtered_history 
                              if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) >= start_dt]
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            filtered_history = [h for h in filtered_history 
                              if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) <= end_dt]
        
        # Apply limit
        result = filtered_history[-limit:]
        
        return {
            "count": len(result),
            "total": len(prediction_history),
            "predictions": result
        }
    
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving prediction history")

# Download predictions endpoint
@app.get("/predictions/download", tags=["Monitoring"])
async def download_predictions(
    format: str = Field("csv", regex="^(csv|json)$"),
    limit: int = Field(1000, ge=1, le=10000)
):
    """Download prediction history."""
    
    try:
        # Get limited history
        data = prediction_history[-limit:]
        
        if not data:
            raise HTTPException(status_code=404, detail="No prediction history available")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            filename = f"predictions_{timestamp}.csv"
            filepath = f"/tmp/{filename}"
            df.to_csv(filepath, index=False)
            return FileResponse(
                path=filepath,
                filename=filename,
                media_type="text/csv"
            )
        
        elif format == "json":
            filename = f"predictions_{timestamp}.json"
            filepath = f"/tmp/{filename}"
            df.to_json(filepath, orient="records", indent=2)
            return FileResponse(
                path=filepath,
                filename=filename,
                media_type="application/json"
            )
    
    except Exception as e:
        logger.error(f"Error downloading predictions: {e}")
        raise HTTPException(status_code=500, detail="Error downloading predictions")

# Model metrics endpoint
@app.get("/model/metrics", tags=["Monitoring"])
async def get_model_metrics(predictor_obj: CreditRiskPredictor = Depends(get_predictor)):
    """Get current model metrics and statistics."""
    
    try:
        stats = predictor_obj.get_prediction_stats()
        
        # Calculate additional metrics
        if prediction_history:
            recent_predictions = prediction_history[-1000:]
            probs = [p['probability_default'] for p in recent_predictions]
            
            additional_metrics = {
                "recent_mean_probability": float(np.mean(probs)),
                "recent_std_probability": float(np.std(probs)),
                "approval_rate_recent": float(np.mean([p['prediction'] == 0 for p in recent_predictions])),
                "high_risk_rate_recent": float(np.mean([p['risk_category'] == 'High Risk' for p in recent_predictions])),
                "history_size": len(prediction_history)
            }
        else:
            additional_metrics = {}
        
        return {
            "basic_stats": stats,
            "additional_metrics": additional_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model metrics")

# Store prediction in history
def store_prediction_history(application_id: str, prediction: Dict, application_data: Dict):
    """Store prediction in history with size limit."""
    
    history_entry = {
        "application_id": application_id,
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction['prediction'],
        "probability_default": prediction['probability_default'],
        "risk_category": prediction['risk_category'],
        "recommendation": prediction['recommendation'],
        "application_data": application_data
    }
    
    prediction_history.append(history_entry)
    
    # Maintain size limit
    if len(prediction_history) > MAX_HISTORY_SIZE:
        # Remove oldest entries
        del prediction_history[:len(prediction_history) - MAX_HISTORY_SIZE]

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Risk Model API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "model_metrics": "/model/metrics",
            "prediction_history": "/predictions/history"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )