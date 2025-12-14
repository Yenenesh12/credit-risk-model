import os
import uuid
import pickle
import joblib
import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="Credit Risk Prediction API")

# ----------------------
# Globals
# ----------------------
model = None
preprocessor = None
model_info = None

# ----------------------
# Pydantic schemas
# ----------------------
class Transaction(BaseModel):
    AccountId: int
    BatchId: int
    CustomerId: int
    FraudResult: int
    ProductId: int
    ProviderId: int
    SubscriptionId: int
    Amount: float
    Value: float
    CurrencyCode: int
    CountryCode: int
    ProductCategory: int
    ChannelId: int
    PricingStrategy: int
    TransactionStartTime: str
    TransactionId: int

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    probability: float
    risk_level: str
    threshold_used: float
    timestamp: str

# ----------------------
# Model Manager
# ----------------------
class ModelManager:
    @staticmethod
    def load_model(model_path: str = "models/gradient_boosting_model.pkl"):
        global model, model_info
        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            model_info = {
                'model_type': model_data.get('model_type', 'unknown'),
                'best_params': model_data.get('best_params'),
                'threshold': model_data.get('threshold', 0.5),
                'metrics': model_data.get('metrics', {}),
                'feature_names': model_data.get('feature_names'),
                'loaded_at': datetime.now().isoformat()
            }
            if not model_info['feature_names']:
                logger.warning(
                    "Model loaded without 'feature_names'. "
                    "Feature alignment will use post-preprocessing columns. "
                    "Ensure input matches training schema exactly!"
                )
            logger.info(f"Model loaded successfully: {model_info['model_type']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @staticmethod
    def load_preprocessor(preprocessor_path: str = "data/processed/preprocessor.pkl"):
        global preprocessor
        try:
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise

    @staticmethod
    def preprocess_transaction(transaction: Dict) -> pd.DataFrame:
        global preprocessor, model_info

        if preprocessor is None:
            raise HTTPException(status_code=500, detail="Preprocessor not loaded")

        df = pd.DataFrame([transaction])

        # 🔥 DROP NON-FEATURE COLUMNS (likely not used during training)
        NON_FEATURE_COLS = {
            'AccountId', 'BatchId', 'CustomerId', 'TransactionId',
            'TransactionStartTime', 'FraudResult'  # FraudResult is likely the TARGET
        }
        cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
        if cols_to_drop:
            logger.debug(f"Dropping non-feature columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Feature engineering (ONLY if used during training!)
        if 'Amount' in df.columns:
            df['Amount_max'] = df['Amount']
            df['Amount_min'] = df['Amount']
            df['Amount_mean'] = df['Amount']
            df['Amount_std'] = 0.0

        # Preprocessing
        config = preprocessor.get('config', {})
        numerical_features = config.get('features', {}).get('numerical', [])
        categorical_features = config.get('features', {}).get('categorical', [])

        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        # Impute numerical
        imputer = preprocessor.get('imputer')
        if imputer and numerical_features:
            df[numerical_features] = imputer.transform(df[numerical_features])

        # Scale numerical
        scaler = preprocessor.get('scaler')
        if scaler and numerical_features:
            df[numerical_features] = scaler.transform(df[numerical_features])

        # Encode categorical
        encoder = preprocessor.get('encoder')
        if encoder and categorical_features:
            encoded = encoder.transform(df[categorical_features])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(categorical_features),
                index=df.index
            )
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        # Feature alignment
        feature_names = model_info.get('feature_names') if model_info else None
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_names]
        else:
            df = df.reindex(sorted(df.columns), axis=1)

        # 🔥 Ensure all columns are float64 to avoid 'ufunc isnan' error
        try:
            df = df.astype('float64')
        except Exception as e:
            logger.error(f"Failed to convert to float64. dtypes: {df.dtypes.to_dict()}")
            raise HTTPException(status_code=500, detail=f"Non-numeric data in features: {e}")

        return df

    @staticmethod
    def predict_single(transaction: Dict) -> Dict[str, Any]:
        global model, model_info
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        try:
            df_processed = ModelManager.preprocess_transaction(transaction)
            threshold = model_info.get('threshold', 0.5) if model_info else 0.5
            proba = model.predict_proba(df_processed)[0, 1]
            prediction = int(proba >= threshold)

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

# ----------------------
# Startup
# ----------------------
@app.on_event("startup")
async def startup_event():
    ModelManager.load_model()
    ModelManager.load_preprocessor()
    logger.info("API startup complete")

# ----------------------
# Prediction endpoint
# ----------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    logger.info("Received prediction request")
    transaction_dict = transaction.dict()  # No exclude_unset!
    result = ModelManager.predict_single(transaction_dict)
    response = PredictionResponse(
        transaction_id=str(uuid.uuid4()),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        threshold_used=result["threshold_used"],
        timestamp=datetime.now().isoformat()
    )
    logger.info(f"Prediction completed: {response}")
    return response

# ----------------------
# Run uvicorn directly (optional)
# ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)