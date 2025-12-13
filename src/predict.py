"""
Inference pipeline for credit risk model predictions.
Includes batch and real-time prediction capabilities.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Union, Optional
import logging
from datetime import datetime
import json
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """Credit risk predictor for production inference."""
    
    def __init__(self, 
                 model_path: str = 'models/best_model.pkl',
                 preprocessor_path: str = 'models/preprocessor.pkl',
                 threshold: float = 0.5,
                 calibration_data_path: Optional[str] = None):
        
        self.threshold = threshold
        self.calibration_data_path = calibration_data_path
        
        # Load model and preprocessor
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load calibration data if provided
        self.calibration_data = None
        if calibration_data_path:
            self._load_calibration_data()
        
        # Initialize prediction cache for monitoring
        self.prediction_cache = []
        self.cache_size = 1000
        
        logger.info("Predictor initialized successfully")
    
    def _load_calibration_data(self):
        """Load calibration data for probability calibration."""
        try:
            self.calibration_data = pd.read_csv(self.calibration_data_path)
            logger.info(f"Loaded calibration data with {len(self.calibration_data)} samples")
        except Exception as e:
            logger.warning(f"Could not load calibration data: {e}")
            self.calibration_data = None
    
    def _apply_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply probability calibration using isotonic regression."""
        if self.calibration_data is None or len(self.calibration_data) < 100:
            logger.warning("Insufficient calibration data, using raw probabilities")
            return probabilities
        
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Fit isotonic regression on calibration data
            X_calib = self.preprocess(self.calibration_data.drop(columns=['target'], errors='ignore'))
            y_calib = self.calibration_data['target'].values
            
            # Get uncalibrated probabilities
            uncalibrated_probs = self.model.predict_proba(X_calib)[:, 1]
            
            # Fit calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(uncalibrated_probs, y_calib)
            
            # Apply calibration
            calibrated_probs = calibrator.transform(probabilities)
            
            logger.info("Applied probability calibration")
            return calibrated_probs
        
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return probabilities
    
    def preprocess(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                data = pd.DataFrame([data])
        
        # Store original data for reference
        self.original_data = data.copy()
        
        # Apply preprocessing pipeline
        try:
            processed_data = self.preprocessor.transform(data)
            logger.info(f"Preprocessed data shape: {processed_data.shape}")
            return processed_data
        
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def predict_proba(self, 
                     data: Union[pd.DataFrame, Dict],
                     calibrated: bool = True) -> np.ndarray:
        """Predict default probabilities."""
        
        # Preprocess data
        processed_data = self.preprocess(data)
        
        # Get probabilities
        try:
            probabilities = self.model.predict_proba(processed_data)[:, 1]
            
            # Apply calibration if requested
            if calibrated:
                probabilities = self._apply_calibration(probabilities)
            
            return probabilities
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict(self, 
               data: Union[pd.DataFrame, Dict],
               threshold: Optional[float] = None,
               calibrated: bool = True) -> np.ndarray:
        """Predict binary default classification."""
        
        if threshold is None:
            threshold = self.threshold
        
        # Get probabilities
        probabilities = self.predict_proba(data, calibrated=calibrated)
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def predict_with_confidence(self,
                              data: Union[pd.DataFrame, Dict],
                              calibrated: bool = True) -> List[Dict]:
        """Predict with confidence intervals and explanations."""
        
        predictions = []
        probabilities = self.predict_proba(data, calibrated=calibrated)
        
        for idx, prob in enumerate(probabilities):
            # Calculate confidence interval (simplified)
            alpha = 0.05  # 95% confidence
            n = len(data) if isinstance(data, pd.DataFrame) else 1
            se = np.sqrt(prob * (1 - prob) / n)
            z_score = stats.norm.ppf(1 - alpha/2)
            
            confidence_interval = [
                max(0, prob - z_score * se),
                min(1, prob + z_score * se)
            ]
            
            # Determine risk category
            if prob < 0.2:
                risk_category = "Low Risk"
                recommendation = "Approve"
            elif prob < 0.5:
                risk_category = "Medium Risk"
                recommendation = "Review"
            elif prob < 0.8:
                risk_category = "High Risk"
                recommendation = "Requires Additional Documentation"
            else:
                risk_category = "Very High Risk"
                recommendation = "Decline"
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(idx, data)
            
            prediction = {
                'probability_default': float(prob),
                'prediction': int(prob >= self.threshold),
                'confidence_interval': [float(ci) for ci in confidence_interval],
                'risk_category': risk_category,
                'recommendation': recommendation,
                'threshold_used': float(self.threshold),
                'feature_importance': feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            
            predictions.append(prediction)
        
        # Cache predictions for monitoring
        self._cache_predictions(predictions)
        
        return predictions
    
    def _get_feature_importance(self, 
                               idx: int, 
                               data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """Get feature importance for a specific prediction."""
        
        try:
            # For tree-based models
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = getattr(self.preprocessor, 'get_feature_names', lambda: [])()
                
                if len(feature_names) == len(importances):
                    top_features = sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    
                    return {
                        'top_features': [
                            {'feature': feat, 'importance': float(imp)} 
                            for feat, imp in top_features
                        ]
                    }
            
            # For linear models
            elif hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
                feature_names = getattr(self.preprocessor, 'get_feature_names', lambda: [])()
                
                if len(feature_names) == len(coefficients):
                    top_features = sorted(
                        zip(feature_names, np.abs(coefficients)),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    
                    return {
                        'top_features': [
                            {'feature': feat, 'coefficient_magnitude': float(mag)} 
                            for feat, mag in top_features
                        ]
                    }
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return None
    
    def _cache_predictions(self, predictions: List[Dict]):
        """Cache predictions for monitoring and drift detection."""
        
        self.prediction_cache.extend(predictions)
        
        # Trim cache if it exceeds size limit
        if len(self.prediction_cache) > self.cache_size:
            self.prediction_cache = self.prediction_cache[-self.cache_size:]
    
    def get_prediction_stats(self) -> Dict:
        """Get statistics from cached predictions."""
        
        if not self.prediction_cache:
            return {}
        
        probs = [p['probability_default'] for p in self.prediction_cache]
        
        return {
            'total_predictions': len(self.prediction_cache),
            'mean_probability': float(np.mean(probs)),
            'std_probability': float(np.std(probs)),
            'approval_rate': float(np.mean([p['prediction'] == 0 for p in self.prediction_cache])),
            'decline_rate': float(np.mean([p['prediction'] == 1 for p in self.prediction_cache])),
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_predict(self, 
                     filepath: str,
                     output_path: Optional[str] = None,
                     calibrated: bool = True) -> pd.DataFrame:
        """Batch prediction from file."""
        
        logger.info(f"Starting batch prediction from {filepath}")
        
        # Load data
        data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(data)} records for batch prediction")
        
        # Get predictions
        predictions = self.predict_with_confidence(data, calibrated=calibrated)
        
        # Create results dataframe
        results = pd.DataFrame(predictions)
        
        # Add original data
        for col in data.columns:
            if col not in results.columns:
                results[col] = data[col].values
        
        # Save results if output path provided
        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Saved batch predictions to {output_path}")
        
        return results
    
    def evaluate_threshold(self, 
                          data: pd.DataFrame,
                          true_labels: pd.Series,
                          thresholds: List[float] = None) -> pd.DataFrame:
        """Evaluate model performance at different thresholds."""
        
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = []
        probabilities = self.predict_proba(data, calibrated=False)
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import (accuracy_score, precision_score, 
                                        recall_score, f1_score, roc_auc_score)
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f1': f1_score(true_labels, predictions, zero_division=0),
                'default_rate': predictions.mean(),
                'true_default_rate': true_labels.mean()
            }
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def save_predictions(self, 
                        predictions: Union[List[Dict], pd.DataFrame],
                        filepath: str,
                        format: str = 'csv'):
        """Save predictions to file."""
        
        if isinstance(predictions, list):
            predictions = pd.DataFrame(predictions)
        
        if format.lower() == 'csv':
            predictions.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            predictions.to_json(filepath, orient='records', indent=2)
        elif format.lower() == 'parquet':
            predictions.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved predictions to {filepath}")


# Factory function for easy predictor creation
def create_predictor(config: Dict[str, Any] = None) -> CreditRiskPredictor:
    """Create predictor with configuration."""
    
    if config is None:
        config = {
            'model_path': 'models/best_model.pkl',
            'preprocessor_path': 'models/preprocessor.pkl',
            'threshold': 0.5,
            'calibration_data_path': 'data/processed/calibration_data.csv'
        }
    
    return CreditRiskPredictor(**config)


def main():
    """Example usage of the predictor."""
    
    # Create predictor
    predictor = create_predictor()
    
    # Example single prediction
    sample_data = {
        'age': 35,
        'income': 50000,
        'debt': 10000,
        'credit_score': 720,
        'employment_length': 5,
        'savings': 20000
    }
    
    print("Single prediction example:")
    prediction = predictor.predict_with_confidence(sample_data)
    print(json.dumps(prediction[0], indent=2))
    
    # Batch prediction example
    print("\nBatch prediction example:")
    try:
        results = predictor.batch_predict(
            filepath='data/processed/test_data.csv',
            output_path='data/predictions/batch_predictions.csv'
        )
        print(f"Batch predictions completed: {len(results)} records")
        
        # Get statistics
        stats = predictor.get_prediction_stats()
        print(f"\nPrediction statistics: {stats}")
        
    except FileNotFoundError:
        print("Test data file not found, skipping batch prediction")
    
    return predictor


if __name__ == "__main__":
    predictor = main()