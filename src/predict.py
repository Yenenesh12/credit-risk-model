"""
Prediction module for credit risk scoring.
Loads trained models and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Union, Optional
import warnings
warnings.filterwarnings('ignore')
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                                   roc_auc_score, f1_score, precision_score,
                                   recall_score)
class CreditRiskPredictor:
    """Credit risk predictor for inference."""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to preprocessor
        """
        self.model = None
        self.preprocessor = None
        self.model_info = {}
        
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from file."""
        import joblib
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.model_info = {
            'model_type': model_data['model_type'],
            'best_params': model_data['best_params'],
            'metrics': model_data.get('metrics', {}),
            'threshold': model_data.get('threshold', 0.5),
            'timestamp': model_data.get('timestamp', 'Unknown')
        }
        
        print(f"Model loaded from {model_path}")
        print(f"Model type: {self.model_info['model_type']}")
        print(f"Threshold: {self.model_info['threshold']:.3f}")
        if 'metrics' in model_data and 'roc_auc' in model_data['metrics']:
            print(f"ROC-AUC: {model_data['metrics']['roc_auc']:.4f}")
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """Load preprocessor from file."""
        with open(preprocessor_path, 'rb') as f:
            preprocessor_dict = pickle.load(f)
        
        self.preprocessor = preprocessor_dict
        print(f"Preprocessor loaded from {preprocessor_path}")
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be loaded before preprocessing")
        
        df = df.copy()
        
      
        
        # Get feature lists from config
        config = self.preprocessor.get('config', {})
        numerical_features = config.get('features', {}).get('numerical', [])
        categorical_features = config.get('features', {}).get('categorical', [])
        
        # Filter to available features only
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # 1. Handle missing values
        imputer = self.preprocessor.get('imputer')
        if imputer:
            df[numerical_features] = imputer.transform(df[numerical_features])
        
        # 2. Handle outliers (winsorizing)
        preprocessing_config = config.get('preprocessing', {})
        if preprocessing_config.get('outlier_capping', True):
            cap_percentile = preprocessing_config.get('cap_percentile', 99)
            for col in numerical_features:
                if col in df.columns:
                    cap_value = df[col].quantile(cap_percentile / 100)
                    df[col] = np.where(df[col] > cap_value, cap_value, df[col])
        
        # 3. Scale numerical features
        scaler = self.preprocessor.get('scaler')
        if scaler and preprocessing_config.get('scale_features', True):
            df[numerical_features] = scaler.transform(df[numerical_features])
        
        # 4. Encode categorical features
        encoder = self.preprocessor.get('encoder')
        if encoder and preprocessing_config.get('encode_categorical', True) and categorical_features:
            encoded_array = encoder.transform(df[categorical_features])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(categorical_features),
                index=df.index
            )
            
            # Drop original categorical columns and add encoded ones
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)
        
        # Ensure columns match training data
        feature_names = self.preprocessor.get('feature_names')
        if feature_names:
            # Add missing columns with zeros
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[feature_names]
        
        return df
    
    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            df: Input DataFrame (raw or preprocessed)
            return_proba: Whether to return probability scores
            
        Returns:
            Predictions or probability scores
        """
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")
        
        # Check if data needs preprocessing
        if self.preprocessor is not None:
            df_processed = self.preprocess_input(df)
        else:
            df_processed = df
        
        # Make predictions
        if return_proba:
            predictions = self.model.predict_proba(df_processed)[:, 1]
        else:
            threshold = self.model_info.get('threshold', 0.5)
            proba = self.model.predict_proba(df_processed)[:, 1]
            predictions = (proba >= threshold).astype(int)
        
        return predictions
    
    def predict_with_explanations(self, df: pd.DataFrame, 
                                 top_features: int = 10) -> pd.DataFrame:
        """
        Make predictions with feature importance explanations.
        
        Args:
            df: Input DataFrame
            top_features: Number of top features to show
            
        Returns:
            DataFrame with predictions and explanations
        """
        
        
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")
        
        # Preprocess data
        if self.preprocessor is not None:
            df_processed = self.preprocess_input(df)
        else:
            df_processed = df
        
        # Get predictions
        threshold = self.model_info.get('threshold', 0.5)
        proba = self.model.predict_proba(df_processed)[:, 1]
        predictions = (proba >= threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': proba,
            'risk_level': pd.cut(proba, 
                               bins=[0, 0.3, 0.7, 1.0],
                               labels=['Low', 'Medium', 'High'])
        })
        
        # Add SHAP explanations if available
        try:
            # Initialize SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'feature_importances_') else shap.LinearExplainer(self.model, df_processed)
                shap_values = explainer.shap_values(df_processed)
                
                # Get top features for each prediction
                for i in range(min(10, len(df))):  # Limit to first 10 for performance
                    if hasattr(shap_values, '__len__') and len(shap_values) > 1:
                        # For classification models
                        shap_arr = shap_values[1][i] if len(shap_values) == 2 else shap_values[i]
                    else:
                        shap_arr = shap_values[i]
                    
                    # Get top contributing features
                    feature_contributions = pd.DataFrame({
                        'feature': df_processed.columns,
                        'contribution': shap_arr
                    }).sort_values('contribution', key=abs, ascending=False)
                    
                    top_pos = feature_contributions.head(top_features//2)
                    top_neg = feature_contributions.tail(top_features//2)
                    top_features_combined = pd.concat([top_pos, top_neg])
                    
                    results.at[i, 'top_contributing_features'] = json.dumps(
                        top_features_combined.set_index('feature')['contribution'].to_dict()
                    )
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            results['top_contributing_features'] = None
        
        return results
    
    def predict_batch(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """
        Predict on a batch of data from file.
        
        Args:
            filepath: Path to input CSV file
            output_path: Path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        print(f"Loading batch data from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Making predictions on {len(df)} samples...")
        results = self.predict_with_explanations(df)
        
        # Combine with original data
        output_df = pd.concat([df, results], axis=1)
        
        # Save predictions
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total samples: {len(output_df)}")
        print(f"High risk predictions: {output_df['prediction'].sum()} "
              f"({output_df['prediction'].mean():.2%})")
        print(f"Risk level distribution:")
        print(output_df['risk_level'].value_counts().sort_index())
        
        return output_df
    
    def evaluate_custom_threshold(self, df: pd.DataFrame, true_labels: pd.Series,
                                threshold: float) -> Dict:
        """
        Evaluate predictions with custom threshold.
        
        Args:
            df: Input features
            true_labels: True labels
            threshold: Custom threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions with custom threshold
        proba = self.predict(df, return_proba=True)
        predictions = (proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'roc_auc': roc_auc_score(true_labels, proba),
            'f1_score': f1_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'accuracy': (predictions == true_labels).mean()
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        
        print(f"\nEvaluation with threshold = {threshold:.3f}:")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
        
        return metrics


def main():
    """Main function for prediction."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Make credit risk predictions')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--model', type=str, 
                       default=r'C:\Users\admin\credit-risk-model\models\gradient_boosting_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--preprocessor', type=str,
                       default=r'C:\Users\admin\credit-risk-model\data\processed\preprocessor.pkl',
                       help='Path to preprocessor')
    parser.add_argument('--output', type=str,
                       default='../predictions/predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Custom threshold for classification')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize predictor
    predictor = CreditRiskPredictor(model_path=args.model, 
                                   preprocessor_path=args.preprocessor)
    
    # Load and predict
    df = pd.read_csv(args.input)
    
    if args.threshold is not None:
        # Update threshold if provided
        predictor.model_info['threshold'] = args.threshold
        print(f"Using custom threshold: {args.threshold}")
    
    # Make predictions
    results = predictor.predict_batch(args.input, args.output)
    
    print("\nPrediction completed successfully!")
    
    # Show sample predictions
    print("\nSample predictions:")
    sample_cols = ['prediction', 'probability', 'risk_level']
    if 'TransactionId' in df.columns:
        sample_cols.insert(0, 'TransactionId')
    if 'CustomerId' in df.columns:
        sample_cols.insert(1, 'CustomerId')
    
    display_cols = [col for col in sample_cols if col in results.columns]
    print(results[display_cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()