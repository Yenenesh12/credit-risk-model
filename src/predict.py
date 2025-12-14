"""
Prediction module for credit risk scoring.
Loads trained models and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
from typing import Dict, List, Tuple, Union, Optional
import warnings
import os
warnings.filterwarnings('ignore')

import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score, recall_score)


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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.model_info = {
            'model_type': model_data['model_type'],
            'best_params': model_data['best_params'],
            'metrics': model_data.get('metrics', {}),
            'threshold': model_data.get('threshold', 0.5),
            'timestamp': model_data.get('timestamp', 'Unknown')
        }

        print(f"✅ Model loaded from {model_path}")
        print(f"Model type: {self.model_info['model_type']}")
        print(f"Threshold: {self.model_info['threshold']:.3f}")
        if 'metrics' in model_data and 'roc_auc' in model_data['metrics']:
            print(f"ROC-AUC: {model_data['metrics']['roc_auc']:.4f}")

    def load_preprocessor(self, preprocessor_path: str) -> None:
        """Load preprocessor from file."""
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        feature_names = self.preprocessor.get('feature_names')
        if feature_names is None:
            raise ValueError("Preprocessor missing 'feature_names'. Ensure it was saved during training.")
        print(f"✅ Preprocessor loaded from {preprocessor_path} (expects {len(feature_names)} features)")

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.

        Args:
            df: Raw input DataFrame

        Returns:
            Preprocessed DataFrame with EXACTLY the same features as during training
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be loaded before preprocessing")

        df = df.copy()

        # Get configuration
        config = self.preprocessor.get('config', {})
        numerical_features = config.get('features', {}).get('numerical', [])
        categorical_features = config.get('features', {}).get('categorical', [])
        feature_names = self.preprocessor.get('feature_names')
        preprocessing_config = config.get('preprocessing', {})

        if feature_names is None:
            raise ValueError("Preprocessor missing 'feature_names'")

        # Filter features to those present in input
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        # 1. Handle missing values for numerical features
        imputer = self.preprocessor.get('imputer')
        if imputer and numerical_features:
            df[numerical_features] = imputer.transform(df[numerical_features])

        # 2. Handle outliers (winsorizing)
        if preprocessing_config.get('outlier_capping', True):
            cap_percentile = preprocessing_config.get('cap_percentile', 99)
            for col in numerical_features:
                cap_value = df[col].quantile(cap_percentile / 100)
                df[col] = np.where(df[col] > cap_value, cap_value, df[col])

        # 3. Scale numerical features
        scaler = self.preprocessor.get('scaler')
        if scaler and preprocessing_config.get('scale_features', True) and numerical_features:
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
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        # 🔥 CRITICAL FIX: Enforce EXACT feature set matching training
        expected_features = set(feature_names)
        current_features = set(df.columns)

        # Drop columns NOT in training features
        extra_cols = current_features - expected_features
        if extra_cols:
            df = df.drop(columns=list(extra_cols))
            print(f"⚠️ Dropped {len(extra_cols)} unexpected column(s): {sorted(extra_cols)[:5]}...")

        # Add missing columns (as zeros)
        missing_cols = expected_features - current_features
        for col in missing_cols:
            df[col] = 0.0

        # Reorder to match training
        df = df[feature_names]

        print(f"✅ Preprocessing complete. Output shape: {df.shape}")
        return df

    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")

        if self.preprocessor is not None:
            df_processed = self.preprocess_input(df)
        else:
            df_processed = df

        if return_proba:
            return self.model.predict_proba(df_processed)[:, 1]
        else:
            threshold = self.model_info.get('threshold', 0.5)
            proba = self.model.predict_proba(df_processed)[:, 1]
            return (proba >= threshold).astype(int)

    def predict_with_explanations(self, df: pd.DataFrame, top_features: int = 10) -> pd.DataFrame:
        """Make predictions with SHAP explanations."""
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")

        if self.preprocessor is not None:
            df_processed = self.preprocess_input(df)
        else:
            df_processed = df

        threshold = self.model_info.get('threshold', 0.5)
        proba = self.model.predict_proba(df_processed)[:, 1]
        predictions = (proba >= threshold).astype(int)

        results = pd.DataFrame({
            'prediction': predictions,
            'probability': proba,
            'risk_level': pd.cut(proba, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        })

        # SHAP explanations (only for first 10 rows to avoid slowdown)
        try:
            if hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(df_processed)
                shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            else:
                explainer = shap.LinearExplainer(self.model, df_processed[:100])
                shap_vals = explainer.shap_values(df_processed)

            for i in range(min(10, len(df))):
                contrib = pd.DataFrame({
                    'feature': df_processed.columns,
                    'contribution': shap_vals[i] if len(shap_vals.shape) > 1 else shap_vals
                }).sort_values('contribution', key=abs, ascending=False).head(top_features)

                results.at[i, 'top_contributing_features'] = json.dumps(
                    contrib.set_index('feature')['contribution'].to_dict()
                )
        except Exception as e:
            print(f"⚠️ SHAP explanation skipped: {e}")
            results['top_contributing_features'] = None

        return results

    def predict_batch(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """Predict on a batch of data from file."""
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"📊 Input shape: {df.shape}")

        print(f"🧠 Making predictions on {len(df)} samples...")
        results = self.predict_with_explanations(df)

        output_df = pd.concat([df, results], axis=1)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to {output_path}")

        print(f"\n✅ Prediction Summary:")
        print(f"Total samples: {len(output_df)}")
        fraud_rate = output_df['prediction'].mean()
        print(f"Predicted fraud: {output_df['prediction'].sum()} ({fraud_rate:.2%})")
        print(f"Risk distribution:\n{output_df['risk_level'].value_counts().sort_index()}")

        return output_df

    def evaluate_custom_threshold(self, df: pd.DataFrame, true_labels: pd.Series, threshold: float) -> Dict:
        """Evaluate predictions with custom threshold."""
        proba = self.predict(df, return_proba=True)
        predictions = (proba >= threshold).astype(int)

        metrics = {
            'threshold': threshold,
            'roc_auc': roc_auc_score(true_labels, proba),
            'f1_score': f1_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'accuracy': (predictions == true_labels).mean()
        }

        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }

        print(f"\n🔍 Evaluation (threshold={threshold:.3f}):")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"{k}: {v:.4f}")
        print(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

        return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Make credit risk predictions')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model', type=str,
                        default=r'C:\Users\admin\credit-risk-model\models\gradient_boosting_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--preprocessor', type=str,
                        default=r'C:\Users\admin\credit-risk-model\data\processed\preprocessor.pkl',
                        help='Path to preprocessor')
    parser.add_argument('--output', type=str,
                        default=r'predictions/predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom classification threshold')

    args = parser.parse_args()

    # Initialize predictor
    predictor = CreditRiskPredictor(model_path=args.model, preprocessor_path=args.preprocessor)

    if args.threshold is not None:
        predictor.model_info['threshold'] = args.threshold
        print(f"🎯 Using custom threshold: {args.threshold}")

    # Run prediction
    results = predictor.predict_batch(args.input, args.output)

    # Show sample
    print("\n📋 Sample Predictions:")
    sample_cols = ['prediction', 'probability', 'risk_level']
    for col in ['TransactionId', 'CustomerId']:
        if col in results.columns:
            sample_cols.insert(0, col)
    display_cols = [c for c in sample_cols if c in results.columns]
    print(results[display_cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()