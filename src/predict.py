import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPredictor:
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        self.model = None
        self.preprocessor = None
        self.model_info = {}
        self.feature_names = None
        self.id_columns = ['TransactionId', 'BatchId', 'AccountId',
                           'SubscriptionId', 'CustomerId', 'ProductId']
        self.target_column = 'FraudResult'

        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)

        logger.info("FraudDetectionPredictor initialized")

    def load_model(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_info = {
            'model_type': model_data['model_type'],
            'best_params': model_data.get('best_params', {}),
            'metrics': model_data.get('metrics', {}),
            'threshold': model_data.get('threshold', 0.5),
            'feature_names': model_data.get('feature_names')
        }

        # Feature names from model or preprocessor
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        elif self.model_info.get('feature_names'):
            self.feature_names = self.model_info['feature_names']

        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Model type: {self.model_info['model_type']}")
        logger.info(f"Threshold: {self.model_info['threshold']:.3f}")

    def load_preprocessor(self, preprocessor_path: str):
        if not Path(preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        logger.info(f"Preprocessor loaded: {preprocessor_path}")

    def _clean_input(self, df: pd.DataFrame):
        """Remove IDs and target column before preprocessing."""
        df_clean = df.drop(columns=[col for col in self.id_columns + [self.target_column]
                                    if col in df.columns], errors='ignore')
        return df_clean

    def _align_features(self, df: pd.DataFrame):
        """Align input features to match training feature names."""
        df = df.copy()

        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Add missing features as 0

        extra_features = [col for col in df.columns if col not in self.feature_names]
        if extra_features:
            df = df.drop(columns=extra_features)
            logger.debug(f"Removed extra features: {extra_features}")

        df = df[self.feature_names]  # Ensure correct order
        return df

    def preprocess_input(self, df: pd.DataFrame):
        """Preprocess input DataFrame using preprocessor if available."""
        df_clean = self._clean_input(df)

        if self.preprocessor is not None:
            config = self.preprocessor.get('config', {})
            numerical_features = config.get('features', {}).get('numerical', [])
            categorical_features = config.get('features', {}).get('categorical', [])

            numerical_features = [f for f in numerical_features if f in df_clean.columns]
            categorical_features = [f for f in categorical_features if f in df_clean.columns]

            # Numeric imputer
            imputer = self.preprocessor.get('imputer')
            if imputer and numerical_features:
                df_clean[numerical_features] = imputer.transform(df_clean[numerical_features])

            # Scale numeric
            scaler = self.preprocessor.get('scaler')
            if scaler and numerical_features:
                df_clean[numerical_features] = scaler.transform(df_clean[numerical_features])

            # Encode categorical
            encoder = self.preprocessor.get('encoder')
            if encoder and categorical_features:
                encoded_array = encoder.transform(df_clean[categorical_features])
                encoded_df = pd.DataFrame(encoded_array,
                                          columns=encoder.get_feature_names_out(categorical_features),
                                          index=df_clean.index)
                df_clean = df_clean.drop(columns=categorical_features)
                df_clean = pd.concat([df_clean, encoded_df], axis=1)

        df_aligned = self._align_features(df_clean)
        return df_aligned

    def predict(self, df: pd.DataFrame, return_proba=False):
        df_processed = self.preprocess_input(df)
        if return_proba:
            return self.model.predict_proba(df_processed)[:, 1]
        threshold = self.model_info.get('threshold', 0.5)
        proba = self.model.predict_proba(df_processed)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_with_metadata(self, df: pd.DataFrame):
        ids_df = df[[col for col in self.id_columns if col in df.columns]]
        probabilities = self.predict(df, return_proba=True)
        threshold = self.model_info.get('threshold', 0.5)
        predictions = (probabilities >= threshold).astype(int)

        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': pd.cut(probabilities, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        })

        if not ids_df.empty:
            results = pd.concat([ids_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
        return results

    def predict_batch(self, input_csv, output_csv=None):
        df = pd.read_csv(input_csv)
        results = self.predict_with_metadata(df)
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_csv, index=False)
            logger.info(f"Predictions saved to {output_csv}")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fraud detection prediction")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='../predictions/predictions.csv')
    parser.add_argument('--model', default=r'C:\Users\admin\credit-risk-model\models\gradient_boosting_model.pkl')
    parser.add_argument('--preprocessor', default=r'C:\Users\admin\credit-risk-model\data\processed\preprocessor.pkl')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    predictor = FraudDetectionPredictor(args.model, args.preprocessor)
    if args.threshold is not None:
        predictor.model_info['threshold'] = args.threshold

    results = predictor.predict_batch(args.input, args.output)
    print(results.head(10))


if __name__ == '__main__':
    main()
