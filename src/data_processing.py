"""
Data processing pipeline for credit risk modeling.
Implements feature engineering, missing value imputation, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import category_encoders as ce
import joblib
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """Main preprocessing pipeline for credit risk data."""
    
    def __init__(self, 
                 numerical_features: List[str],
                 categorical_features: List[str],
                 target_col: str = 'default_flag',
                 use_woe: bool = True,
                 woe_params: Dict[str, Any] = None):
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.use_woe = use_woe
        self.woe_params = woe_params or {}
        
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Create the complete preprocessing pipeline."""
        
        # Numerical preprocessing
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
        
        # Optional WOE encoding
        if self.use_woe:
            self.woe_encoder = ce.WOEEncoder(**self.woe_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the preprocessing pipeline."""
        logger.info("Fitting preprocessing pipeline...")
        
        if self.use_woe and y is not None:
            # Fit WOE encoder on categorical features
            X_cat = X[self.categorical_features]
            self.woe_encoder.fit(X_cat, y)
            logger.info("WOE encoder fitted successfully")
        
        # Fit the main preprocessor
        self.preprocessor.fit(X)
        logger.info("Preprocessing pipeline fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data."""
        logger.info("Transforming data...")
        
        # Apply WOE encoding if enabled
        if self.use_woe and hasattr(self, 'woe_encoder'):
            X_transformed = X.copy()
            X_cat = X[self.categorical_features]
            woe_encoded = self.woe_encoder.transform(X_cat)
            
            # Replace categorical columns with WOE encoded values
            for col in self.categorical_features:
                X_transformed[col] = woe_encoded[col]
            
            # Now apply standard preprocessing
            result = self.preprocessor.transform(X_transformed)
        else:
            result = self.preprocessor.transform(X)
        
        logger.info(f"Transformed data shape: {result.shape}")
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        # Categorical features (one-hot encoded)
        if hasattr(self.preprocessor.named_transformers_['cat'], 'named_steps'):
            onehot = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = onehot.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_features)
        
        return feature_names


class FeatureEngineer:
    """Feature engineering for credit risk modeling."""
    
    @staticmethod
    def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features."""
        df_engineered = df.copy()
        
        # Common financial ratios
        if all(col in df.columns for col in ['income', 'debt']):
            df_engineered['debt_to_income'] = df['debt'] / (df['income'] + 1e-6)
        
        if all(col in df.columns for col in ['credit_limit', 'current_balance']):
            df_engineered['credit_utilization'] = df['current_balance'] / (df['credit_limit'] + 1e-6)
        
        if all(col in df.columns for col in ['savings', 'income']):
            df_engineered['savings_ratio'] = df['savings'] / (df['income'] + 1e-6)
        
        return df_engineered
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df_engineered = df.copy()
        
        # Age-income interaction
        if all(col in df.columns for col in ['age', 'income']):
            df_engineered['age_income_interaction'] = df['age'] * df['income']
        
        # Employment-income interaction
        if all(col in df.columns for col in ['employment_length', 'income']):
            df_engineered['emp_income_interaction'] = df['employment_length'] * df['income']
        
        return df_engineered
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create temporal features from date column."""
        df_engineered = df.copy()
        
        if date_col in df.columns:
            df_engineered[date_col] = pd.to_datetime(df[date_col])
            
            # Extract date components
            df_engineered[f'{date_col}_year'] = df_engineered[date_col].dt.year
            df_engineered[f'{date_col}_month'] = df_engineered[date_col].dt.month
            df_engineered[f'{date_col}_quarter'] = df_engineered[date_col].dt.quarter
            df_engineered[f'{date_col}_dayofweek'] = df_engineered[date_col].dt.dayofweek
            df_engineered[f'{date_col}_is_weekend'] = df_engineered[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Time since reference date
            reference_date = pd.Timestamp('2020-01-01')
            df_engineered[f'{date_col}_days_since_ref'] = (df_engineered[date_col] - reference_date).dt.days
        
        return df_engineered
    
    @staticmethod
    def create_aggregation_features(df: pd.DataFrame, 
                                   group_cols: List[str], 
                                   agg_cols: List[str]) -> pd.DataFrame:
        """Create aggregated features."""
        df_engineered = df.copy()
        
        for group_col in group_cols:
            if group_col in df.columns:
                for agg_col in agg_cols:
                    if agg_col in df.columns:
                        # Calculate group statistics
                        group_stats = df.groupby(group_col)[agg_col].agg([
                            'mean', 'std', 'min', 'max', 'median'
                        ]).add_prefix(f'{agg_col}_per_{group_col}_')
                        
                        # Merge back to original dataframe
                        df_engineered = df_engineered.merge(
                            group_stats, 
                            how='left', 
                            left_on=group_col, 
                            right_index=True
                        )
        
        return df_engineered


def create_proxy_target(df: pd.DataFrame, 
                        delinquency_col: str,
                        threshold_days: int = 90) -> pd.Series:
    """
    Create proxy default target based on delinquency.
    
    Args:
        df: Input dataframe
        delinquency_col: Column containing days delinquent
        threshold_days: Number of days delinquent to consider as default proxy
    
    Returns:
        Proxy target series (1 = default, 0 = non-default)
    """
    logger.info(f"Creating proxy target with {threshold_days} days threshold")
    
    if delinquency_col not in df.columns:
        raise ValueError(f"Delinquency column '{delinquency_col}' not found in dataframe")
    
    # Create binary target based on delinquency threshold
    proxy_target = (df[delinquency_col] >= threshold_days).astype(int)
    
    default_rate = proxy_target.mean() * 100
    logger.info(f"Proxy default rate: {default_rate:.2f}%")
    logger.info(f"Class distribution: {proxy_target.value_counts().to_dict()}")
    
    return proxy_target


def load_and_preprocess_data(filepath: str, 
                            config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, CreditRiskPreprocessor]:
    """
    Load and preprocess data with comprehensive pipeline.
    
    Args:
        filepath: Path to data file
        config: Configuration dictionary
    
    Returns:
        Processed features, target, and fitted preprocessor
    """
    logger.info(f"Loading data from {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Create proxy target
    target = create_proxy_target(
        df, 
        config['delinquency_col'],
        config.get('delinquency_threshold', 90)
    )
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    
    # Create ratio features
    df = feature_engineer.create_ratio_features(df)
    
    # Create interaction features
    df = feature_engineer.create_interaction_features(df)
    
    # Create temporal features if date column exists
    if 'date_col' in config and config['date_col'] in df.columns:
        df = feature_engineer.create_temporal_features(df, config['date_col'])
    
    # Identify feature types
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from features
    numerical_features = [col for col in numerical_features if col != config['delinquency_col']]
    
    logger.info(f"Numerical features: {len(numerical_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Initialize and fit preprocessor
    preprocessor = CreditRiskPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_col='default_flag',
        use_woe=config.get('use_woe', True),
        woe_params=config.get('woe_params', {})
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(df, target)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    logger.info(f"Final processed data shape: {X_processed_df.shape}")
    
    return X_processed_df, target, preprocessor


def save_pipeline(preprocessor: CreditRiskPreprocessor, 
                  filepath: str = 'models/preprocessor.pkl'):
    """Save preprocessing pipeline."""
    logger.info(f"Saving pipeline to {filepath}")
    joblib.dump(preprocessor, filepath)


def load_pipeline(filepath: str = 'models/preprocessor.pkl') -> CreditRiskPreprocessor:
    """Load preprocessing pipeline."""
    logger.info(f"Loading pipeline from {filepath}")
    return joblib.load(filepath)


if __name__ == "__main__":
    # Example configuration
    config = {
        'delinquency_col': 'days_delinquent',
        'delinquency_threshold': 90,
        'date_col': 'application_date',
        'use_woe': True,
        'woe_params': {'random_state': 42}
    }
    
    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess_data(
        'data/raw/credit_data.csv',
        config
    )
    
    # Save pipeline
    save_pipeline(preprocessor)