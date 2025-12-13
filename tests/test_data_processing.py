"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import (
    CreditRiskPreprocessor,
    FeatureEngineer,
    create_proxy_target,
    load_and_preprocess_data
)


class TestFeatureEngineer:
    """Test feature engineering functions."""
    
    def setup_method(self):
        """Setup test data."""
        self.df = pd.DataFrame({
            'income': [50000, 60000, 70000],
            'debt': [10000, 15000, 20000],
            'savings': [20000, 25000, 30000],
            'credit_limit': [10000, 15000, 20000],
            'current_balance': [3000, 7500, 12000],
            'age': [30, 40, 50],
            'employment_length': [2, 5, 10]
        })
    
    def test_create_ratio_features(self):
        """Test ratio feature creation."""
        df_engineered = FeatureEngineer.create_ratio_features(self.df)
        
        # Check that new columns are created
        assert 'debt_to_income' in df_engineered.columns
        assert 'credit_utilization' in df_engineered.columns
        assert 'savings_ratio' in df_engineered.columns
        
        # Check calculations
        expected_dti = 10000 / 50000
        assert df_engineered.loc[0, 'debt_to_income'] == pytest.approx(expected_dti)
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        df_engineered = FeatureEngineer.create_interaction_features(self.df)
        
        assert 'age_income_interaction' in df_engineered.columns
        assert 'emp_income_interaction' in df_engineered.columns
        
        # Check calculations
        expected_interaction = 30 * 50000
        assert df_engineered.loc[0, 'age_income_interaction'] == expected_interaction
    
    def test_create_temporal_features(self):
        """Test temporal feature creation."""
        df = pd.DataFrame({
            'application_date': ['2023-01-15', '2023-02-20', '2023-03-25']
        })
        
        df_engineered = FeatureEngineer.create_temporal_features(df, 'application_date')
        
        assert 'application_date_year' in df_engineered.columns
        assert 'application_date_month' in df_engineered.columns
        assert 'application_date_quarter' in df_engineered.columns
        assert 'application_date_dayofweek' in df_engineered.columns
        assert 'application_date_is_weekend' in df_engineered.columns
        assert 'application_date_days_since_ref' in df_engineered.columns
        
        # Check data types
        assert df_engineered['application_date_is_weekend'].dtype == np.int64


class TestProxyTarget:
    """Test proxy target creation."""
    
    def test_create_proxy_target(self):
        """Test proxy target creation with threshold."""
        df = pd.DataFrame({
            'days_delinquent': [0, 30, 60, 90, 120]
        })
        
        # Test with 90 day threshold
        target = create_proxy_target(df, 'days_delinquent', threshold_days=90)
        
        expected = pd.Series([0, 0, 0, 1, 1])
        pd.testing.assert_series_equal(target, expected, check_dtype=False)
        
        # Test default rate calculation
        default_rate = target.mean() * 100
        assert default_rate == 40.0  # 2 out of 5
    
    def test_missing_column(self):
        """Test error when delinquency column is missing."""
        df = pd.DataFrame({'other_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Delinquency column 'days_delinquent' not found"):
            create_proxy_target(df, 'days_delinquent')


class TestCreditRiskPreprocessor:
    """Test credit risk preprocessor."""
    
    def setup_method(self):
        """Setup test data."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        # Convert to DataFrame with mixed types
        self.X = pd.DataFrame(
            X,
            columns=[f'num_{i}' for i in range(8)] + [f'cat_{i}' for i in range(2)]
        )
        
        # Convert some columns to categorical
        for col in ['cat_0', 'cat_1']:
            self.X[col] = pd.cut(self.X[col], bins=3, labels=['A', 'B', 'C'])
        
        self.y = pd.Series(y)
        
        # Define feature lists
        self.numerical_features = [f'num_{i}' for i in range(8)]
        self.categorical_features = ['cat_0', 'cat_1']
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = CreditRiskPreprocessor(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            use_woe=True
        )
        
        assert preprocessor.numerical_features == self.numerical_features
        assert preprocessor.categorical_features == self.categorical_features
        assert preprocessor.use_woe == True
        assert hasattr(preprocessor, 'preprocessor')
    
    def test_fit_transform_without_woe(self):
        """Test fit and transform without WOE encoding."""
        preprocessor = CreditRiskPreprocessor(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            use_woe=False
        )
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(self.X, self.y)
        
        # Check shape
        # 8 numerical + 3*2 categorical (one-hot encoded) = 14 features
        assert X_transformed.shape[1] == 14
    
    def test_fit_transform_with_woe(self):
        """Test fit and transform with WOE encoding."""
        preprocessor = CreditRiskPreprocessor(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            use_woe=True
        )
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(self.X, self.y)
        
        # With WOE, categorical features are replaced, not one-hot encoded
        # So we should have 8 numerical + 2 WOE encoded = 10 features
        assert X_transformed.shape[1] == 10
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        # Add missing values
        X_missing = self.X.copy()
        X_missing.loc[0, 'num_0'] = np.nan
        X_missing.loc[1, 'cat_0'] = np.nan
        
        preprocessor = CreditRiskPreprocessor(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            use_woe=False
        )
        
        # Should handle missing values without error
        X_transformed = preprocessor.fit_transform(X_missing, self.y)
        assert not np.any(np.isnan(X_transformed))
    
    def test_get_feature_names(self):
        """Test getting feature names after transformation."""
        preprocessor = CreditRiskPreprocessor(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            use_woe=False
        )
        
        preprocessor.fit(self.X, self.y)
        feature_names = preprocessor.get_feature_names()
        
        # Should return list of strings
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_load_and_preprocess_data(self, tmp_path):
        """Test complete data loading and preprocessing pipeline."""
        # Create test data
        test_data = pd.DataFrame({
            'income': [50000, 60000, 70000, 80000],
            'debt': [10000, 15000, 20000, 25000],
            'days_delinquent': [0, 30, 90, 120],
            'savings': [20000, 25000, 30000, 35000],
            'credit_score': [650, 700, 720, 680],
            'employment_status': ['employed', 'self-employed', 'employed', 'unemployed']
        })
        
        # Save test data
        data_path = tmp_path / "test_data.csv"
        test_data.to_csv(data_path, index=False)
        
        # Configuration
        config = {
            'delinquency_col': 'days_delinquent',
            'delinquency_threshold': 90,
            'use_woe': False
        }
        
        # Load and preprocess
        X, y, preprocessor = load_and_preprocess_data(str(data_path), config)
        
        # Check outputs
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert hasattr(preprocessor, 'transform')
        
        # Check target creation
        assert y.sum() == 2  # 2 defaults (days_delinquent >= 90)
        
        # Check feature engineering
        assert 'debt_to_income' in X.columns
        assert 'savings_ratio' in X.columns
    
    def test_pipeline_save_load(self, tmp_path):
        """Test saving and loading the preprocessing pipeline."""
        # Create and fit preprocessor
        preprocessor = CreditRiskPreprocessor(
            numerical_features=['income', 'debt'],
            categorical_features=['employment_status'],
            use_woe=False
        )
        
        test_data = pd.DataFrame({
            'income': [50000, 60000, 70000],
            'debt': [10000, 15000, 20000],
            'employment_status': ['employed', 'self-employed', 'unemployed']
        })
        
        test_target = pd.Series([0, 1, 0])
        
        preprocessor.fit(test_data, test_target)
        
        # Save pipeline
        pipeline_path = tmp_path / "pipeline.pkl"
        import joblib
        joblib.dump(preprocessor, pipeline_path)
        
        # Load pipeline
        loaded_pipeline = joblib.load(pipeline_path)
        
        # Test transformation
        new_data = pd.DataFrame({
            'income': [55000],
            'debt': [12000],
            'employment_status': ['employed']
        })
        
        original_transformed = preprocessor.transform(new_data)
        loaded_transformed = loaded_pipeline.transform(new_data)
        
        # Should produce same results
        np.testing.assert_array_almost_equal(
            original_transformed, 
            loaded_transformed
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])