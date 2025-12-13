"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.data_processing import DataProcessor

class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'TransactionId': [f'TXN{i:06d}' for i in range(n_samples)],
            'CustomerId': [f'CUST{i:04d}' for i in np.random.randint(1, 20, n_samples)],
            'Amount': np.random.normal(100, 50, n_samples).round(2),
            'Value': np.abs(np.random.normal(100, 50, n_samples)).round(2),
            'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food'], n_samples),
            'ChannelId': np.random.choice(['Web', 'Android', 'iOS'], n_samples),
            'ProviderId': np.random.choice(['P1', 'P2', 'P3'], n_samples),
            'CountryCode': np.random.choice([254, 255, 256], n_samples),
            'CurrencyCode': np.random.choice(['USD', 'EUR', 'GBP'], n_samples),
            'TransactionStartTime': pd.date_range('2023-01-01', periods=n_samples, freq='H').strftime('%Y-%m-%d %H:%M:%S'),
            'FraudResult': np.random.binomial(1, 0.1, n_samples)
        }
        
        # Add some missing values
        data['Amount'][:5] = np.nan
        data['ProductCategory'][:3] = None
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance."""
        return DataProcessor()
    
    def test_load_data(self, processor, tmp_path):
        """Test data loading functionality."""
        # Create a test CSV file
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        test_file = tmp_path / 'test.csv'
        test_data.to_csv(test_file, index=False)
        
        # Test loading
        loaded_data = processor.load_data(str(test_file))
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == ['col1', 'col2']
    
    def test_create_proxy_target(self, processor, sample_data):
        """Test proxy target creation."""
        df_with_target = processor.create_proxy_target(sample_data)
        
        # Check that new columns are added
        assert 'proxy_risk_score' in df_with_target.columns
        assert 'high_risk' in df_with_target.columns
        
        # Check that high_risk is binary
        assert set(df_with_target['high_risk'].unique()).issubset({0, 1})
        
        # Check that fraud transactions get higher risk scores
        fraud_indices = df_with_target[df_with_target['FraudResult'] == 1].index
        if len(fraud_indices) > 0:
            assert all(df_with_target.loc[fraud_indices, 'proxy_risk_score'] >= 3)
    
    def test_engineer_features(self, processor, sample_data):
        """Test feature engineering."""
        df_with_features = processor.engineer_features(sample_data)
        
        # Check that time-based features are created
        if 'TransactionStartTime' in sample_data.columns:
            assert 'transaction_hour' in df_with_features.columns
            assert 'transaction_dayofweek' in df_with_features.columns
            assert 'time_of_day' in df_with_features.columns
        
        # Check that customer features are created
        if 'CustomerId' in sample_data.columns:
            assert 'transaction_count' in df_with_features.columns
            assert 'Amount_mean' in df_with_features.columns
    
    def test_preprocess_features(self, processor, sample_data):
        """Test feature preprocessing."""
        # Create proxy target first
        df_with_target = processor.create_proxy_target(sample_data)
        df_with_features = processor.engineer_features(df_with_target)
        
        # Preprocess features
        X = df_with_features.drop(columns=['high_risk', 'proxy_risk_score'], errors='ignore')
        X_processed = processor.preprocess_features(X, fit=True)
        
        # Check that missing values are handled
        assert not X_processed.isnull().any().any()
        
        # Check that features are scaled (mean ~0, std ~1 for numerical)
        numerical_cols = processor.config['features']['numerical']
        available_numerical = [col for col in numerical_cols if col in X_processed.columns]
        
        for col in available_numerical:
            # Skip if column was encoded
            if X_processed[col].nunique() > 10:  # Likely numerical
                assert abs(X_processed[col].mean()) < 1e-10  # Approximately 0
                assert abs(X_processed[col].std() - 1) < 1e-10  # Approximately 1
    
    def test_prepare_training_data(self, processor, sample_data):
        """Test training data preparation."""
        X_train, X_test, y_train, y_test = processor.prepare_training_data(sample_data, test_size=0.3)
        
        # Check shapes
        total_samples = len(sample_data)
        expected_train = int(total_samples * 0.7)
        expected_test = total_samples - expected_train
        
        assert len(X_train) == expected_train
        assert len(X_test) == expected_test
        assert len(y_train) == expected_train
        assert len(y_test) == expected_test
        
        # Check that features are preprocessed
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()
        
        # Check that target exists
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
    
    def test_save_load_preprocessor(self, processor, sample_data, tmp_path):
        """Test preprocessor saving and loading."""
        # Process some data to fit the preprocessor
        X_train, _, _, _ = processor.prepare_training_data(sample_data, test_size=0.2)
        
        # Save preprocessor
        save_path = tmp_path / 'preprocessor.pkl'
        processor.save_preprocessor(str(save_path))
        
        # Create new processor and load
        new_processor = DataProcessor()
        new_processor.load_preprocessor(str(save_path))
        
        # Check that loaded objects exist
        assert new_processor.scaler is not None
        assert new_processor.imputer is not None
        assert new_processor.feature_names is not None
        
        # Test that loaded preprocessor can transform new data
        test_data = sample_data.copy()
        test_data = new_processor.create_proxy_target(test_data)
        test_data = new_processor.engineer_features(test_data)
        X_test = test_data.drop(columns=['high_risk', 'proxy_risk_score'], errors='ignore')
        
        X_processed = new_processor.preprocess_features(X_test, fit=False)
        assert not X_processed.isnull().any().any()
        assert X_processed.shape[1] == len(new_processor.feature_names)
    
    def test_config_loading(self, tmp_path):
        """Test configuration loading."""
        # Create a test config file
        config = {
            'features': {
                'numerical': ['Amount', 'Value'],
                'categorical': ['ProductCategory']
            },
            'preprocessing': {
                'outlier_capping': False,
                'scale_features': False
            }
        }
        
        import json
        config_file = tmp_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Create processor with config
        processor = DataProcessor(config_path=str(config_file))
        
        # Check that config was loaded
        assert not processor.config['preprocessing']['outlier_capping']
        assert not processor.config['preprocessing']['scale_features']
        assert processor.config['features']['numerical'] == ['Amount', 'Value']
    
    def test_missing_value_handling(self, processor):
        """Test missing value imputation."""
        # Create data with missing values
        data = pd.DataFrame({
            'Amount': [100, np.nan, 200, np.nan, 300],
            'Value': [100, 150, np.nan, 200, 250],
            'ProductCategory': ['A', 'B', 'A', None, 'B']
        })
        
        # Add to config
        processor.config['features']['numerical'] = ['Amount', 'Value']
        processor.config['features']['categorical'] = ['ProductCategory']
        
        # Preprocess
        X_processed = processor.preprocess_features(data, fit=True)
        
        # Check no missing values
        assert not X_processed.isnull().any().any()
        
        # Check categorical encoding
        if processor.config['preprocessing']['encode_categorical']:
            # Should have encoded columns for categories
            assert any('ProductCategory' in col for col in X_processed.columns)
    
    def test_outlier_capping(self, processor):
        """Test outlier capping functionality."""
        # Create data with outliers
        data = pd.DataFrame({
            'Amount': [100, 200, 300, 400, 1000],  # 1000 is an outlier
            'Value': [50, 60, 70, 80, 90]
        })
        
        processor.config['features']['numerical'] = ['Amount', 'Value']
        processor.config['preprocessing']['outlier_capping'] = True
        processor.config['preprocessing']['cap_percentile'] = 90
        
        # Preprocess
        X_processed = processor.preprocess_features(data, fit=True)
        
        # Check that outlier was capped
        cap_value = data['Amount'].quantile(0.9)
        assert X_processed['Amount'].max() <= cap_value
    
    def test_feature_names_preservation(self, processor, sample_data):
        """Test that feature names are preserved after preprocessing."""
        X_train, _, _, _ = processor.prepare_training_data(sample_data, test_size=0.2)
        
        # Check that processor stores feature names
        assert processor.feature_names is not None
        assert len(processor.feature_names) == X_train.shape[1]
        
        # Check that feature names match columns
        assert all(col in processor.feature_names for col in X_train.columns)
        assert all(col in X_train.columns for col in processor.feature_names)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])