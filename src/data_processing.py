"""
Data processing module for credit risk modeling.
Handles feature engineering, preprocessing, and data preparation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Main data processor for credit risk modeling."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize data processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.feature_names = None
        self.scaler = None
        self.encoder = None
        self.imputer = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        default_config = {
            'data_paths': {
                'raw': r'C:\Users\admin\credit-risk-model\data\raw\data.csv',
                'processed':r'C:/Users/admin/credit-risk-model/data/processed/'
            },
            'features': {
                'numerical': ['Amount', 'Value', 'CountryCode'],
                'categorical': ['ProductCategory', 'ChannelId', 'ProviderId', 
                               'PricingStrategy', 'CurrencyCode'],
                'datetime': ['TransactionStartTime']
            },
            'preprocessing': {
                'outlier_capping': True,
                'cap_percentile': 99,
                'scale_features': True,
                'encode_categorical': True,
                'impute_strategy': 'median'
            },
            'feature_engineering': {
                'time_features': True,
                'customer_features': True,
                'transaction_features': True
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with default config
                default_config.update(user_config)
            except FileNotFoundError:
                print(f"Config file {config_path} not found. Using default configuration.")
        
        return default_config
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load raw transaction data.
        
        Args:
            filepath: Path to raw data file
            
        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            filepath = self.config['data_paths']['raw']
        
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def create_proxy_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proxy target variable for credit risk.
        
        Business logic:
        - High risk: Fraudulent transactions OR high-value transactions from new customers
        - Medium risk: Transactions with certain product categories
        - Low risk: Regular transactions from established customers
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with proxy target column
        """
        df = df.copy()
        
        # Initialize proxy target
        df['proxy_risk_score'] = 0
        
        # 1. Fraud transactions are high risk
        if 'FraudResult' in df.columns:
            df.loc[df['FraudResult'] == 1, 'proxy_risk_score'] += 3
        
        # 2. High value transactions (top 5%)
        if 'Value' in df.columns:
            high_value_threshold = df['Value'].quantile(0.95)
            df.loc[df['Value'] > high_value_threshold, 'proxy_risk_score'] += 2
        
        # 3. New customers (first 7 days)
        if 'TransactionStartTime' in df.columns and 'CustomerId' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            customer_first_transaction = df.groupby('CustomerId')['TransactionStartTime'].min()
            df['days_since_first_transaction'] = df.apply(
                lambda row: (row['TransactionStartTime'] - customer_first_transaction[row['CustomerId']]).days
                if row['CustomerId'] in customer_first_transaction else 0, axis=1
            )
            df.loc[df['days_since_first_transaction'] <= 7, 'proxy_risk_score'] += 1
        
        # 4. Certain product categories might be higher risk
        high_risk_categories = ['Electronics', 'Services']  # Example categories
        if 'ProductCategory' in df.columns:
            df.loc[df['ProductCategory'].isin(high_risk_categories), 'proxy_risk_score'] += 1
        
        # Create binary target (1 = high risk, 0 = low/medium risk)
        df['high_risk'] = (df['proxy_risk_score'] >= 3).astype(int)
        
        print(f"Proxy target created: {df['high_risk'].mean():.2%} high risk transactions")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for credit risk modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # 1. Time-based features
        if 'TransactionStartTime' in df.columns and self.config['feature_engineering']['time_features']:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            
            # Extract time components
            df['transaction_hour'] = df['TransactionStartTime'].dt.hour
            df['transaction_day'] = df['TransactionStartTime'].dt.day
            df['transaction_month'] = df['TransactionStartTime'].dt.month
            df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
            df['transaction_weekend'] = df['transaction_dayofweek'].isin([5, 6]).astype(int)
            
            # Time of day categories
            df['time_of_day'] = pd.cut(df['transaction_hour'], 
                                      bins=[0, 6, 12, 18, 24],
                                      labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                      include_lowest=True)
        
        # 2. Customer behavior features (requires customer-level aggregation)
        if 'CustomerId' in df.columns and self.config['feature_engineering']['customer_features']:
            customer_features = self._create_customer_features(df)
            df = df.merge(customer_features, on='CustomerId', how='left')
        
        # 3. Transaction pattern features
        if self.config['feature_engineering']['transaction_features']:
            # Transaction frequency (if time-based features exist)
            if 'TransactionStartTime' in df.columns and 'CustomerId' in df.columns:
                df = df.sort_values(['CustomerId', 'TransactionStartTime'])
                df['time_since_last_transaction'] = df.groupby('CustomerId')['TransactionStartTime'].diff().dt.total_seconds() / 3600
                df['time_since_last_transaction'].fillna(24, inplace=True)  
            
            # Transaction amount ratios
            if 'Amount' in df.columns and 'Value' in df.columns:
                df['amount_to_value_ratio'] = df['Amount'] / (df['Value'] + 1e-6)
        
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregated features."""
        customer_features = pd.DataFrame(index=df['CustomerId'].unique())
        customer_features.index.name = 'CustomerId'
        
        # Basic customer statistics
        customer_agg = df.groupby('CustomerId').agg({
            'TransactionId': 'count',
            'Amount': ['mean', 'std', 'sum', 'min', 'max'],
            'Value': ['mean', 'std', 'sum']
        }).fillna(0)
        
        # Flatten column names
        customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
        customer_agg.rename(columns={'TransactionId_count': 'transaction_count'}, inplace=True)
        
        # Merge with customer_features
        customer_features = customer_features.merge(customer_agg, 
                                                  left_index=True, 
                                                  right_index=True, 
                                                  how='left')
        
        # Additional customer metrics
        if 'TransactionStartTime' in df.columns:
            customer_dates = df.groupby('CustomerId')['TransactionStartTime'].agg(['min', 'max'])
            customer_dates['customer_tenure_days'] = (customer_dates['max'] - customer_dates['min']).dt.days
            customer_features = customer_features.merge(customer_dates[['customer_tenure_days']], 
                                                      left_index=True, 
                                                      right_index=True, 
                                                      how='left')
        
        if 'ProductCategory' in df.columns:
            # Number of unique product categories per customer
            unique_cats = df.groupby('CustomerId')['ProductCategory'].nunique()
            customer_features['unique_product_categories'] = unique_cats
        
        return customer_features.reset_index()
    
    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess features: handle missing values, outliers, scaling, encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)
            
        Returns:
            Preprocessed DataFrame
        """
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        df = df.copy()
        
        # Select features based on configuration
        numerical_features = self.config['features']['numerical']
        categorical_features = self.config['features']['categorical']
        
        # Filter to available features only
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # 1. Handle missing values
        if fit:
            self.imputer = SimpleImputer(strategy=self.config['preprocessing']['impute_strategy'])
            df[numerical_features] = self.imputer.fit_transform(df[numerical_features])
        else:
            if self.imputer is not None:
                df[numerical_features] = self.imputer.transform(df[numerical_features])
        
        # 2. Handle outliers (winsorizing)
        if self.config['preprocessing']['outlier_capping']:
            for col in numerical_features:
                cap_value = df[col].quantile(self.config['preprocessing']['cap_percentile'] / 100)
                df[col] = np.where(df[col] > cap_value, cap_value, df[col])
        
        # 3. Scale numerical features
        if self.config['preprocessing']['scale_features']:
            if fit:
                self.scaler = StandardScaler()
                df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            else:
                if self.scaler is not None:
                    df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        # 4. Encode categorical features
        if self.config['preprocessing']['encode_categorical'] and categorical_features:
            if fit:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_array = self.encoder.fit_transform(df[categorical_features])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.encoder.get_feature_names_out(categorical_features),
                    index=df.index
                )
            else:
                if self.encoder is not None:
                    encoded_array = self.encoder.transform(df[categorical_features])
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=self.encoder.get_feature_names_out(categorical_features),
                        index=df.index
                    )
            
            # Drop original categorical columns and add encoded ones
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """
        Prepare training and testing datasets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Create proxy target if not exists
        if 'high_risk' not in df.columns:
            df = self.create_proxy_target(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Separate features and target
        X = df.drop(columns=['high_risk', 'proxy_risk_score'], errors='ignore')
        y = df['high_risk']
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Class distribution - Train: {y_train.mean():.2%} high risk, "
              f"Test: {y_test.mean():.2%} high risk")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor objects."""
        preprocessor_dict = {
            'scaler': self.scaler,
            'encoder': self.encoder,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_dict, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor objects."""
        with open(filepath, 'rb') as f:
            preprocessor_dict = pickle.load(f)
        
        self.scaler = preprocessor_dict['scaler']
        self.encoder = preprocessor_dict['encoder']
        self.imputer = preprocessor_dict['imputer']
        self.feature_names = preprocessor_dict['feature_names']
        self.config = preprocessor_dict['config']
        
        print(f"Preprocessor loaded from {filepath}")


def main():
    """Main function for data processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process transaction data for credit risk modeling')
    parser.add_argument('--input', type=str, default= r'C:\Users\admin\credit-risk-model\data\raw\data.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=r'C:/Users/admin/credit-risk-model/data/processed/',
                       help='Path to output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DataProcessor(config_path=args.config)
    
    # Load data
    df = processor.load_data(args.input)
    
    # Process data
    X_train, X_test, y_train, y_test = processor.prepare_training_data(df)
    
    # Save processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(f'{args.output}/train_data.csv', index=False)
    test_data.to_csv(f'{args.output}/test_data.csv', index=False)
    
    # Save preprocessor
    processor.save_preprocessor(f'{args.output}/preprocessor.pkl')
    
    print("Data processing completed successfully!")


if __name__ == '__main__':
    main()