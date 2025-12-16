"""
Data Loading and Preprocessing Module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsuranceDataProcessor:
    """Process insurance data for EDA analysis"""

    def __init__(self, config_path: str):
        """
        Initialize data processor with configuration
        
        Args:
            config_path: Path to configuration file (e.g., "../config/config.yaml")
        """
        # Resolve the config path to an absolute path
        self.config_path = Path(config_path).resolve()
        
        # Project root is the parent of the 'config' directory
        if self.config_path.parent.name != "config":
            raise ValueError(f"Config file must be inside a 'config' folder. Got: {self.config_path}")
        self.project_root = self.config_path.parent.parent

        # Load config with UTF-8
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Build ABSOLUTE path to raw data
        raw_rel = self.config['data']['raw_path']
        self.data_path = (self.project_root / raw_rel).resolve()

        self.df = None
        self.metadata = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load and validate insurance data
        
        Returns:
            DataFrame: Loaded and validated data
        """
        # Verify file exists before attempting to load
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at resolved path:\n{self.data_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Config used: {self.config_path}"
            )

        logger.info(f"Loading data from {self.data_path}")

        try:
            suffix = self.data_path.suffix.lower().strip()

            if suffix == '.csv':
                self.df = pd.read_csv(self.data_path, low_memory=False)
            elif suffix == '.parquet':
                self.df = pd.read_parquet(self.data_path)
            elif suffix in ('.xlsx', '.xls'):
                self.df = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")

            # Store metadata
            self.metadata['original_shape'] = self.df.shape
            self.metadata['original_columns'] = list(self.df.columns)

            return self.df

        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise

    def validate_data_structure(self) -> Dict:
        """
        Validate data structure and types
        
        Returns:
            Dict: Validation results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        validation_results = {
            'data_types': {},
            'missing_values': {},
            'unique_counts': {},
            'basic_stats': {}
        }

        # Check data types
        validation_results['data_types'] = self.df.dtypes.astype(str).to_dict()

        # Check missing values
        missing_counts = self.df.isnull().sum()
        missing_percentage = (missing_counts / len(self.df)) * 100
        validation_results['missing_values'] = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentage
        }).to_dict()

        # Check unique values for categorical columns
        categorical_cols = self.config['analysis']['categorical_columns']
        for col in categorical_cols:
            if col in self.df.columns:
                validation_results['unique_counts'][col] = {
                    'count': self.df[col].nunique(),
                    'values': self.df[col].unique()[:20].tolist()  # First 20 unique values
                }

        # Basic statistics for numerical columns
        numerical_cols = self.config['analysis']['numerical_columns']
        for col in numerical_cols:
            if col in self.df.columns:
                validation_results['basic_stats'][col] = {
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'median': float(self.df[col].median()),
                    'q1': float(self.df[col].quantile(0.25)),
                    'q3': float(self.df[col].quantile(0.75))
                }

        return validation_results

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data for analysis"""
        logger.info("Starting data preprocessing")

        # Convert date columns
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(
                self.df['TransactionMonth'], errors='coerce'
            )
            self.df['TransactionYear'] = self.df['TransactionMonth'].dt.year
            self.df['TransactionMonthNum'] = self.df['TransactionMonth'].dt.month
            self.df['TransactionQuarter'] = self.df['TransactionMonth'].dt.quarter

        # Calculate derived metrics
        self._calculate_derived_metrics()

        # Handle missing values
        self._handle_missing_values()

        # Clean categorical variables
        self._clean_categorical_variables()

        # Remove extreme outliers
        self._remove_extreme_outliers()

        logger.info(f"Data preprocessing completed. Final shape: {self.df.shape}")
        return self.df

    def _calculate_derived_metrics(self):
        """Calculate derived metrics for analysis"""
        # Loss Ratio
        self.df['LossRatio'] = np.where(
            self.df['TotalPremium'] > 0,
            self.df['TotalClaims'] / self.df['TotalPremium'],
            np.nan
        )

        # Claim Frequency (if PolicyID available)
        if 'PolicyID' in self.df.columns:
            total_policies = self.df['PolicyID'].nunique()
            if total_policies > 0:
                self.df['ClaimFrequency'] = len(self.df) / total_policies

        # Premium to SumInsured Ratio
        if 'SumInsured' in self.df.columns:
            self.df['PremiumToSumInsuredRatio'] = np.where(
                self.df['SumInsured'] > 0,
                self.df['TotalPremium'] / self.df['SumInsured'],
                np.nan
            )

    def _handle_missing_values(self):
        """Handle missing values based on column type"""
        numerical_cols = self.config['analysis']['numerical_columns']
        for col in numerical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")

        categorical_cols = self.config['analysis']['categorical_columns']
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col] = self.df[col].fillna(mode_val)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")

    def _clean_categorical_variables(self):
        """Clean and standardize categorical variables"""
        categorical_cols = self.config['analysis']['categorical_columns']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.title()
                    .replace(['', 'Nan', 'None', 'N/A', 'nan', 'NULL'], 'Unknown')
                )

    def _remove_extreme_outliers(self):
        """Remove extreme outliers using 1st/99th percentiles"""
        numerical_cols = self.config['analysis']['numerical_columns']
        initial_rows = len(self.df)

        for col in numerical_cols:
            if col in self.df.columns:
                lower = self.df[col].quantile(0.01)
                upper = self.df[col].quantile(0.99)
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

        removed_rows = initial_rows - len(self.df)
        logger.info(f"Removed {removed_rows} extreme outlier rows")

    def save_processed_data(self, output_path: Optional[str] = None):
        """Save processed data to file"""
        if output_path is None:
            output_path = self.config['data']['processed_path']

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_parquet(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        pd.Series(self.metadata).to_json(metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")