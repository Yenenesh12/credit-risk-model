#!/usr/bin/env python3
"""
Data Loading and Preprocessing Script for DVC Pipeline
"""
import sys
import json
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import InsuranceDataProcessor
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting data loading and preprocessing...")
    
    try:
        # ✅ Config path relative to SCRIPT, not CWD
        config_path = script_dir.parent / "config" / "config.yaml"
        
        processor = InsuranceDataProcessor(str(config_path))
        
        df = processor.load_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        df_processed = processor.preprocess_data()
        logger.info(f"Data preprocessed. New shape: {df_processed.shape}")
        
        processor.save_processed_data()
        logger.info("Processed data saved successfully")
        
        # Save metadata
        metadata = {
            'processing_date': str(pd.Timestamp.now()),
            'original_shape': processor.metadata.get('original_shape', df.shape),
            'processed_shape': df_processed.shape,
            'missing_values_original': int(df.isnull().sum().sum()),
            'missing_values_processed': int(df_processed.isnull().sum().sum()),
            'columns_processed': list(df_processed.columns)
        }
        
        metadata_path = project_root / "data" / "processed" / "data_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in data loading and preprocessing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)