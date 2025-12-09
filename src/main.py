"""
Main execution script for insurance risk analysis tasks.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from hypothesis_testing import InsuranceHypothesisTester
from modeling import InsuranceModelBuilder
import pandas as pd
import numpy as np


def run_task_3(data_path: str, output_dir: str = '../reports'):
    """Execute Task 3: Hypothesis Testing."""
    print("=" * 80)
    print("TASK 3: HYPOTHESIS TESTING")
    print("=" * 80)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Initialize tester
    tester = InsuranceHypothesisTester(df)
    tester.calculate_risk_metrics()
    
    # Run all tests
    print("\nRunning hypothesis tests...")
    
    try:
        province_results = tester.test_province_risk()
        print("✓ Province risk test completed")
    except Exception as e:
        print(f"✗ Province risk test failed: {e}")
        province_results = None
    
    try:
        zipcode_results = tester.test_zipcode_risk(top_n=10)
        print("✓ Zip code risk test completed")
    except Exception as e:
        print(f"✗ Zip code risk test failed: {e}")
        zipcode_results = None
    
    try:
        margin_results = tester.test_zipcode_margin(top_n=10)
        print("✓ Zip code margin test completed")
    except Exception as e:
        print(f"✗ Zip code margin test failed: {e}")
        margin_results = None
    
    try:
        gender_results = tester.test_gender_risk()
        print("✓ Gender risk test completed")
    except Exception as e:
        print(f"✗ Gender risk test failed: {e}")
        gender_results = None
    
    # Generate report
    print("\nGenerating report...")
    report = tester.generate_report()
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'task_3_report.md')
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to {report_path}")
    
    # Save results summary
    summary_df = pd.DataFrame()
    
    for test_name, results in tester.results.items():
        for metric_name, metric_results in results.items():
            if 'p_value' in metric_results:
                summary_df.loc[f"{test_name}_{metric_name}", 'p_value'] = metric_results['p_value']
                summary_df.loc[f"{test_name}_{metric_name}", 'significant'] = metric_results['p_value'] < 0.05
    
    summary_path = os.path.join(output_dir, 'task_3_results.csv')
    summary_df.to_csv(summary_path)
    print(f"✓ Results summary saved to {summary_path}")
    
    return tester.results


def run_task_4(data_path: str, output_dir: str = '../models'):
    """Execute Task 4: Predictive Modeling."""
    print("\n" + "=" * 80)
    print("TASK 4: PREDICTIVE MODELING")
    print("=" * 80)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Initialize model builder
    model_builder = InsuranceModelBuilder(df)
    
    # Train severity models
    print("\nTraining claim severity models...")
    severity_results = model_builder.train_severity_models()
    print(f"✓ Trained {len([k for k in severity_results.keys() if 'severity' in k])} severity models")
    
    # Train classification models
    print("\nTraining claim probability models...")
    classification_results = model_builder.train_classification_models()
    print(f"✓ Trained {len([k for k in classification_results.keys() if 'classification' in k])} classification models")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    for model_name in [model_builder.best_severity_model, model_builder.best_classification_model]:
        importance = model_builder.analyze_feature_importance(model_name)
        if importance is not None:
            print(f"✓ Feature importance analyzed for {model_name}")
    
    # Generate SHAP analysis
    print("\nGenerating SHAP analysis...")
    for model_name in [model_builder.best_severity_model, model_builder.best_classification_model]:
        shap_values, features = model_builder.generate_shap_analysis(model_name)
        if shap_values is not None:
            print(f"✓ SHAP analysis completed for {model_name}")
    
    # Generate report
    print("\nGenerating modeling report...")
    report = model_builder.generate_report()
    
    # Save report
    report_dir = '../reports'
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'task_4_report.md')
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to {report_path}")
    
    # Save models
    print("\nSaving models...")
    model_builder.save_models(output_dir)
    
    return model_builder.results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Insurance Risk Analysis')
    parser.add_argument('--task', type=int, choices=[3, 4, 34], default=34,
                       help='Task to run: 3 (hypothesis testing), 4 (modeling), or 34 (both)')
    parser.add_argument('--data', type=str, default='../data/processed/insurance_data_clean.csv',
                       help='Path to input data file')
    parser.add_argument('--output', type=str, default='../outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run tasks
    if args.task in [3, 34]:
        task3_results = run_task_3(args.data, args.output)
    
    if args.task in [4, 34]:
        task4_results = run_task_4(args.data, args.output)
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()