import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.hypothesis_testing import HypothesisTester
from src.modeling import InsuranceModeling
import pandas as pd

def run_task_3():
    """Execute Task 3: Hypothesis Testing"""
    print("Running Task 3: Hypothesis Testing...")
    
    tester = HypothesisTester('data/processed/insurance_data_clean.csv')
    tester.calculate_risk_metrics()
    
    # Run all hypothesis tests
    tests = [
        ('Province Risk', tester.test_province_risk),
        ('ZipCode Risk', tester.test_zipcode_risk),
        ('ZipCode Margin', tester.test_zipcode_margin),
        ('Gender Risk', tester.test_gender_risk)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        results[test_name] = test_func()
    
    # Generate report
    report = tester.generate_report()
    with open('reports/task_3_report.md', 'w') as f:
        f.write(report)
    
    print("Task 3 completed. Report saved to reports/task_3_report.md")
    return results

def run_task_4():
    """Execute Task 4: Modeling"""
    print("Running Task 4: Predictive Modeling...")
    
    modeler = InsuranceModeling('data/processed/insurance_data_clean.csv')
    
    # Train severity models
    print("Training claim severity models...")
    modeler.train_severity_models()
    
    # Train premium models
    print("Training premium prediction models...")
    modeler.train_premium_models()
    
    # Generate report
    report = modeler.generate_model_report()
    with open('reports/task_4_report.md', 'w') as f:
        f.write(report)
    
    # Save models
    for name, model in modeler.models.items():
        joblib.dump(model, f'models/{name}.pkl')
    
    print("Task 4 completed. Models saved to models/ directory.")
    print("Report saved to reports/task_4_report.md")
    
    return modeler.results

if __name__ == "__main__":
    # Run both tasks
    task3_results = run_task_3()
    task4_results = run_task_4()
    
    print("\n" + "="*50)
    print("All tasks completed successfully!")
    print("="*50)