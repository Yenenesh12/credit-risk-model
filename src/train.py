"""
Model training module for FRAUD DETECTION (based on Xente Challenge data).
Trains and evaluates models using FraudResult as target.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, List, Any
import warnings
import os
import argparse
warnings.filterwarnings('ignore')

# Import sklearn components
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, f1_score)
import joblib


class FraudDetectionModel:
    """Fraud detection model trainer (replaces misleading 'CreditRiskModel')."""
    
    def __init__(self, model_type: str = 'logistic', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.metrics = {}
        self.threshold = 0.5
        
        self.model_configs = {
            'logistic': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [1000],
                    'class_weight': ['balanced', None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample', None],
                    'random_state': [random_state]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.8, 1.0],
                    'random_state': [random_state]
                }
            }
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              cv_folds: int = 5, n_jobs: int = -1) -> None:
        print(f"Training {self.model_type} model...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Class distribution: {y_train.mean():.2%} positive")
        
        if self.model_type not in self.model_configs:
            raise ValueError(f"Model type {self.model_type} not supported. "
                           f"Choose from {list(self.model_configs.keys())}")
        
        config = self.model_configs[self.model_type]
        
        # Final safety: ensure all features are numeric
        assert X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), \
            "FATAL: Non-numeric columns detected in X_train. Fix preprocessing."
        
        grid_search = GridSearchCV(
            estimator=config['model'](),
            param_grid=config['params'],
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
        
        self._calculate_feature_importance(X_train)
    
    def _calculate_feature_importance(self, X_train: pd.DataFrame) -> None:
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\nEvaluating model on test set ({X_test.shape[0]} samples)...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        self.metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_recall_curve(y_test, y_pred_proba)[0].mean(),
            'recall': precision_recall_curve(y_test, y_pred_proba)[1].mean()
        }
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate', 'Fraud']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
        
        print(f"\nKey Metrics:")
        print(f"ROC-AUC: {self.metrics['roc_auc']:.4f}")
        print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        print(f"F1-Score: {self.metrics['f1_score']:.4f}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      save_path: str = None) -> None:
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_type.title()}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   save_path: str = None) -> None:
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_type.title()}')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        no_skill = len(y_test[y_test==1]) / len(y_test)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', 
                label='No Skill', color='red')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20, 
                               save_path: str = None) -> None:
        if self.feature_importance is None:
            print("No feature importance available for this model type")
            return
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type.title()}')
        plt.gca().invert_yaxis()
        
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'].values)):
            plt.text(importance, i, f' {importance:.4f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, 
                              method: str = 'f1') -> float:
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        if method == 'f1':
            thresholds = np.arange(0.1, 0.9, 0.01)
            f1_scores = [f1_score(y_val, (y_pred_proba >= t).astype(int)) for t in thresholds]
            self.threshold = thresholds[np.argmax(f1_scores)]
        elif method == 'youden':
            fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
            self.threshold = thresholds[np.argmax(tpr - fpr)]
        
        print(f"Optimal threshold ({method}): {self.threshold:.3f}")
        return self.threshold
    
    def save_model(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'threshold': self.threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data.get('metrics', {})
        self.threshold = model_data.get('threshold', 0.5)
        
        print(f"Model loaded from {filepath}")
        print(f"Model type: {self.model_type}")
        print(f"Training timestamp: {model_data.get('timestamp', 'Unknown')}")


def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  model_types: List[str] = None) -> pd.DataFrame:
    if model_types is None:
        model_types = ['logistic', 'random_forest', 'gradient_boosting']
    
    results = []
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type}...")
        print('='*60)
        
        model = FraudDetectionModel(model_type=model_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        results.append({'model_type': model_type, **metrics})
        model.save_model(f'../models/{model_type}_model.pkl')
        model.plot_roc_curve(X_test, y_test, f'../reports/roc_curve_{model_type}.png')
        model.plot_precision_recall_curve(X_test, y_test, f'../reports/pr_curve_{model_type}.png')
        model.plot_feature_importance(save_path=f'../reports/feature_importance_{model_type}.png')
    
    results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print('='*60)
    print(results_df.to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.2
    plt.bar([i - width for i in x], results_df['roc_auc'], width, label='ROC-AUC')
    plt.bar([i for i in x], results_df['average_precision'], width, label='Avg Precision')
    plt.bar([i + width for i in x], results_df['f1_score'], width, label='F1-Score')
    plt.xlabel('Model Type')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, results_df['model_type'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--train_data', type=str, 
                       default=r'C:\Users\admin\credit-risk-model\data\processed\train_data.csv')
    parser.add_argument('--test_data', type=str,
                       default=r'C:\Users\admin\credit-risk-model\data\processed\test_data.csv')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['logistic', 'random_forest', 'gradient_boosting', 'all'])
    parser.add_argument('--output_dir', type=str, default='../models')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    
    print("Loading data...")
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    
    # ✅ CRITICAL FIXES BELOW ✅
    # 1. Use correct target: FraudResult (not high_risk)
    target_col = 'FraudResult'
    
    # 2. Drop ID columns and invalid targets
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    invalid_cols = ['high_risk']  # Remove this fabricated column
    
    drop_cols = id_cols + invalid_cols
    
    X_train = train_data.drop(columns=drop_cols + [target_col], errors='ignore')
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=drop_cols + [target_col], errors='ignore')
    y_test = test_data[target_col]
    
    # 3. Ensure all features are numeric (final safety net)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Class distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    if args.model_type == 'all':
        model_types = ['logistic', 'random_forest', 'gradient_boosting']
        results = compare_models(X_train, y_train, X_test, y_test, model_types)
        results.to_csv('../reports/model_comparison.csv', index=False)
        best_model = results.iloc[0]['model_type']
        print(f"\nBest model: {best_model} (ROC-AUC: {results.iloc[0]['roc_auc']:.4f})")
    else:
        model = FraudDetectionModel(model_type=args.model_type)
        model.train(X_train, y_train)
        model.evaluate(X_test, y_test)
        model.save_model(f'{args.output_dir}/{args.model_type}_model.pkl')
        model.plot_roc_curve(X_test, y_test, f'../reports/roc_curve_{args.model_type}.png')
        model.plot_precision_recall_curve(X_test, y_test, f'../reports/pr_curve_{args.model_type}.png')
        model.plot_feature_importance(save_path=f'../reports/feature_importance_{args.model_type}.png')
    
    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()