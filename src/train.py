import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            classification_report, roc_curve, precision_recall_curve)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import warnings
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Credit risk model trainer with multiple algorithms."""
    
    def __init__(self, 
                 experiment_name: str = "credit_risk_modeling",
                 tracking_uri: str = "http://localhost:5000",
                 random_state: int = 42):
        
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model configurations."""
        
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 1.0]
                }
            }
        }
    
    def prepare_data(self, 
                    X: pd.DataFrame, 
                    y: pd.Series,
                    test_size: float = 0.2,
                    use_smote: bool = True,
                    smote_params: Dict = None) -> Tuple:
        """Prepare train/test data with optional SMOTE."""
        
        logger.info("Preparing data...")
        logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Apply SMOTE if requested
        if use_smote and smote_params is None:
            smote_params = {
                'random_state': self.random_state,
                'sampling_strategy': 'auto',
                'k_neighbors': 5
            }
        
        if use_smote:
            
            smote = SMOTE(**smote_params)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train shape: {X_train_resampled.shape}")
            logger.info(f"After SMOTE class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
            return X_train_resampled, X_test, y_train_resampled, y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, 
                   model_name: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   cv_folds: int = 5,
                   scoring: str = 'roc_auc') -> Dict:
        """Train a single model with cross-validation."""
        
        logger.info(f"Training {model_name}...")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configs")
        
        model_config = self.model_configs[model_name]
        model = model_config['model']
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv, scoring=scoring, n_jobs=-1)
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Store results
        results = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_name': model_name,
            'training_date': datetime.now().isoformat()
        }
        
        self.models[model_name] = model
        self.results[model_name] = results
        
        logger.info(f"{model_name} - CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def evaluate_model(self, 
                      model_name: str,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      threshold: float = 0.5) -> Dict:
        """Evaluate model performance."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        logger.info(f"\n{model_name} Evaluation:")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def hyperparameter_tuning(self,
                            model_name: str,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            n_trials: int = 50,
                            scoring: str = 'roc_auc') -> Dict:
        """Hyperparameter tuning using Optuna."""
        
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configs")
        
        model_class = type(self.model_configs[model_name]['model'])
        
        def objective(trial):
            # Define search space based on model
            if model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_loguniform('C', 0.01, 100),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'max_iter': 1000,
                    'random_state': self.random_state,
                    'class_weight': 'balanced'
                }
            
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'class_weight': 'balanced',
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            
            elif model_name in ['xgboost', 'lightgbm', 'gradient_boosting']:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'random_state': self.random_state
                }
                
                if model_name == 'xgboost':
                    params['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
                    params['eval_metric'] = 'logloss'
                    params['use_label_encoder'] = False
                
                elif model_name == 'lightgbm':
                    params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)
                    params['class_weight'] = 'balanced'
            
            # Create model
            model = model_class(**params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, 
                                    cv=cv, scoring=scoring, n_jobs=-1)
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_name}_tuning',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        logger.info(f"Best trial for {model_name}:")
        logger.info(f"  Value: {study.best_value:.4f}")
        logger.info(f"  Params: {study.best_params}")
        
        # Train best model
        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(X_train, y_train)
        
        # Store results
        self.models[f'{model_name}_tuned'] = best_model
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study,
            'model': best_model
        }
    
    def log_to_mlflow(self,
                     model_name: str,
                     model,
                     metrics: Dict,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     params: Dict = None):
        """Log model and metrics to MLflow."""
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics({
                'roc_auc': metrics.get('roc_auc', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1', 0)
            })
            
            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, 
                model_name,
                signature=signature
            )
            
            # Log artifacts
            if 'confusion_matrix' in metrics:
                # Create and save confusion matrix plot
                cm = np.array(metrics['confusion_matrix'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig('confusion_matrix.png')
                mlflow.log_artifact('confusion_matrix.png')
                plt.close()
            
            logger.info(f"Logged {model_name} to MLflow")
    
    def create_model_card(self, 
                         model_name: str,
                         metrics: Dict,
                         feature_importance: pd.DataFrame = None) -> Dict:
        """Create model card for documentation."""
        
        model_card = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'performance_metrics': {
                'roc_auc': float(metrics.get('roc_auc', 0)),
                'accuracy': float(metrics.get('accuracy', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1_score': float(metrics.get('f1', 0))
            },
            'model_parameters': self._get_model_params(model_name),
            'data_summary': {
                'n_samples_train': getattr(self, 'X_train_shape', (0,))[0],
                'n_features': getattr(self, 'X_train_shape', (0,))[1],
                'class_distribution': getattr(self, 'class_distribution', {})
            },
            'interpretability': {
                'feature_importance': feature_importance.to_dict() if feature_importance is not None else None
            },
            'regulatory_considerations': {
                'model_type': model_name,
                'interpretability': 'high' if model_name == 'logistic_regression' else 'medium',
                'validation_status': 'pending'
            }
        }
        
        return model_card
    
    def _get_model_params(self, model_name: str) -> Dict:
        """Get model parameters."""
        if model_name in self.models:
            model = self.models[model_name]
            return model.get_params()
        return {}
    
    def save_model(self, model_name: str, filepath: str):
        """Save model to disk."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Saved {model_name} to {filepath}")
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def save_all_artifacts(self, base_path: str = 'models'):
        """Save all models and artifacts."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(base_path, f'{model_name}.pkl')
            self.save_model(model_name, model_path)
        
        # Save results
        results_path = os.path.join(base_path, 'training_results.json')
        with open(results_path, 'w') as f:
            # Convert results to serializable format
            serializable_results = {}
            for name, result in self.results.items():
                serializable_results[name] = {
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std']),
                    'model_name': result['model_name'],
                    'training_date': result['training_date']
                }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved all artifacts to {base_path}")


def main():
    """Main training pipeline."""
    
    # Initialize trainer
    trainer = CreditRiskModelTrainer(
        experiment_name="credit_risk_baseline",
        random_state=42
    )
    
    # Load processed data
    logger.info("Loading processed data...")
    X = pd.read_csv('data/processed/features.csv')
    y = pd.read_csv('data/processed/target.csv').squeeze()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        X, y,
        test_size=0.2,
        use_smote=True
    )
    
    # Store shapes for documentation
    trainer.X_train_shape = X_train.shape
    trainer.class_distribution = pd.Series(y_train).value_counts().to_dict()
    
    # Train models
    models_to_train = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    for model_name in models_to_train:
        # Train with cross-validation
        results = trainer.train_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            cv_folds=5
        )
        
        # Evaluate
        metrics = trainer.evaluate_model(
            model_name=model_name,
            X_test=X_test,
            y_test=y_test
        )
        
        # Hyperparameter tuning (optional)
        if model_name in ['xgboost', 'lightgbm']:
            tuning_results = trainer.hyperparameter_tuning(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                n_trials=30
            )
            
            # Evaluate tuned model
            tuned_metrics = trainer.evaluate_model(
                model_name=f'{model_name}_tuned',
                X_test=X_test,
                y_test=y_test
            )
        
        # Log to MLflow
        trainer.log_to_mlflow(
            model_name=model_name,
            model=trainer.models[model_name],
            metrics=metrics,
            X_train=X_train,
            y_train=y_train
        )
        
        # Create feature importance for tree-based models
        feature_importance = None
        if hasattr(trainer.models[model_name], 'feature_importances_'):
            importance_scores = trainer.models[model_name].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        # Create model card
        model_card = trainer.create_model_card(
            model_name=model_name,
            metrics=metrics,
            feature_importance=feature_importance
        )
        
        # Save model card
        card_path = f'models/{model_name}_card.json'
        with open(card_path, 'w') as f:
            json.dump(model_card, f, indent=2)
    
    # Save all artifacts
    trainer.save_all_artifacts()
    
    # Select best model based on ROC-AUC
    best_model_name = max(trainer.results.items(), 
                         key=lambda x: x[1]['cv_mean'])[0]
    logger.info(f"\nBest model: {best_model_name}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()