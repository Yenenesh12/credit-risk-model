import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys
import os
import tempfile
import json
import joblib
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import CreditRiskModelTrainer


class TestCreditRiskModelTrainer:
    """Test credit risk model trainer."""
    
    def setup_method(self):
        """Setup test data."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            weights=[0.8, 0.2],  # Imbalanced
            random_state=42
        )
        
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        self.y = pd.Series(y)
        
        # Initialize trainer
        self.trainer = CreditRiskModelTrainer(
            experiment_name="test_experiment",
            tracking_uri="sqlite:///:memory:",  # In-memory tracking
            random_state=42
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.random_state == 42
        assert 'logistic_regression' in self.trainer.model_configs
        assert 'random_forest' in self.trainer.model_configs
        assert 'xgboost' in self.trainer.model_configs
        assert len(self.trainer.models) == 0
        assert len(self.trainer.results) == 0
    
    def test_prepare_data_without_smote(self):
        """Test data preparation without SMOTE."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        # Check shapes
        assert X_train.shape[0] == 160  # 80% of 200
        assert X_test.shape[0] == 40    # 20% of 200
        assert X_train.shape[1] == 20   # All features
        assert X_test.shape[1] == 20    # All features
        
        # Check stratification (approximately)
        train_class_ratio = y_train.mean()
        test_class_ratio = y_test.mean()
        assert abs(train_class_ratio - test_class_ratio) < 0.05
    
    def test_prepare_data_with_smote(self):
        """Test data preparation with SMOTE."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=True
        )
        
        # With SMOTE, training data should be balanced
        assert y_train.mean() == 0.5  # SMOTE creates balanced data
        
        # Test data should remain original distribution
        original_ratio = self.y.mean()
        test_ratio = y_test.mean()
        assert abs(test_ratio - original_ratio) < 0.05
    
    def test_train_logistic_regression(self):
        """Test training logistic regression model."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        # Train model
        results = self.trainer.train_model(
            model_name='logistic_regression',
            X_train=X_train,
            y_train=y_train,
            cv_folds=3,  # Smaller for faster tests
            scoring='roc_auc'
        )
        
        # Check results
        assert 'logistic_regression' in self.trainer.models
        assert 'logistic_regression' in self.trainer.results
        assert 'cv_mean' in results
        assert 'cv_std' in results
        assert results['cv_mean'] > 0.5  # Should be better than random
    
    def test_train_random_forest(self):
        """Test training random forest model."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        # Train model
        results = self.trainer.train_model(
            model_name='random_forest',
            X_train=X_train,
            y_train=y_train,
            cv_folds=3,
            scoring='roc_auc'
        )
        
        # Check results
        assert 'random_forest' in self.trainer.models
        assert results['cv_mean'] > 0.5
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        # Train model
        self.trainer.train_model(
            model_name='logistic_regression',
            X_train=X_train,
            y_train=y_train,
            cv_folds=3
        )
        
        # Evaluate
        metrics = self.trainer.evaluate_model(
            model_name='logistic_regression',
            X_test=X_test,
            y_test=y_test,
            threshold=0.5
        )
        
        # Check metrics
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        
        # Metrics should be between 0 and 1 (or 0-100 for some)
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        
        # Confusion matrix should be 2x2
        cm = metrics['confusion_matrix']
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning (light version)."""
        # Prepare data (small subset for speed)
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X.iloc[:100], self.y.iloc[:100],
            test_size=0.2,
            use_smote=False
        )
        
        # Hyperparameter tuning
        tuning_results = self.trainer.hyperparameter_tuning(
            model_name='random_forest',
            X_train=X_train,
            y_train=y_train,
            n_trials=5,  # Small number for speed
            scoring='roc_auc'
        )
        
        # Check results
        assert 'best_params' in tuning_results
        assert 'best_score' in tuning_results
        assert 'study' in tuning_results
        assert 'model' in tuning_results
        
        # Tuned model should be saved
        assert 'random_forest_tuned' in self.trainer.models
    
    def test_create_model_card(self):
        """Test model card creation."""
        # Prepare and train a model
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        self.trainer.train_model(
            model_name='logistic_regression',
            X_train=X_train,
            y_train=y_train
        )
        
        # Store shapes for model card
        self.trainer.X_train_shape = X_train.shape
        self.trainer.class_distribution = pd.Series(y_train).value_counts().to_dict()
        
        # Evaluate to get metrics
        metrics = self.trainer.evaluate_model(
            model_name='logistic_regression',
            X_test=X_test,
            y_test=y_test
        )
        
        # Create model card
        model_card = self.trainer.create_model_card(
            model_name='logistic_regression',
            metrics=metrics,
            feature_importance=None
        )
        
        # Check model card structure
        assert 'model_name' in model_card
        assert 'performance_metrics' in model_card
        assert 'model_parameters' in model_card
        assert 'data_summary' in model_card
        assert 'regulatory_considerations' in model_card
        
        # Check specific fields
        assert model_card['model_name'] == 'logistic_regression'
        assert 'roc_auc' in model_card['performance_metrics']
        assert model_card['regulatory_considerations']['model_type'] == 'logistic_regression'
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading models."""
        # Train a model
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        self.trainer.train_model(
            model_name='logistic_regression',
            X_train=X_train,
            y_train=y_train
        )
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        self.trainer.save_model('logistic_regression', str(model_path))
        
        # Check file exists
        assert model_path.exists()
        
        # Load model and compare predictions
        
        loaded_model = joblib.load(model_path)
        
        # Make predictions with both models
        original_preds = self.trainer.models['logistic_regression'].predict_proba(X_test)[:, 1]
        loaded_preds = loaded_model.predict_proba(X_test)[:, 1]
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
    
    def test_save_all_artifacts(self, tmp_path):
        """Test saving all training artifacts."""
        # Train multiple models
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        for model_name in ['logistic_regression', 'random_forest']:
            self.trainer.train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                cv_folds=3
            )
        
        # Save all artifacts
        artifacts_dir = tmp_path / "models"
        self.trainer.save_all_artifacts(str(artifacts_dir))
        
        # Check files were created
        assert (artifacts_dir / "logistic_regression.pkl").exists()
        assert (artifacts_dir / "random_forest.pkl").exists()
        assert (artifacts_dir / "training_results.json").exists()
        
        # Check training results JSON
        with open(artifacts_dir / "training_results.json", 'r') as f:
            results = json.load(f)
        
        assert 'logistic_regression' in results
        assert 'random_forest' in results
        assert 'cv_mean' in results['logistic_regression']
    
    def test_model_selection(self):
        """Test automatic model selection."""
        # Train multiple models
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y,
            test_size=0.2,
            use_smote=False
        )
        
        models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in models:
            self.trainer.train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                cv_folds=3
            )
        
        # Find best model based on CV score
        best_model_name = max(
            self.trainer.results.items(),
            key=lambda x: x[1]['cv_mean']
        )[0]
        
        # Best model should be one of the trained models
        assert best_model_name in models
        
        # Best model should have the highest CV mean
        best_score = self.trainer.results[best_model_name]['cv_mean']
        for model_name in models:
            if model_name != best_model_name:
                assert best_score >= self.trainer.results[model_name]['cv_mean']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])