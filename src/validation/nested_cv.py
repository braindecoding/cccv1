"""
Nested Cross-Validation for Academic Integrity
==============================================

Implementation of nested cross-validation to prevent hyperparameter overfitting
and ensure unbiased performance estimation for academic research.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, ParameterGrid
from typing import Dict, List, Tuple, Any, Optional
import json
import warnings
from datetime import datetime


class NestedCrossValidation:
    """
    Nested cross-validation implementation for unbiased hyperparameter tuning.
    
    This class implements proper nested CV to prevent hyperparameter overfitting:
    - Outer loop: Unbiased performance estimation
    - Inner loop: Hyperparameter optimization
    - Complete isolation between loops
    """
    
    def __init__(self, outer_folds: int = 5, inner_folds: int = 3, 
                 random_state: int = 42, verbose: bool = True):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.verbose = verbose
        self.results_log = []
        
    def nested_cv_evaluation(self, X: torch.Tensor, y: torch.Tensor, 
                           model_class: type, param_grid: Dict[str, List],
                           device: str = 'cuda') -> Dict[str, Any]:
        """
        Perform nested cross-validation evaluation.
        
        Args:
            X: Input features
            y: Target values
            model_class: Model class to instantiate
            param_grid: Hyperparameter grid for tuning
            device: Device for computation
            
        Returns:
            Dictionary with nested CV results
        """
        
        print(f"🔄 NESTED CROSS-VALIDATION")
        print("=" * 50)
        print(f"📊 Outer folds: {self.outer_folds}, Inner folds: {self.inner_folds}")
        print(f"🎛️ Parameter combinations: {len(list(ParameterGrid(param_grid)))}")
        print(f"📈 Total model trainings: {self.outer_folds * self.inner_folds * len(list(ParameterGrid(param_grid)))}")
        
        # Outer CV for unbiased performance estimation
        outer_cv = KFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state)
        
        outer_scores = []
        best_params_per_fold = []
        inner_cv_scores = []
        
        for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X)):
            if self.verbose:
                print(f"\n🔄 Outer Fold {outer_fold + 1}/{self.outer_folds}")
                print("-" * 30)
            
            # Split data for outer fold
            X_train_outer = X[train_outer_idx]
            y_train_outer = y[train_outer_idx]
            X_test_outer = X[test_outer_idx]
            y_test_outer = y[test_outer_idx]
            
            # Inner CV for hyperparameter optimization
            best_params, inner_scores = self._inner_cv_optimization(
                X_train_outer, y_train_outer, model_class, param_grid, device, outer_fold
            )
            
            # Train final model with best parameters on full outer training set
            final_model = self._train_final_model(
                X_train_outer, y_train_outer, model_class, best_params, device
            )
            
            # Evaluate on outer test set (unbiased evaluation)
            outer_score = self._evaluate_model(final_model, X_test_outer, y_test_outer, device)
            
            outer_scores.append(outer_score)
            best_params_per_fold.append(best_params)
            inner_cv_scores.append(inner_scores)
            
            if self.verbose:
                print(f"   🎯 Outer fold score: {outer_score:.6f}")
                print(f"   🏆 Best params: {best_params}")
        
        # Aggregate results
        results = self._aggregate_nested_cv_results(
            outer_scores, best_params_per_fold, inner_cv_scores, param_grid
        )
        
        # Log results
        self._log_nested_cv_results(results)
        
        return results
    
    def _inner_cv_optimization(self, X_train: torch.Tensor, y_train: torch.Tensor,
                              model_class: type, param_grid: Dict[str, List],
                              device: str, outer_fold: int) -> Tuple[Dict[str, Any], List[float]]:
        """
        Inner CV loop for hyperparameter optimization.
        
        Args:
            X_train: Training features for this outer fold
            y_train: Training targets for this outer fold
            model_class: Model class to instantiate
            param_grid: Hyperparameter grid
            device: Device for computation
            outer_fold: Current outer fold number
            
        Returns:
            Tuple of (best_params, inner_cv_scores)
        """
        
        inner_cv = KFold(n_splits=self.inner_folds, shuffle=True, 
                        random_state=self.random_state + outer_fold)
        
        param_combinations = list(ParameterGrid(param_grid))
        param_scores = []
        
        for param_idx, params in enumerate(param_combinations):
            if self.verbose:
                print(f"     🎛️ Testing params {param_idx + 1}/{len(param_combinations)}: {params}")
            
            # Inner CV evaluation for this parameter combination
            inner_fold_scores = []
            
            for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train)):
                # Split data for inner fold
                X_train_inner = X_train[train_inner_idx]
                y_train_inner = y_train[train_inner_idx]
                X_val_inner = X_train[val_inner_idx]
                y_val_inner = y_train[val_inner_idx]
                
                # Train model with current parameters
                model = self._train_model_with_params(
                    X_train_inner, y_train_inner, model_class, params, device
                )
                
                # Evaluate on inner validation set
                val_score = self._evaluate_model(model, X_val_inner, y_val_inner, device)
                inner_fold_scores.append(val_score)
                
                # Cleanup
                del model
                torch.cuda.empty_cache()
            
            # Average score for this parameter combination
            mean_score = np.mean(inner_fold_scores)
            param_scores.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': np.std(inner_fold_scores),
                'fold_scores': inner_fold_scores
            })
            
            if self.verbose:
                print(f"       📊 Mean CV score: {mean_score:.6f} ± {np.std(inner_fold_scores):.6f}")
        
        # Find best parameters
        best_param_result = min(param_scores, key=lambda x: x['mean_score'])
        best_params = best_param_result['params']
        
        if self.verbose:
            print(f"   🏆 Best inner CV score: {best_param_result['mean_score']:.6f}")
        
        return best_params, [result['mean_score'] for result in param_scores]
    
    def _train_model_with_params(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                model_class: type, params: Dict[str, Any], 
                                device: str) -> nn.Module:
        """Train model with specific hyperparameters."""
        
        # Create model with parameters
        model = model_class(
            input_dim=X_train.shape[1],
            device=device,
            config={'architecture': params.get('architecture', {})}
        )
        
        # Training configuration
        training_config = params.get('training', {})
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.get('lr', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-6)
        )
        
        criterion = nn.MSELoss()
        
        # Create data loader
        batch_size = training_config.get('batch_size', 16)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        epochs = training_config.get('epochs', 50)  # Reduced for inner CV
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output, _ = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=training_config.get('gradient_clip', 0.5)
                )
                
                optimizer.step()
                epoch_loss += loss.item()
        
        return model
    
    def _train_final_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          model_class: type, best_params: Dict[str, Any],
                          device: str) -> nn.Module:
        """Train final model with best parameters on full outer training set."""
        
        # Use more epochs for final model
        final_params = best_params.copy()
        if 'training' in final_params:
            final_params['training']['epochs'] = final_params['training'].get('epochs', 50) * 2
        
        return self._train_model_with_params(X_train, y_train, model_class, final_params, device)
    
    def _evaluate_model(self, model: nn.Module, X_test: torch.Tensor, 
                       y_test: torch.Tensor, device: str) -> float:
        """Evaluate model and return MSE score."""
        
        model.eval()
        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            output, _ = model(X_test)
            mse = nn.MSELoss()(output, y_test).item()
        
        return mse
    
    def _aggregate_nested_cv_results(self, outer_scores: List[float],
                                   best_params_per_fold: List[Dict[str, Any]],
                                   inner_cv_scores: List[List[float]],
                                   param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Aggregate nested CV results."""
        
        outer_scores = np.array(outer_scores)
        
        # Parameter frequency analysis
        param_frequency = {}
        for params in best_params_per_fold:
            param_str = json.dumps(params, sort_keys=True)
            param_frequency[param_str] = param_frequency.get(param_str, 0) + 1
        
        # Most frequent parameter combination
        most_frequent_params_str = max(param_frequency, key=param_frequency.get)
        most_frequent_params = json.loads(most_frequent_params_str)
        
        results = {
            'nested_cv_performance': {
                'mean_score': float(np.mean(outer_scores)),
                'std_score': float(np.std(outer_scores)),
                'scores_per_fold': outer_scores.tolist(),
                'confidence_interval_95': [
                    float(np.mean(outer_scores) - 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores))),
                    float(np.mean(outer_scores) + 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores)))
                ]
            },
            'hyperparameter_analysis': {
                'best_params_per_fold': best_params_per_fold,
                'most_frequent_params': most_frequent_params,
                'parameter_stability': param_frequency,
                'total_combinations_tested': len(list(ParameterGrid(param_grid)))
            },
            'inner_cv_analysis': {
                'inner_scores_per_outer_fold': inner_cv_scores,
                'mean_inner_cv_performance': [np.mean(scores) for scores in inner_cv_scores]
            },
            'methodology': {
                'outer_folds': self.outer_folds,
                'inner_folds': self.inner_folds,
                'total_model_trainings': self.outer_folds * self.inner_folds * len(list(ParameterGrid(param_grid))),
                'unbiased_estimation': True,
                'hyperparameter_overfitting_prevented': True
            }
        }
        
        return results
    
    def _log_nested_cv_results(self, results: Dict[str, Any]) -> None:
        """Log nested CV results for documentation."""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'nested_cross_validation',
            'results': results,
            'academic_integrity': {
                'unbiased_performance_estimation': True,
                'hyperparameter_overfitting_prevented': True,
                'proper_train_val_test_separation': True
            }
        }
        
        self.results_log.append(log_entry)
        
        # Print summary
        perf = results['nested_cv_performance']
        print(f"\n🎯 NESTED CV RESULTS SUMMARY")
        print("=" * 40)
        print(f"📊 Unbiased Performance: {perf['mean_score']:.6f} ± {perf['std_score']:.6f}")
        print(f"📈 95% CI: [{perf['confidence_interval_95'][0]:.6f}, {perf['confidence_interval_95'][1]:.6f}]")
        print(f"🎛️ Most Stable Params: {results['hyperparameter_analysis']['most_frequent_params']}")
        print(f"🔄 Total Model Trainings: {results['methodology']['total_model_trainings']}")
        print(f"✅ Academic Integrity: Hyperparameter overfitting prevented")
    
    def get_methodology_report(self) -> Dict[str, Any]:
        """Get comprehensive methodology report."""
        
        return {
            'nested_cv_configuration': {
                'outer_folds': self.outer_folds,
                'inner_folds': self.inner_folds,
                'random_state': self.random_state
            },
            'academic_integrity_measures': {
                'unbiased_performance_estimation': 'outer_cv_provides_unbiased_estimates',
                'hyperparameter_overfitting_prevention': 'inner_cv_isolates_hyperparameter_tuning',
                'proper_data_separation': 'complete_isolation_between_outer_and_inner_loops',
                'reproducibility': 'fixed_random_seeds_for_all_splits'
            },
            'methodology_advantages': [
                'Prevents hyperparameter overfitting',
                'Provides unbiased performance estimates',
                'Maintains proper train/validation/test separation',
                'Enables hyperparameter stability analysis',
                'Follows academic best practices'
            ],
            'results_log': self.results_log
        }


def create_hyperparameter_grid() -> Dict[str, List]:
    """
    Create hyperparameter grid for nested CV.
    
    This grid should be pre-specified and not modified based on results.
    """
    
    return {
        'architecture': [
            {'dropout_encoder': 0.05, 'dropout_decoder': 0.02, 'clip_residual_weight': 0.08},
            {'dropout_encoder': 0.06, 'dropout_decoder': 0.02, 'clip_residual_weight': 0.1},
            {'dropout_encoder': 0.04, 'dropout_decoder': 0.015, 'clip_residual_weight': 0.05}
        ],
        'training': [
            {'lr': 0.0003, 'batch_size': 8, 'weight_decay': 1e-8, 'epochs': 50},
            {'lr': 0.0005, 'batch_size': 12, 'weight_decay': 5e-8, 'epochs': 50},
            {'lr': 0.001, 'batch_size': 16, 'weight_decay': 1e-6, 'epochs': 50}
        ]
    }


if __name__ == "__main__":
    # Example usage
    print("🧪 TESTING NESTED CROSS-VALIDATION")
    print("=" * 50)
    
    # Create dummy data for testing
    X_dummy = torch.randn(100, 50)
    y_dummy = torch.randn(100, 1, 28, 28)
    
    # Create parameter grid
    param_grid = {
        'architecture': [
            {'dropout_encoder': 0.05, 'dropout_decoder': 0.02}
        ],
        'training': [
            {'lr': 0.001, 'batch_size': 16, 'epochs': 5}
        ]
    }
    
    print(f"✅ Nested CV test setup complete!")
    print(f"📊 Data shape: {X_dummy.shape}")
    print(f"🎛️ Parameter combinations: {len(list(ParameterGrid(param_grid)))}")
    print(f"⚠️ Note: Full test requires actual model class")
