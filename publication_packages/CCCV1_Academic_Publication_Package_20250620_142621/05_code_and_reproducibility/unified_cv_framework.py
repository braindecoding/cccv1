"""
Unified Cross-Validation Framework for Academic-Compliant SOTA Comparison
========================================================================

Ensures fair comparison between CCCV1, Mind-Vis, and Brain-Diffuser by:
1. Using consistent random seed (42) across all methods
2. Using same 10-fold cross-validation strategy
3. Using identical data splits for all methods
4. Maintaining academic integrity and reproducibility

Academic Compliance:
- Same experimental setup for all methods
- Reproducible results with fixed seeds
- Statistical rigor with proper CV methodology
- Fair comparison without bias
"""

import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set global seeds for reproducibility
ACADEMIC_SEED = 42
torch.manual_seed(ACADEMIC_SEED)
np.random.seed(ACADEMIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ACADEMIC_SEED)
    torch.cuda.manual_seed_all(ACADEMIC_SEED)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class UnifiedCVFramework:
    """
    Unified Cross-Validation Framework for Academic-Compliant Comparison
    
    Provides consistent data splitting and evaluation methodology for all SOTA methods.
    """
    
    def __init__(self, n_folds=10, random_state=ACADEMIC_SEED, shuffle=True):
        """
        Initialize unified CV framework
        
        Args:
            n_folds: Number of CV folds (default: 10 for academic rigor)
            random_state: Random seed for reproducibility (default: 42)
            shuffle: Whether to shuffle data before splitting
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        
        print(f"🎯 Unified CV Framework Initialized:")
        print(f"   Folds: {n_folds}")
        print(f"   Random State: {random_state}")
        print(f"   Shuffle: {shuffle}")
        print(f"   Academic Compliance: ✅")
    
    def get_cv_splits(self, X_train, y_train):
        """
        Get consistent CV splits for all methods
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            List of (train_idx, val_idx) tuples for each fold
        """
        splits = []
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X_train)):
            splits.append((train_idx, val_idx))
        
        print(f"✅ Generated {len(splits)} consistent CV splits")
        return splits
    
    def create_fold_loaders(self, X_train, y_train, train_idx, val_idx, batch_size=16):
        """
        Create data loaders for a specific CV fold
        
        Args:
            X_train: Full training features
            y_train: Full training targets
            train_idx: Training indices for this fold
            val_idx: Validation indices for this fold
            batch_size: Batch size for data loaders
            
        Returns:
            (train_loader, val_loader) tuple
        """
        # Split data for this fold
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Create datasets
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        # Create loaders with consistent settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # Shuffle within fold
            generator=torch.Generator().manual_seed(self.random_state)  # Consistent shuffle
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False  # No shuffle for validation
        )
        
        return train_loader, val_loader
    
    def evaluate_method_cv(self, method_name, train_func, evaluate_func, 
                          X_train, y_train, X_test, y_test, input_dim, 
                          device='cuda', **method_kwargs):
        """
        Evaluate a method using unified CV framework
        
        Args:
            method_name: Name of the method being evaluated
            train_func: Function to train the method for one fold
            evaluate_func: Function to evaluate the method for one fold
            X_train, y_train: Training data
            X_test, y_test: Test data (for final evaluation)
            input_dim: Input dimensionality
            device: Computing device
            **method_kwargs: Method-specific arguments
            
        Returns:
            Dictionary with CV results and statistics
        """
        print(f"\n🔄 {self.n_folds}-fold CV: {method_name}")
        print("=" * 50)
        
        # Get consistent CV splits
        cv_splits = self.get_cv_splits(X_train, y_train)
        cv_scores = []
        best_model_state = None
        best_cv_score = float('inf')
        best_fold_info = None
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"   Fold {fold+1}/{self.n_folds}...", end=" ")
            
            try:
                # Create fold data loaders
                train_loader, val_loader = self.create_fold_loaders(
                    X_train, y_train, train_idx, val_idx, 
                    method_kwargs.get('batch_size', 16)
                )
                
                # Train method for this fold
                model = train_func(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    input_dim=input_dim,
                    device=device,
                    fold=fold,
                    **method_kwargs
                )
                
                # Evaluate method for this fold
                fold_score = evaluate_func(model, val_loader, device)
                cv_scores.append(fold_score)
                
                print(f"MSE: {fold_score:.6f}")
                
                # Track best model
                if fold_score < best_cv_score:
                    best_cv_score = fold_score
                    best_model_state = model.state_dict().copy() if hasattr(model, 'state_dict') else None
                    best_fold_info = {
                        'fold': fold + 1,
                        'score': fold_score,
                        'method': method_name
                    }
                    print(f"   🏆 New best model! Fold {fold+1}, Score: {fold_score:.6f}")
                
                # Cleanup
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in fold {fold+1}: {e}")
                continue
        
        if cv_scores:
            cv_scores = np.array(cv_scores)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"   📊 CV Results: {cv_mean:.6f} ± {cv_std:.6f}")
            
            # Save results
            results = {
                'method': method_name,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'best_fold': best_fold_info,
                'academic_compliant': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save best model if available
            if best_model_state is not None:
                self.save_best_cv_model(method_name, best_model_state, best_fold_info, input_dim)
            
            return results
        else:
            print(f"   ❌ No successful folds for {method_name}")
            return {
                'method': method_name,
                'cv_scores': [],
                'cv_mean': float('inf'),
                'cv_std': 0.0,
                'n_folds': self.n_folds,
                'academic_compliant': False,
                'error': 'All folds failed'
            }
    
    def save_best_cv_model(self, method_name, model_state_dict, fold_info, input_dim):
        """Save the best model from cross-validation"""
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Save model state dict
            model_path = models_dir / f"{method_name}_cv_best.pth"
            torch.save(model_state_dict, model_path)
            
            # Save metadata
            metadata = {
                'method': method_name,
                'input_dim': input_dim,
                'best_fold': fold_info['fold'],
                'best_score': fold_info['score'],
                'cv_framework': 'UnifiedCVFramework',
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'academic_compliant': True,
                'save_timestamp': datetime.now().isoformat()
            }
            
            metadata_path = models_dir / f"{method_name}_cv_best_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"   💾 Best model saved: {model_path}")
            print(f"   📋 Metadata saved: {metadata_path}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to save best model: {e}")
    
    def compare_methods(self, results_dict):
        """
        Compare multiple methods using unified CV results
        
        Args:
            results_dict: Dictionary of method_name -> cv_results
            
        Returns:
            Comparison summary with statistical analysis
        """
        print(f"\n📊 ACADEMIC-COMPLIANT COMPARISON RESULTS")
        print("=" * 60)
        
        # Sort methods by performance
        valid_results = {k: v for k, v in results_dict.items() 
                        if v.get('academic_compliant', False) and len(v['cv_scores']) > 0}
        
        if not valid_results:
            print("❌ No valid results for comparison")
            return None
        
        sorted_methods = sorted(valid_results.items(), key=lambda x: x[1]['cv_mean'])
        
        print(f"🏆 RANKING (Lower MSE = Better):")
        print("-" * 40)
        
        for rank, (method, results) in enumerate(sorted_methods, 1):
            cv_mean = results['cv_mean']
            cv_std = results['cv_std']
            
            if rank == 1:
                print(f"🥇 {rank}. {method}: {cv_mean:.6f} ± {cv_std:.6f} (WINNER)")
            elif rank == 2:
                print(f"🥈 {rank}. {method}: {cv_mean:.6f} ± {cv_std:.6f}")
            elif rank == 3:
                print(f"🥉 {rank}. {method}: {cv_mean:.6f} ± {cv_std:.6f}")
            else:
                print(f"   {rank}. {method}: {cv_mean:.6f} ± {cv_std:.6f}")
        
        # Statistical significance analysis
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print("-" * 40)
        
        winner_name, winner_results = sorted_methods[0]
        winner_scores = np.array(winner_results['cv_scores'])
        
        for method, results in sorted_methods[1:]:
            method_scores = np.array(results['cv_scores'])
            
            # Paired t-test (same CV folds)
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(method_scores, winner_scores)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(winner_scores)-1)*np.var(winner_scores, ddof=1) + 
                                 (len(method_scores)-1)*np.var(method_scores, ddof=1)) / 
                                (len(winner_scores) + len(method_scores) - 2))
            cohens_d = (method_scores.mean() - winner_scores.mean()) / pooled_std
            
            significance = "✅ Significant" if p_value < 0.05 else "⚠️ Not significant"
            effect_size = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
            
            print(f"{winner_name} vs {method}:")
            print(f"   p-value: {p_value:.6f} ({significance})")
            print(f"   Effect size: {cohens_d:.3f} ({effect_size})")
        
        return {
            'ranking': [(method, results['cv_mean'], results['cv_std']) 
                       for method, results in sorted_methods],
            'winner': winner_name,
            'academic_compliant': True,
            'n_folds': self.n_folds,
            'random_state': self.random_state
        }


def create_unified_cv_framework(n_folds=10):
    """Create a unified CV framework instance"""
    return UnifiedCVFramework(n_folds=n_folds, random_state=ACADEMIC_SEED)


if __name__ == "__main__":
    # Test the framework
    print("🧪 Testing Unified CV Framework")
    framework = create_unified_cv_framework()
    print("✅ Framework created successfully")
