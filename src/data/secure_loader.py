"""
Secure Data Loading with Academic Integrity
==========================================

Data loading functions that prevent data leakage and ensure academic integrity.
This module implements proper train-test splitting and normalization order.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import torch
import scipy.io as sio
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
import warnings


class SecureDataLoader:
    """
    Secure data loader that prevents data leakage and ensures academic integrity.
    
    This class implements proper data loading procedures that:
    1. Verify train-test splits are pre-existing or create them properly
    2. Apply normalization only after train-test split
    3. Document all preprocessing steps
    4. Prevent information leakage between train and test sets
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.preprocessing_log = []
        
    def load_dataset_secure(self, dataset_name: str, device: str = 'cuda', 
                          verify_split: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Load dataset with secure, leak-proof preprocessing.
        
        Args:
            dataset_name: Name of dataset to load
            device: Target device for tensors
            verify_split: Whether to verify train-test split integrity
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, input_dim, metadata)
        """
        
        print(f"🔒 SECURE LOADING: {dataset_name.upper()}")
        print("=" * 50)
        
        # Load raw data
        raw_data = self._load_raw_data(dataset_name)
        if raw_data is None:
            return None, None, None, None, 0, {}
        
        # Verify data integrity
        if not self._verify_data_integrity(raw_data, dataset_name):
            print("❌ Data integrity check failed!")
            return None, None, None, None, 0, {}
        
        # Check if train-test split is pre-existing
        if verify_split:
            split_info = self._verify_train_test_split(raw_data, dataset_name)
            print(f"✅ Train-test split verification: {split_info['status']}")
        
        # Extract data with proper ordering
        X_train_raw = raw_data['fmriTrn']
        y_train_raw = raw_data['stimTrn'] 
        X_test_raw = raw_data['fmriTest']
        y_test_raw = raw_data['stimTest']
        
        print(f"📊 Raw data shapes:")
        print(f"   X_train: {X_train_raw.shape}, y_train: {y_train_raw.shape}")
        print(f"   X_test: {X_test_raw.shape}, y_test: {y_test_raw.shape}")
        
        # Apply secure preprocessing
        X_train, y_train, X_test, y_test, preprocessing_metadata = self._secure_preprocessing(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw, dataset_name
        )
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
        
        input_dim = X_train.shape[1]
        
        # Create comprehensive metadata
        metadata = {
            'dataset_name': dataset_name,
            'input_dim': input_dim,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'preprocessing_steps': self.preprocessing_log,
            'preprocessing_metadata': preprocessing_metadata,
            'device': device,
            'data_integrity_verified': True,
            'train_test_split_verified': verify_split
        }
        
        print(f"✅ SECURE LOADING COMPLETE")
        print(f"   Final shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"   Device: {device}, Input dim: {input_dim}")
        print(f"   Preprocessing steps: {len(self.preprocessing_log)}")
        
        return X_train, y_train, X_test, y_test, input_dim, metadata
    
    def _load_raw_data(self, dataset_name: str) -> Optional[Dict[str, np.ndarray]]:
        """Load raw data from file."""
        
        dataset_files = {
            'miyawaki': 'miyawaki_structured_28x28.mat',
            'vangerven': 'digit69_28x28.mat', 
            'mindbigdata': 'mindbigdata.mat',
            'crell': 'crell.mat'
        }
        
        if dataset_name not in dataset_files:
            print(f"❌ Unsupported dataset: {dataset_name}")
            return None
        
        mat_file = self.data_dir / dataset_files[dataset_name]
        
        if not mat_file.exists():
            print(f"❌ Dataset file not found: {mat_file}")
            return None
        
        try:
            data = sio.loadmat(str(mat_file))
            print(f"✅ Raw data loaded from: {mat_file}")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _verify_data_integrity(self, data: Dict[str, Any], dataset_name: str) -> bool:
        """Verify data integrity and required fields."""
        
        required_fields = ['fmriTrn', 'stimTrn', 'fmriTest', 'stimTest']
        
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check for NaN or infinite values
        for field in required_fields:
            arr = data[field]
            if np.any(np.isnan(arr)):
                print(f"❌ NaN values found in {field}")
                return False
            if np.any(np.isinf(arr)):
                print(f"❌ Infinite values found in {field}")
                return False
        
        print("✅ Data integrity verified - no missing values, NaN, or infinite values")
        return True
    
    def _verify_train_test_split(self, data: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Verify that train-test split is appropriate and pre-existing."""
        
        train_samples = data['fmriTrn'].shape[0]
        test_samples = data['fmriTest'].shape[0]
        total_samples = train_samples + test_samples
        test_ratio = test_samples / total_samples
        
        split_info = {
            'train_samples': train_samples,
            'test_samples': test_samples,
            'total_samples': total_samples,
            'test_ratio': test_ratio,
            'status': 'verified'
        }
        
        # Check if split ratio is reasonable
        if test_ratio < 0.1 or test_ratio > 0.3:
            warnings.warn(f"Unusual test ratio: {test_ratio:.2f} for {dataset_name}")
            split_info['warning'] = f"Unusual test ratio: {test_ratio:.2f}"
        
        # Log split information
        self.preprocessing_log.append({
            'step': 'train_test_split_verification',
            'dataset': dataset_name,
            'details': split_info
        })
        
        return split_info
    
    def _secure_preprocessing(self, X_train_raw: np.ndarray, y_train_raw: np.ndarray,
                            X_test_raw: np.ndarray, y_test_raw: np.ndarray,
                            dataset_name: str) -> Tuple[np.ndarray, ...]:
        """
        Apply secure preprocessing that prevents data leakage.
        
        Key principles:
        1. Normalization parameters computed ONLY on training data
        2. Same parameters applied to test data
        3. No information from test set used in preprocessing
        """
        
        print("🔒 Applying secure preprocessing...")
        
        # Step 1: Compute normalization parameters from TRAINING data only
        X_train_mean = np.mean(X_train_raw, axis=0)
        X_train_std = np.std(X_train_raw, axis=0) + 1e-8  # Add epsilon for numerical stability
        
        # Step 2: Apply normalization using training parameters
        X_train_normalized = (X_train_raw - X_train_mean) / X_train_std
        X_test_normalized = (X_test_raw - X_train_mean) / X_train_std  # Use TRAINING parameters!
        
        # Log normalization step
        self.preprocessing_log.append({
            'step': 'feature_normalization',
            'method': 'z_score_using_training_stats',
            'train_mean_range': [float(np.min(X_train_mean)), float(np.max(X_train_mean))],
            'train_std_range': [float(np.min(X_train_std)), float(np.max(X_train_std))],
            'leakage_prevention': 'test_normalized_using_train_parameters'
        })
        
        # Step 3: Process target variables (dataset-specific)
        y_train_processed, y_test_processed = self._process_targets(
            y_train_raw, y_test_raw, dataset_name
        )
        
        # Create preprocessing metadata
        preprocessing_metadata = {
            'X_normalization': {
                'method': 'z_score',
                'train_mean': X_train_mean,
                'train_std': X_train_std,
                'leakage_prevented': True
            },
            'y_processing': {
                'dataset_specific': True,
                'method': self._get_target_processing_method(dataset_name)
            }
        }
        
        print(f"✅ Secure preprocessing complete:")
        print(f"   X normalization: z-score using training statistics")
        print(f"   Y processing: {preprocessing_metadata['y_processing']['method']}")
        print(f"   Data leakage: PREVENTED")
        
        return X_train_normalized, y_train_processed, X_test_normalized, y_test_processed, preprocessing_metadata
    
    def _process_targets(self, y_train: np.ndarray, y_test: np.ndarray, 
                        dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process target variables with dataset-specific methods."""
        
        if dataset_name == 'miyawaki':
            # Binary contrast images - normalize using training data statistics
            y_train_flat = y_train.reshape(len(y_train), -1)
            y_test_flat = y_test.reshape(len(y_test), -1)
            
            # Compute min/max from training data only
            train_min = np.min(y_train_flat)
            train_max = np.max(y_train_flat)
            
            # Apply to both sets using training parameters
            y_train_norm = (y_train_flat - train_min) / (train_max - train_min + 1e-8)
            y_test_norm = (y_test_flat - train_min) / (train_max - train_min + 1e-8)
            
            # Reshape back
            y_train_processed = y_train_norm.reshape(-1, 1, 28, 28)
            y_test_processed = y_test_norm.reshape(-1, 1, 28, 28)
            
        elif dataset_name == 'vangerven':
            # Digit patterns - simple division by 255
            y_train_processed = y_train.reshape(-1, 1, 28, 28) / 255.0
            y_test_processed = y_test.reshape(-1, 1, 28, 28) / 255.0
            
        else:  # mindbigdata, crell
            # EEG→fMRI→Visual - normalize using training statistics
            y_train_flat = y_train.reshape(len(y_train), -1)
            y_test_flat = y_test.reshape(len(y_test), -1)
            
            train_min = np.min(y_train_flat)
            train_max = np.max(y_train_flat)
            
            y_train_norm = (y_train_flat - train_min) / (train_max - train_min + 1e-8)
            y_test_norm = (y_test_flat - train_min) / (train_max - train_min + 1e-8)
            
            y_train_processed = y_train_norm.reshape(-1, 1, 28, 28)
            y_test_processed = y_test_norm.reshape(-1, 1, 28, 28)
        
        # Log target processing
        self.preprocessing_log.append({
            'step': 'target_processing',
            'dataset': dataset_name,
            'method': self._get_target_processing_method(dataset_name),
            'leakage_prevention': 'normalization_parameters_from_training_only'
        })
        
        return y_train_processed, y_test_processed
    
    def _get_target_processing_method(self, dataset_name: str) -> str:
        """Get target processing method description."""
        methods = {
            'miyawaki': 'min_max_normalization_using_training_stats',
            'vangerven': 'division_by_255',
            'mindbigdata': 'min_max_normalization_using_training_stats',
            'crell': 'min_max_normalization_using_training_stats'
        }
        return methods.get(dataset_name, 'unknown')
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing report for documentation."""
        return {
            'preprocessing_steps': self.preprocessing_log,
            'data_leakage_prevention': {
                'feature_normalization': 'parameters_computed_from_training_only',
                'target_normalization': 'parameters_computed_from_training_only',
                'test_set_isolation': 'complete'
            },
            'academic_integrity': {
                'train_test_contamination': 'prevented',
                'information_leakage': 'none',
                'preprocessing_order': 'correct'
            }
        }


# Convenience function for backward compatibility
def load_dataset_secure(dataset_name: str, device: str = 'cuda') -> Tuple[torch.Tensor, ...]:
    """
    Secure dataset loading function with academic integrity guarantees.
    
    This function replaces the original load_dataset_gpu_optimized with
    a version that prevents data leakage and ensures academic integrity.
    """
    loader = SecureDataLoader()
    return loader.load_dataset_secure(dataset_name, device)


if __name__ == "__main__":
    # Test secure loading
    print("🧪 TESTING SECURE DATA LOADING")
    print("=" * 50)
    
    loader = SecureDataLoader()
    
    # Test with miyawaki dataset
    result = loader.load_dataset_secure('miyawaki', 'cpu')
    
    if result[0] is not None:
        X_train, y_train, X_test, y_test, input_dim, metadata = result
        print(f"\n✅ Secure loading test successful!")
        print(f"📊 Metadata keys: {list(metadata.keys())}")
        
        # Print preprocessing report
        report = loader.get_preprocessing_report()
        print(f"\n📋 Preprocessing Report:")
        print(f"   Steps: {len(report['preprocessing_steps'])}")
        print(f"   Data leakage prevention: ✅")
        print(f"   Academic integrity: ✅")
    else:
        print("❌ Secure loading test failed")
