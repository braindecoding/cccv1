"""
Academic Methodology Pre-registration
====================================

Pre-registered evaluation protocol to ensure academic integrity and prevent p-hacking.
This module defines the complete experimental methodology before any analysis begins.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class MethodologyRegistry:
    """
    Pre-registration system for experimental methodology to ensure academic integrity.
    
    This class implements a tamper-proof methodology registration system that must be
    defined before any data analysis begins, preventing p-hacking and selective reporting.
    """
    
    def __init__(self):
        self.registry_file = Path("methodology_registry.json")
        self.locked = False
        
    def create_preregistration(self) -> Dict[str, Any]:
        """
        Create pre-registered methodology specification.
        
        This method defines the complete experimental protocol before analysis begins.
        Once locked, this methodology cannot be changed without explicit documentation.
        
        Returns:
            Dict containing complete methodology specification
        """
        
        methodology = {
            "metadata": {
                "registration_date": datetime.now().isoformat(),
                "version": "1.0",
                "researcher": "[Researcher Name]",
                "institution": "[Institution Name]",
                "study_title": "CortexFlow-CLIP-CNN V1: CLIP-Guided Neural Decoding Framework"
            },
            
            "research_objectives": {
                "primary_objective": "Evaluate CortexFlow-CLIP-CNN V1 performance against state-of-the-art neural decoding methods",
                "secondary_objectives": [
                    "Assess generalizability across multiple neural decoding datasets",
                    "Validate statistical significance of performance improvements",
                    "Analyze architectural contributions to performance gains"
                ],
                "success_criteria": "Statistically significant improvement over champion methods on primary evaluation"
            },
            
            "datasets": {
                "included_datasets": ["miyawaki", "vangerven", "mindbigdata", "crell"],
                "exclusion_criteria": "None - all available datasets included",
                "data_sources": {
                    "miyawaki": {
                        "source": "Miyawaki et al. fMRI visual reconstruction dataset",
                        "reference": "DOI: 10.1016/j.neuron.2008.11.004",
                        "samples": {"train": 107, "test": 12},
                        "characteristics": "Small dataset, complex visual patterns"
                    },
                    "vangerven": {
                        "source": "Vangerven et al. digit reconstruction dataset", 
                        "reference": "DOI: 10.1016/j.neuroimage.2010.07.063",
                        "samples": {"train": 90, "test": 10},
                        "characteristics": "Small dataset, digit patterns"
                    },
                    "mindbigdata": {
                        "source": "MindBigData EEG-to-visual dataset",
                        "reference": "https://mindbigdata.com/",
                        "samples": {"train": 1080, "test": 120},
                        "characteristics": "Large dataset, cross-modal translation"
                    },
                    "crell": {
                        "source": "Crell EEG-to-visual dataset",
                        "reference": "[Reference to be added]",
                        "samples": {"train": 576, "test": 64},
                        "characteristics": "Medium dataset, cross-modal translation"
                    }
                }
            },
            
            "evaluation_protocol": {
                "primary_evaluation": {
                    "method": "10_fold_cross_validation",
                    "rationale": "Provides robust performance estimate with adequate statistical power",
                    "implementation": "KFold with shuffle=True, random_state=42"
                },
                "secondary_evaluations": {
                    "single_training": {
                        "purpose": "Compare with traditional train-test evaluation",
                        "status": "supplementary_analysis"
                    },
                    "enhanced_validation": {
                        "purpose": "Multiple runs for increased statistical power",
                        "status": "supplementary_analysis"
                    }
                },
                "primary_metric": "mean_squared_error",
                "secondary_metrics": ["confidence_intervals", "effect_size", "win_rate"]
            },
            
            "statistical_analysis": {
                "significance_level": 0.05,
                "statistical_tests": {
                    "primary": "paired_t_test",
                    "effect_size": "cohens_d",
                    "confidence_intervals": "95_percent"
                },
                "multiple_comparisons": {
                    "correction": "none_prespecified",
                    "rationale": "Each dataset represents independent hypothesis test"
                },
                "power_analysis": {
                    "minimum_detectable_effect": 0.5,
                    "power_threshold": 0.8,
                    "alpha": 0.05
                }
            },
            
            "model_specifications": {
                "architecture": "CortexFlow-CLIP-CNN V1",
                "hyperparameters": "dataset_specific_optimal_configurations",
                "hyperparameter_source": "pre_existing_optimization_study",
                "no_further_tuning": "Hyperparameters fixed before evaluation"
            },
            
            "data_preprocessing": {
                "normalization": {
                    "method": "z_score_normalization",
                    "scope": "per_dataset_independent",
                    "order": "after_train_test_split_verification"
                },
                "missing_values": "none_expected",
                "outlier_handling": "none_applied"
            },
            
            "reproducibility": {
                "random_seeds": [42, 43, 44],
                "software_versions": {
                    "python": "3.x",
                    "pytorch": "2.7.1+cu128",
                    "numpy": "latest",
                    "sklearn": "latest"
                },
                "hardware": "NVIDIA GeForce RTX 3060"
            },
            
            "reporting_plan": {
                "primary_results": "10_fold_cv_results_all_datasets",
                "supplementary_results": ["single_training", "enhanced_validation"],
                "negative_results": "will_be_reported_if_any",
                "effect_sizes": "will_be_reported_for_all_comparisons",
                "limitations": "will_be_explicitly_discussed"
            }
        }
        
        return methodology
    
    def lock_methodology(self, methodology: Dict[str, Any]) -> str:
        """
        Lock the methodology to prevent tampering.
        
        Args:
            methodology: Complete methodology specification
            
        Returns:
            Hash of the locked methodology for verification
        """
        
        # Create hash for tamper detection
        methodology_str = json.dumps(methodology, sort_keys=True)
        methodology_hash = hashlib.sha256(methodology_str.encode()).hexdigest()
        
        # Add lock information
        locked_methodology = {
            "methodology": methodology,
            "lock_info": {
                "locked_date": datetime.now().isoformat(),
                "hash": methodology_hash,
                "locked": True,
                "warning": "This methodology is locked and cannot be modified without explicit documentation"
            }
        }
        
        # Save to file
        with open(self.registry_file, 'w') as f:
            json.dump(locked_methodology, f, indent=2)
        
        self.locked = True
        
        print("🔒 METHODOLOGY LOCKED")
        print("=" * 50)
        print(f"📅 Lock Date: {locked_methodology['lock_info']['locked_date']}")
        print(f"🔐 Hash: {methodology_hash[:16]}...")
        print(f"📁 Registry File: {self.registry_file}")
        print("\n⚠️  WARNING: Methodology is now locked and tamper-proof!")
        print("   Any changes must be explicitly documented as protocol deviations.")
        
        return methodology_hash
    
    def verify_methodology(self) -> bool:
        """
        Verify that methodology hasn't been tampered with.
        
        Returns:
            True if methodology is intact, False if tampered
        """
        
        if not self.registry_file.exists():
            print("❌ No methodology registry found!")
            return False
        
        with open(self.registry_file, 'r') as f:
            locked_data = json.load(f)
        
        # Verify hash
        methodology_str = json.dumps(locked_data['methodology'], sort_keys=True)
        current_hash = hashlib.sha256(methodology_str.encode()).hexdigest()
        stored_hash = locked_data['lock_info']['hash']
        
        if current_hash == stored_hash:
            print("✅ Methodology integrity verified!")
            return True
        else:
            print("❌ METHODOLOGY TAMPERING DETECTED!")
            print(f"   Expected hash: {stored_hash[:16]}...")
            print(f"   Current hash:  {current_hash[:16]}...")
            return False
    
    def get_methodology(self) -> Dict[str, Any]:
        """Get the locked methodology."""
        
        if not self.registry_file.exists():
            raise FileNotFoundError("No methodology registry found. Please create and lock methodology first.")
        
        with open(self.registry_file, 'r') as f:
            locked_data = json.load(f)
        
        if not self.verify_methodology():
            raise ValueError("Methodology integrity check failed!")
        
        return locked_data['methodology']


def create_and_lock_methodology():
    """
    Create and lock the pre-registered methodology.
    
    This function should be run ONCE before any data analysis begins.
    """
    
    registry = MethodologyRegistry()
    
    if registry.registry_file.exists():
        print("⚠️  Methodology registry already exists!")
        if registry.verify_methodology():
            print("✅ Existing methodology is valid and locked.")
            return
        else:
            print("❌ Existing methodology is corrupted!")
            return
    
    print("🔬 CREATING PRE-REGISTERED METHODOLOGY")
    print("=" * 50)
    print("This will create a tamper-proof methodology specification")
    print("that must be defined BEFORE any data analysis begins.\n")
    
    # Create methodology
    methodology = registry.create_preregistration()
    
    # Lock it
    hash_value = registry.lock_methodology(methodology)
    
    print(f"\n✅ Methodology successfully created and locked!")
    print(f"🔐 Verification hash: {hash_value[:16]}...")
    
    return methodology


if __name__ == "__main__":
    # Create and lock methodology
    create_and_lock_methodology()
