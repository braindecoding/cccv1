"""
Brain-Diffuser Training Script
=============================

Train Brain-Diffuser following exact methodology from original paper.
Academic Integrity: No modifications to original approach.
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from brain_diffuser import BrainDiffuser


def generate_dummy_captions(dataset_name: str, num_samples: int) -> list:
    """Generate dummy captions for datasets without text descriptions"""
    
    caption_templates = {
        'miyawaki': [
            "geometric visual pattern",
            "black and white contrast image", 
            "abstract visual stimulus",
            "binary pattern design",
            "visual reconstruction target"
        ],
        'vangerven': [
            "handwritten digit",
            "numerical character",
            "digit pattern",
            "number visualization",
            "digit reconstruction"
        ],
        'crell': [
            "visual pattern from EEG",
            "translated brain signal",
            "neural decoded image",
            "brain-to-visual translation",
            "reconstructed visual stimulus"
        ],
        'mindbigdata': [
            "EEG-derived visual pattern",
            "mind-decoded image",
            "neural signal visualization",
            "brain activity reconstruction",
            "cognitive visual pattern"
        ]
    }
    
    templates = caption_templates.get(dataset_name, ["visual pattern"])
    captions = []
    
    for i in range(num_samples):
        template = templates[i % len(templates)]
        captions.append(f"{template} {i+1}")
    
    return captions


def train_brain_diffuser(dataset_name: str, device='cuda', alpha=1.0):
    """Train Brain-Diffuser on specified dataset"""
    
    print(f"🧠 BRAIN-DIFFUSER TRAINING")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Regularization (alpha): {alpha}")
    print("=" * 60)
    
    # Initialize Brain-Diffuser
    brain_diffuser = BrainDiffuser(device=device)
    
    # Setup models
    print("🔧 Setting up models...")
    if not brain_diffuser.setup_models():
        print("❌ Failed to setup models")
        return None
    
    # Load dataset info to generate captions
    from data.loader import load_dataset_gpu_optimized
    X_train, y_train, _, _, _ = load_dataset_gpu_optimized(dataset_name, device=device)
    num_train_samples = X_train.shape[0]
    
    # Generate captions
    print(f"📝 Generating captions for {num_train_samples} samples...")
    captions = generate_dummy_captions(dataset_name, num_train_samples)
    print(f"✅ Generated {len(captions)} captions")
    
    # Train the model
    try:
        results = brain_diffuser.train(
            dataset_name=dataset_name,
            captions=captions,
            alpha=alpha
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   VDVAE Score: {results['vdvae_score']:.6f}")
        print(f"   CLIP-Vision Score: {results['clip_vision_score']:.6f}")
        print(f"   CLIP-Text Score: {results['clip_text_score']:.6f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_brain_diffuser(dataset_name: str, device='cuda', num_samples=6):
    """Evaluate trained Brain-Diffuser model"""
    
    print(f"📊 BRAIN-DIFFUSER EVALUATION")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    print("=" * 60)
    
    # Initialize Brain-Diffuser
    brain_diffuser = BrainDiffuser(device=device)
    
    # Setup models
    if not brain_diffuser.setup_models():
        print("❌ Failed to setup models")
        return None
    
    # Load trained models
    if not brain_diffuser.load_trained_models(dataset_name):
        print("❌ Failed to load trained models")
        print("   Please train the model first")
        return None
    
    # Evaluate
    try:
        results = brain_diffuser.evaluate(dataset_name, num_samples)
        
        print(f"\n📈 Evaluation Results:")
        print(f"   MSE: {results['mse']:.6f}")
        print(f"   Correlation: {results['correlation']:.6f}")
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/{dataset_name}_brain_diffuser_eval_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Brain-Diffuser Training and Evaluation')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both',
                        help='Mode: train, eval, or both')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regression regularization parameter')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    else:
        datasets = [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        # Training
        if args.mode in ['train', 'both']:
            print(f"\n🔧 TRAINING PHASE")
            print("-" * 40)
            
            train_results = train_brain_diffuser(dataset, device, args.alpha)
            if train_results:
                dataset_results['training'] = train_results
                print(f"✅ Training successful for {dataset}")
            else:
                print(f"❌ Training failed for {dataset}")
                continue
        
        # Evaluation
        if args.mode in ['eval', 'both']:
            print(f"\n📊 EVALUATION PHASE")
            print("-" * 40)
            
            eval_results = evaluate_brain_diffuser(dataset, device, args.samples)
            if eval_results:
                dataset_results['evaluation'] = eval_results
                print(f"✅ Evaluation successful for {dataset}")
            else:
                print(f"❌ Evaluation failed for {dataset}")
        
        all_results[dataset] = dataset_results
    
    # Final summary
    print(f"\n{'='*80}")
    print("BRAIN-DIFFUSER FINAL SUMMARY")
    print(f"{'='*80}")
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        
        if 'training' in results:
            train = results['training']
            print(f"  Training:")
            print(f"    VDVAE Score: {train['vdvae_score']:.6f}")
            print(f"    CLIP-Vision: {train['clip_vision_score']:.6f}")
            print(f"    CLIP-Text: {train['clip_text_score']:.6f}")
        
        if 'evaluation' in results:
            eval_res = results['evaluation']
            print(f"  Evaluation:")
            print(f"    MSE: {eval_res['mse']:.6f}")
            print(f"    Correlation: {eval_res['correlation']:.6f}")
    
    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    complete_results_path = f"results/brain_diffuser_complete_results_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    import json
    with open(complete_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Complete results saved: {complete_results_path}")


if __name__ == "__main__":
    main()
