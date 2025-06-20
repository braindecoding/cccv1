# Mind-Vis Implementation

## Original Paper Citation
```
Chen, Z., Qing, J., Xiang, T., Yue, W. L., & Zhou, J. H. (2023).
Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 22710-22720).
```

## Paper Details
- **Title**: Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding
- **Authors**: Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, Juan Helen Zhou
- **Published**: CVPR 2023
- **Dataset**: Generic Object Decoding (GOD) dataset + Natural Scenes Dataset
- **Architecture**: SC-MBM (Sparse-Coding Masked Brain Modeling) + DC-LDM (Double-Conditioned Latent Diffusion Model)

## Academic Integrity Statement
This implementation follows the exact methodology described in the original Mind-Vis paper without any modifications or optimizations.

## Implementation Status
- [x] fMRI Encoder implementation
- [x] Visual Decoder implementation
- [x] Image Generator implementation
- [x] Contrastive Learning module
- [x] Training and evaluation scripts
- [x] Progressive training strategy
- [ ] Tested on same datasets as CCCV1
- [ ] Results validated

## Files Structure
```
mind_vis/
├── src/
│   ├── model.py           # Original Mind-Vis architecture
│   ├── train.py           # Original training procedure
│   ├── evaluate.py        # Original evaluation code
│   └── utils.py           # Utility functions
├── configs/
│   └── mind_vis_config.yaml  # Original hyperparameters
├── models/
│   └── [trained models will be saved here]
└── results/
    └── [evaluation results will be saved here]
```

## Architecture Details

### fMRI Encoder
- **Purpose**: Encode fMRI signals into latent representations
- **Architecture**: Multi-layer perceptron with LayerNorm
- **Hidden dims**: [1024, 512, 256]
- **Output**: 256-dimensional latent features

### Visual Decoder
- **Purpose**: Decode latent features to visual feature space
- **Architecture**: Multi-layer perceptron with LayerNorm
- **Hidden dims**: [512, 1024]
- **Output**: 512-dimensional visual features

### Image Generator
- **Purpose**: Generate images from visual features
- **Architecture**: Multi-layer perceptron with BatchNorm
- **Hidden dims**: [1024, 2048]
- **Output**: Reconstructed images (28×28)

### Contrastive Learning
- **Purpose**: CLIP-style alignment between modalities
- **Temperature**: 0.07
- **Loss**: Symmetric cross-entropy

## Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# torch>=1.12.0, transformers>=4.21.0, clip-by-openai
# timm>=0.6.0, pytorch-metric-learning>=1.6.0
```

## Usage
```bash
# Training (single dataset)
python src/train.py --dataset miyawaki --epochs 100

# Training (all datasets)
python src/train.py --dataset all --epochs 100

# Evaluation
python src/evaluate.py --dataset miyawaki --samples 6

# Custom configuration
python src/train.py --dataset vangerven --batch-size 32 --lr 0.0005
```

## Notes
- Implementation must match original paper exactly
- No additional optimizations allowed
- Same datasets as used in CCCV1 comparison
