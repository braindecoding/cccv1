# SOTA Methods Methodology Comparison

## Academic Integrity Protocol

This document outlines the exact methodologies from original papers to ensure fair and accurate comparison.

## 1. Brain-Diffuser (Ozcelik & VanRullen, 2023)

### Architecture
- **Stage 1**: VDVAE (Very Deep Variational Autoencoder)
  - 75 layers, uses first 31 layers for regression
  - 91,168-dim concatenated latent vectors
  - Ridge regression: fMRI → VDVAE latents
  - Generates 64×64 "initial guess" images

- **Stage 2**: Versatile Diffusion
  - Two regression models:
    - fMRI → CLIP-Vision features (257×768-dim)
    - fMRI → CLIP-Text features (77×768-dim)
  - Image-to-image pipeline with dual guidance
  - 37 steps forward diffusion (75% of 50 steps)
  - Final output: 512×512 images

### Training Details
- **Dataset**: Natural Scenes Dataset (NSD)
- **Subjects**: 4 subjects (sub1, sub2, sub5, sub7)
- **Training**: 8,859 images, 24,980 fMRI trials
- **Testing**: 982 images, 2,770 fMRI trials
- **ROI**: NSDGeneral ROI mask (visual areas)
- **Preprocessing**: Single-trial beta weights with GLMDenoise

### Key Parameters
- VDVAE: Pretrained on 64×64 ImageNet
- Versatile Diffusion: Pretrained on Laion2B-en, 512×512
- CLIP guidance weights: Vision 0.6, Text 0.4
- Ridge regression for all fMRI-to-latent mappings

## 2. Mind-Vis (Chen et al., 2023)

### Architecture
- **Stage A**: SC-MBM (Sparse-Coding Masked Brain Modeling)
  - Self-supervised pre-training on large-scale fMRI
  - Sparse coding approach for brain representation learning
  - Masked modeling strategy

- **Stage B**: DC-LDM (Double-Conditioned Latent Diffusion Model)
  - Conditional diffusion model
  - Double conditioning mechanism
  - Vision decoding from brain recordings

### Training Details
- **Dataset**: Generic Object Decoding (GOD) + Natural Scenes Dataset
- **Pre-training**: Large-scale resting-state fMRI dataset
- **Limited annotations**: Few training pairs approach
- **Performance**: 23.9% top-1 accuracy (100-way classification)

### Key Features
- First to show non-invasive brain recordings comparable to invasive measures
- State-of-the-art FID score: 1.67
- Outperformed previous methods by 66% (semantic) and 41% (generation quality)

## 3. CCCV1 (Our Method - Baseline Version)

### Architecture
- **Core**: CortexFlowCLIPCNNV1Optimized
- **Components**:
  - CNN encoder for fMRI feature extraction
  - CLIP integration for multimodal features
  - Decoder for image reconstruction

### Training Details
- **Dataset**: miyawaki, vangerven, crell, mindbigdata
- **Cross-Validation**: 10-fold CV
- **Optimization**: Adam optimizer with specific hyperparameters
- **Device**: CUDA-enabled GPU training

### Baseline Configuration
For fair comparison, we will use:
- Standard hyperparameters (no optimizations)
- Basic training procedures
- Same evaluation metrics as other methods
- Same experimental setup

## Comparison Protocol

### Datasets
- **Primary**: Natural Scenes Dataset (for Brain-Diffuser comparison)
- **Secondary**: Generic Object Decoding (for Mind-Vis comparison)
- **Our datasets**: miyawaki, vangerven, crell, mindbigdata

### Evaluation Metrics
1. **Semantic Metrics**:
   - Top-1 classification accuracy
   - Top-5 classification accuracy
   - Semantic similarity scores

2. **Generation Quality**:
   - FID (Fréchet Inception Distance)
   - SSIM (Structural Similarity Index)
   - LPIPS (Learned Perceptual Image Patch Similarity)

3. **Low-level Metrics**:
   - MSE (Mean Squared Error)
   - Correlation coefficients
   - Pixel-level accuracy

### Implementation Requirements

#### Brain-Diffuser
- [ ] Implement VDVAE stage exactly as described
- [ ] Implement Versatile Diffusion stage with correct parameters
- [ ] Use same NSD dataset preprocessing
- [ ] Ridge regression for all mappings
- [ ] No modifications to original architecture

#### Mind-Vis
- [ ] Implement SC-MBM pre-training stage
- [ ] Implement DC-LDM generation stage
- [ ] Use same GOD dataset preprocessing
- [ ] Sparse coding approach for brain modeling
- [ ] No modifications to original architecture

#### CCCV1 Baseline
- [ ] Remove all optimizations from current CCCV1
- [ ] Use standard hyperparameters
- [ ] Implement basic training procedures
- [ ] Ensure fair comparison baseline

### Academic Integrity Checklist
- [ ] All implementations follow original papers exactly
- [ ] No unauthorized optimizations or modifications
- [ ] Same experimental setup across all methods
- [ ] Proper citations and acknowledgments
- [ ] Reproducible code with clear documentation
- [ ] Statistical significance testing for results
