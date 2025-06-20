# Brain-Diffuser Implementation

## Original Paper Citation
```
Ozcelik, F., VanRullen, R. Natural scene reconstruction from fMRI signals using generative latent diffusion.
Sci Rep 13, 15666 (2023). https://doi.org/10.1038/s41598-023-42891-8
```

## Paper Details
- **Title**: Natural scene reconstruction from fMRI signals using generative latent diffusion
- **Authors**: Furkan Ozcelik, Rufin VanRullen
- **Published**: Scientific Reports, September 20, 2023
- **Dataset**: Natural Scenes Dataset (NSD) - 4 subjects, COCO images
- **Architecture**: Two-stage framework with VDVAE + Versatile Diffusion

## Academic Integrity Statement
This implementation follows the exact methodology described in the original Brain-Diffuser paper without any modifications or optimizations.

## Implementation Status
- [x] Stage 1: VDVAE implementation
- [x] Stage 2: Versatile Diffusion implementation
- [x] fMRI → VDVAE latent regression
- [x] fMRI → CLIP features regression
- [x] Image-to-image diffusion pipeline
- [x] Training and evaluation scripts
- [ ] Tested on same datasets as CCCV1
- [ ] Results validated

## Files Structure
```
brain_diffuser/
├── src/
│   ├── model.py           # Original Brain-Diffuser architecture
│   ├── train.py           # Original training procedure
│   ├── evaluate.py        # Original evaluation code
│   └── utils.py           # Utility functions
├── configs/
│   └── brain_diffuser_config.yaml  # Original hyperparameters
├── models/
│   └── [trained models will be saved here]
└── results/
    └── [evaluation results will be saved here]
```

## Architecture Details

### Stage 1: VDVAE (Very Deep Variational Autoencoder)
- **Purpose**: Generate 64×64 initial guess images
- **Model**: Pretrained on ImageNet (64×64)
- **Layers**: Uses first 31 of 75 total layers
- **Latent Dim**: 91,168 (concatenated from 31 layers)
- **Regression**: Ridge regression (fMRI → VDVAE latents)

### Stage 2: Versatile Diffusion
- **Purpose**: Generate 512×512 final images
- **Model**: Pretrained on Laion2B-en
- **CLIP Model**: ViT-L/14
- **Features**:
  - Vision: 257×768 dim (patches)
  - Text: 77×768 dim (tokens)
- **Pipeline**: Image-to-image with dual guidance
- **Steps**: 37 forward + 37 reverse (75% noise)

## Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# torch>=1.12.0, diffusers>=0.21.0, transformers>=4.21.0
# clip-by-openai, scikit-learn>=1.0.0
```

## Usage
```bash
# Training (single dataset)
python src/train.py --dataset miyawaki --mode train

# Training (all datasets)
python src/train.py --dataset all --mode train

# Evaluation
python src/evaluate.py --dataset miyawaki --samples 6

# Both training and evaluation
python src/train.py --dataset miyawaki --mode both
```

## Notes
- Implementation must match original paper exactly
- No additional optimizations allowed
- Same datasets as used in CCCV1 comparison
