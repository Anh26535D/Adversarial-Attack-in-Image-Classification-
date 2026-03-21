# Adversarial Training Research on CIFAR-10 (Structured Version)

## Overview
This project focuses on researching adversarial training techniques for image classification using CIFAR-10 and the `resnetv2_50x1_bit` model. The codebase is organized into a modular structure for better scalability and readability.

## Project Structure
```text
image_classification_adversarial_training/
├── src/
│   ├── models/        # Custom model architectures
│   ├── attacks/       # Adversarial attack implementations (FGSM, PGD, BIM)
│   ├── defenses/      # Post-processing defense methods
│   ├── utils/         # Data loaders and general utilities
│   ├── train.py       # Main training script (standard & adversarial)
│   └── evaluate.py    # Robustness evaluation script
├── notebooks/         # Jupyter notebooks for experiments and Colab
├── checkpoints/       # Saved model weights (.pth)
├── results/           # Evaluation metrics and reports
├── README.md
└── requirements.txt
```

## Getting Started
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model**:
   ```bash
   python src/train.py
   ```
3. **Evaluate robustness**:
   ```bash
   python src/evaluate.py
   ```

## Key Components
- **`src/attacks/base_attacks.py`**: Implementation of FGSM and PGD.
- **`src/defenses/post_processing.py`**: Defenses like Gaussian Smoothing and Bit Depth Reduction.
- **`src/models/resnet_bit.py`**: Custom ResNetV2-BiT class optimized for CIFAR-10.

## Research Objectives
- Analyze the impact of adversarial training on model robustness.
- Evaluate the effectiveness of simple post-processing defenses.
- Document results across different attack parameters (epsilon, alpha).
