#!/bin/bash

echo "Starting environment setup for Google Colab..."

# Install python dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints results data

# Verify installation
python -c "import torch; import timm; print('Setup successful! PyTorch version:', torch.__version__)"

echo "Environment is ready for experiments."
