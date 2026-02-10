#!/bin/bash
# Training script for DeepFilterNet-SE
# Handles cuDNN compatibility and environment setup

set -e

cd /home/ubuntu/Noise-Cancellation

# Fix cuDNN version mismatch - use PyTorch bundled version
unset LD_LIBRARY_PATH

# Set Python path for imports
export PYTHONPATH=/home/ubuntu/Noise-Cancellation

# Training parameters
EPOCHS=${1:-100}
BATCH_SIZE=${2:-64}
LR=${3:-0.001}

echo "=============================================="
echo "DeepFilterNet-SE Training"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Data: data/generated/"
echo "=============================================="

# Run training
python3 training/train.py \
    --pregenerated-dir data/generated \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee training.log

echo "Training complete!"
