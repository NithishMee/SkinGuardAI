#!/bin/bash
# ==============================================================================
# SkinGuardAI: High-Performance Retraining Script for College HPC (H200 NVL)
# Targets: >95% Validation Accuracy with Minority Class Boosting & High-Res 384x384
# ==============================================================================
set -euo pipefail

# Activate Python environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=============================================================================="
echo " Starting SkinGuardAI Retraining with 384x384 & Balanced Sampler"
echo "=============================================================================="

# If previous checkpoints exist and user wants a fresh clean start:
# If you want to start fresh from pretrained ImageNet weights (recommended):
# remove or backup checkpoints/medfusionnet_last.pt

python train.py \
  --epochs 50 \
  --batch-size 16 \
  --image-size 384 \
  --vit-type swin \
  --optimizer adamw \
  --scheduler cosine \
  --warmup-epochs 3 \
  --learning-rate 2e-4 \
  --backbone-lr 2e-5 \
  --weight-decay 1e-2 \
  --loss focal \
  --focal-gamma 2.0 \
  --label-smoothing 0.05 \
  --balance balanced \
  --mixup 0.20 \
  --tta \
  --best-metric accuracy \
  --patience 10

echo "=============================================================================="
echo " Training Complete! Evaluating Best Model..."
echo "=============================================================================="
python evaluate.py --checkpoint checkpoints/medfusionnet_best.pt

echo "=============================================================================="
echo " Creating downloadable zip bundle..."
echo "=============================================================================="
zip -r medfusionnet_retrained_bundle.zip checkpoints/medfusionnet_best.pt outputs/
chmod 644 medfusionnet_retrained_bundle.zip
echo "Done! You can now download 'medfusionnet_retrained_bundle.zip' directly to your PC."
