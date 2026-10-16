#!/bin/bash
set -euo pipefail

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hpc.txt

if ! python -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "CUDA PyTorch not detected in system packages. Installing PyTorch with CUDA 12.4 via pip..."
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
