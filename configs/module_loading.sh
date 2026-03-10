#!/bin/bash
# Standard module loading for BigRed200 GPU work
# Use: source configs/module_loading.sh

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true
module load nccl/2.27.7-1 2>/dev/null || true

# Verify
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
