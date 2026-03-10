#!/bin/bash
#SBATCH -J fsdp_bench
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 00:45:00
#SBATCH -o /N/scratch/$USER/fsdp_bench-%j.out
#SBATCH -e /N/scratch/$USER/fsdp_bench-%j.err

set -euo pipefail

echo "=== FSDP/DDP Training Benchmark ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=1

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "BF16: $(python3 -c 'import torch; torch.cuda.set_device(0); print(torch.cuda.is_bf16_supported())' 2>/dev/null || echo 'check on GPU node')"
echo ""

torchrun --nproc_per_node=4 /N/scratch/$USER/fsdp_bench.py

echo ""
echo "=== COMPLETE ==="
echo "Date: $(date)"
