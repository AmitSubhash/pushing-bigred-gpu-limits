#!/bin/bash
#SBATCH -J nccl_bench
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o /N/scratch/$USER/nccl_bench-%j.out
#SBATCH -e /N/scratch/$USER/nccl_bench-%j.err

set -euo pipefail

echo "=== NCCL Collectives Benchmark ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

torchrun --nproc_per_node=4 /N/scratch/$USER/nccl_bench.py

echo ""
echo "=== COMPLETE ==="
echo "Date: $(date)"
