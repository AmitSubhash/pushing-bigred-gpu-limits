#!/bin/bash
# Template: gpu partition for longer training runs (2 day max)
#SBATCH -J <job_name>
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 04:00:00
#SBATCH -o /N/scratch/$USER/%x-%j.out
#SBATCH -e /N/scratch/$USER/%x-%j.err

set -euo pipefail

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=16
export TORCH_NCCL_BLOCKING_WAIT=1

# Enable TF32 by default
export NVIDIA_TF32_OVERRIDE=1
