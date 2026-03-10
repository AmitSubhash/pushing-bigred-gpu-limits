#!/bin/bash
# Template: gpu-debug partition (instant allocation, 1 hour max)
#SBATCH -J <job_name>
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o /N/scratch/$USER/%x-%j.out
#SBATCH -e /N/scratch/$USER/%x-%j.err

set -euo pipefail

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
