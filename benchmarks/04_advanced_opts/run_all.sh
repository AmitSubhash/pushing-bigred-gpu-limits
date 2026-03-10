#!/bin/bash
#SBATCH -J adv_gpu_bench
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 00:55:00
#SBATCH -o /N/scratch/$USER/adv_gpu_bench-%j.out
#SBATCH -e /N/scratch/$USER/adv_gpu_bench-%j.err

set -euo pipefail

echo "=== Advanced GPU Optimization Benchmarks ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "=========================================="
echo "TEST 1: SDPA / FlashAttention Backends"
echo "=========================================="
python3 "$BENCH_DIR/sdpa_flash_bench.py"

echo ""
echo "=========================================="
echo "TEST 2: CUDA Graphs for Inference"
echo "=========================================="
python3 "$BENCH_DIR/cuda_graphs_inference.py"

echo ""
echo "=========================================="
echo "TEST 3: Quantization (FP32/FP16/BF16/INT8)"
echo "=========================================="
python3 "$BENCH_DIR/quantization_bench.py"

echo ""
echo "=========================================="
echo "TEST 4: Activation Checkpointing (needs torchrun)"
echo "=========================================="
torchrun --nproc_per_node=4 "$BENCH_DIR/activation_checkpoint_bench.py"

echo ""
echo "=========================================="
echo "TEST 5: Tensor Parallelism Scaling"
echo "=========================================="
torchrun --nproc_per_node=4 "$BENCH_DIR/tp_inference_bench.py"

echo ""
echo "=== ALL BENCHMARKS COMPLETE ==="
echo "Date: $(date)"
