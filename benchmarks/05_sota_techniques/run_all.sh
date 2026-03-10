#!/bin/bash
#SBATCH -J sota_bench
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 00:55:00
#SBATCH -o /N/scratch/atsubhas/sota_bench-%j.out
#SBATCH -e /N/scratch/atsubhas/sota_bench-%j.err

set -euo pipefail

echo "=== SOTA GPU Optimization Benchmarks ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
module load python/gpu/3.12.5 2>/dev/null || true

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL

BD=/N/scratch/atsubhas/sota_benchmarks

echo ""
echo "=========================================="
echo "TEST 1: FlexAttention vs SDPA"
echo "=========================================="
python3 "$BD/flex_attention_bench.py" || echo "TEST 1 FAILED"

echo ""
echo "=========================================="
echo "TEST 2: torch.compile Regional vs Whole"
echo "=========================================="
python3 "$BD/compile_regional_bench.py" || echo "TEST 2 FAILED"

echo ""
echo "=========================================="
echo "TEST 3: Liger-Kernel Fused Ops"
echo "=========================================="
python3 "$BD/liger_kernel_bench.py" || echo "TEST 3 FAILED"

echo ""
echo "=========================================="
echo "TEST 4: GaLore Optimizer"
echo "=========================================="
python3 "$BD/galore_bench.py" || echo "TEST 4 FAILED"

echo ""
echo "=========================================="
echo "TEST 5: N-gram Speculative Decoding"
echo "=========================================="
python3 "$BD/ngram_spec_decode_bench.py" || echo "TEST 5 FAILED"

echo ""
echo "=========================================="
echo "TEST 6: QLoRA vs Full BF16"
echo "=========================================="
python3 "$BD/qlora_bench.py" || echo "TEST 6 FAILED"

echo ""
echo "=========================================="
echo "TEST 7: NCCL Tuning (default)"
echo "=========================================="
torchrun --nproc_per_node=4 "$BD/nccl_tuning_bench.py" || echo "TEST 7a FAILED"

echo ""
echo "-- NCCL Ring + LL128 --"
export NCCL_ALGO=Ring
export NCCL_PROTO=LL128
torchrun --nproc_per_node=4 "$BD/nccl_tuning_bench.py" || echo "TEST 7b FAILED"

echo ""
echo "-- NCCL Tree + LL128 --"
export NCCL_ALGO=Tree
export NCCL_PROTO=LL128
torchrun --nproc_per_node=4 "$BD/nccl_tuning_bench.py" || echo "TEST 7c FAILED"

echo ""
echo "-- NCCL Ring + Simple --"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
torchrun --nproc_per_node=4 "$BD/nccl_tuning_bench.py" || echo "TEST 7d FAILED"
unset NCCL_ALGO NCCL_PROTO

echo ""
echo "=========================================="
echo "TEST 8: Communication-Computation Overlap"
echo "=========================================="
torchrun --nproc_per_node=4 "$BD/comm_overlap_bench.py" || echo "TEST 8a FAILED"

echo ""
echo "-- With CUDA_DEVICE_MAX_CONNECTIONS=1 --"
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=4 "$BD/comm_overlap_bench.py" || echo "TEST 8b FAILED"
unset CUDA_DEVICE_MAX_CONNECTIONS

echo ""
echo "=========================================="
echo "TEST 9: Optimizer Memory (FSDP)"
echo "=========================================="
torchrun --nproc_per_node=4 "$BD/optimizer_memory_bench.py" || echo "TEST 9 FAILED"

echo ""
echo "=========================================="
echo "TEST 10: Context Parallelism"
echo "=========================================="
torchrun --nproc_per_node=4 "$BD/context_parallel_bench.py" || echo "TEST 10 FAILED"

echo ""
echo "=== ALL SOTA BENCHMARKS COMPLETE ==="
echo "Date: $(date)"
