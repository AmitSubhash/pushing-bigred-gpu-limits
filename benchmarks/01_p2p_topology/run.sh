#!/bin/bash
#SBATCH -J gpu_topo_bench
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o /N/scratch/$USER/gpu_topo_bench-%j.out
#SBATCH -e /N/scratch/$USER/gpu_topo_bench-%j.err

set -euo pipefail

echo "=== GPU TOPOLOGY & P2P BENCHMARK ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# ---- Module setup ----
module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || true
echo "Modules loaded."
echo ""

# ---- 1. GPU Info ----
echo "=========================================="
echo "1. NVIDIA-SMI GPU INFO"
echo "=========================================="
nvidia-smi
echo ""

# ---- 2. GPU Topology ----
echo "=========================================="
echo "2. GPU TOPOLOGY MATRIX (nvidia-smi topo -m)"
echo "=========================================="
nvidia-smi topo -m
echo ""

# ---- 3. NVLink Status ----
echo "=========================================="
echo "3. NVLINK STATUS"
echo "=========================================="
nvidia-smi nvlink -s 2>/dev/null || echo "nvlink query not supported on this driver"
echo ""
for i in 0 1 2 3; do
    echo "--- GPU $i NVLink connections ---"
    nvidia-smi nvlink -s -i $i 2>/dev/null || echo "  N/A"
done
echo ""

# ---- 4. Build p2pBandwidthLatencyTest ----
echo "=========================================="
echo "4. BUILDING p2pBandwidthLatencyTest"
echo "=========================================="
CUDA_HOME=/N/soft/sles15sp6/cuda/gnu/12.6
P2P_DIR=/N/scratch/$USER/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
cd "$P2P_DIR"

nvcc -O3 -std=c++11 \
    -I/N/scratch/$USER/cuda-samples/Common \
    -gencode arch=compute_80,code=sm_80 \
    -o p2pBandwidthLatencyTest \
    p2pBandwidthLatencyTest.cu \
    -lcudart 2>&1

echo "Build complete."
echo ""

# ---- 5. Run p2pBandwidthLatencyTest ----
echo "=========================================="
echo "5. P2P BANDWIDTH & LATENCY TEST"
echo "=========================================="
./p2pBandwidthLatencyTest
echo ""

# ---- 6. cudaMemcpy baseline (host<->device) ----
echo "=========================================="
echo "6. BANDWIDTH TEST (Host <-> Device)"
echo "=========================================="
BWTEST_DIR=/N/scratch/$USER/cuda-samples/Samples/1_Utilities/bandwidthTest
if [ -d "$BWTEST_DIR" ]; then
    cd "$BWTEST_DIR"
    nvcc -O3 -std=c++11 \
        -I/N/scratch/$USER/cuda-samples/Common \
        -gencode arch=compute_80,code=sm_80 \
        -o bandwidthTest \
        bandwidthTest.cu \
        -lcudart 2>&1
    ./bandwidthTest --dtoh --htod --device=all
else
    echo "bandwidthTest sample not found, skipping."
fi
echo ""

echo "=== BENCHMARK COMPLETE ==="
echo "Date: $(date)"
