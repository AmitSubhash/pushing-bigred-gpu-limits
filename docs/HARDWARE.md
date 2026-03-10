# BigRed200 GPU Node Hardware Details

## Node Architecture
- **Platform:** HPE Cray EX235n blade
- **CPU:** AMD EPYC 7713 64C 2GHz (Milan, Zen 3)
- **GPUs:** 4x NVIDIA A100-SXM4-40GB
- **System interconnect:** Slingshot-10 (200 Gbps)
- **Driver:** 565.57.01
- **CUDA (driver):** 12.7
- **CUDA (toolkit):** 12.6

## GPU Topology

```
nvidia-smi topo -m output:

        GPU0  GPU1  GPU2  GPU3  NIC0  NIC1
GPU0     X    NV4   NV4   NV4   NODE  PHB
GPU1    NV4    X    NV4   NV4   NODE  NODE
GPU2    NV4   NV4    X    NV4   NODE  NODE
GPU3    NV4   NV4   NV4    X    PHB   NODE

NIC0: mlx5_0 (Mellanox ConnectX InfiniBand)
NIC1: mlx5_1 (Mellanox ConnectX InfiniBand)
```

**NV4** = 4 bonded NVLink 3.0 connections between each GPU pair.

### NVLink Configuration
- Each GPU has **12 NVLink 3.0 links** at 25 GB/s per direction each
- 12 links distributed across 3 peer GPUs = **4 links per peer**
- Per-pair bandwidth: 4 x 25 = **100 GB/s per direction** (200 GB/s bidirectional theoretical)
- Per-GPU total: 12 x 25 = **300 GB/s per direction** (600 GB/s bidirectional theoretical)
- **Full mesh topology** -- every GPU directly connected to every other GPU

### Measured vs Theoretical

| Metric | Measured | Theoretical | Efficiency |
|--------|----------|-------------|------------|
| Unidirectional per pair | 93.5 GB/s | 100 GB/s | 93.5% |
| Bidirectional per pair | 185 GB/s | 200 GB/s | 92.5% |
| P2P latency (NVLink) | 2.2 us | -- | -- |
| P2P latency (PCIe) | 12-53 us | -- | -- |
| Local GPU memory BW | ~1,350 GB/s | 1,555 GB/s | ~87% |

### NIC Affinity
- GPU3 is PHB (PCIe Host Bridge) to NIC0 (mlx5_0) -- best for inter-node GPU3 transfers
- GPU0 is PHB to NIC1 (mlx5_1)
- Other GPU-NIC pairs traverse NUMA node interconnect (NODE)

## Compute Capabilities
- Compute capability: 8.0 (Ampere)
- Tensor Cores: 3rd gen (FP64, TF32, BF16, FP16, INT8, INT4)
- TF32: 156 TFLOPS per GPU
- BF16: 312 TFLOPS per GPU (with sparsity)
- FP32: 19.5 TFLOPS per GPU
- MIG: Supported (up to 7 instances)
- Memory: 40 GB HBM2e per GPU, 1,555 GB/s bandwidth

## Available Software (via modules)

### Best Environment: `python/gpu/3.12.5`
| Package | Version |
|---------|---------|
| PyTorch | 2.8.0+cu126 |
| Triton | 3.4.0 |
| NCCL | 2.27.3 |
| Transformers | 4.57.3 |
| Accelerate | 1.12.0 |
| bitsandbytes | 0.49.1 |
| torch-geometric | 2.7.0 |
| FlashAttention/SDPA | Native in PyTorch 2.8 |

### Alternative: `python/gpu/3.11.5`
| Package | Version |
|---------|---------|
| PyTorch | 2.2.0+cu118 |
| Triton | 2.2.0 |
| NCCL | 2.19.3 |
| Accelerate | 1.10.0 |

### NVIDIA HPC SDK: `nvhpc/25.7`
- NVSHMEM 3.2.5 (GPU-initiated RDMA)
- cuBLASmp (multi-process BLAS)
- cufftMp (multi-process FFT)
- NCCL (bundled)
- HPC-X MPI

### Other Notable Modules
- `hpc_llm/gpu/1.0` -- llama.cpp with CUDA, pre-downloaded models
- `nccl/2.27.7-1` -- standalone NCCL
- `cudnn/9.10.1.4_cuda12` -- cuDNN 9.10
