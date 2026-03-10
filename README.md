# Pushing BigRed200 GPU Limits

Benchmarks, optimizations, and experiments for squeezing maximum performance out of Indiana University's BigRed200 GPU nodes for LLM training and inference.

## Hardware

| Spec | Value |
|------|-------|
| Node | HPE Cray EX235n |
| GPUs | 4x NVIDIA A100-SXM4-40GB |
| Intra-node | NVLink 3.0, NV4 full mesh (12 links/GPU, 25 GB/s each) |
| Inter-node | Slingshot-10 (200 Gbps) |
| CPU | AMD EPYC 7713 64C 2GHz (Milan) |
| Driver | 565.57.01, CUDA 12.7 |

## Verified Performance (2026-03-10)

### P2P NVLink Bandwidth
- **93.5 GB/s unidirectional** / **185 GB/s bidirectional** per GPU pair (93% of theoretical)
- Latency: **2.2 us** (vs 12-53 us over PCIe) -- 6-24x improvement

### NCCL Collectives (4 GPUs, NVLink)
| Operation | Peak Bus BW (GB/s) |
|-----------|--------------------|
| AllReduce | 196.7 |
| Broadcast | 210.2 |
| ReduceScatter | 139.7 |
| AllGather | 35.0 |

### Training Throughput (218M param transformer, batch=8, seq=512)
| Config | tok/s | Speedup | Peak Mem |
|--------|-------|---------|----------|
| DDP + FP32 | 12,740 | 1.0x | 8.26 GB |
| DDP + TF32 | 35,875 | 2.8x | 8.25 GB |
| DDP + BF16 | 38,897 | 3.1x | 6.89 GB |
| **FSDP + BF16** | **52,967** | **4.2x** | **5.40 GB** |

## Software Stack

Best environment: `module load python/gpu/3.12.5`

| Package | Version |
|---------|---------|
| PyTorch | 2.8.0+cu126 |
| Triton | 3.4.0 |
| NCCL | 2.27.3 |
| Transformers | 4.57.3 |
| Accelerate | 1.12.0 |
| bitsandbytes | 0.49.1 |
| FlashAttention/SDPA | Native (PyTorch) |

Additional via `nvhpc/25.7`: NVSHMEM, cuBLASmp, cufftMp

## Repository Structure

```
benchmarks/
  01_p2p_topology/       # NVLink P2P bandwidth & latency
  02_nccl_collectives/   # AllReduce, AllGather, ReduceScatter, Broadcast
  03_training_baselines/ # DDP vs FSDP, FP32/TF32/BF16
  04_advanced_opts/      # Flash attention, CUDA graphs, quantization, etc.
results/
  raw/                   # Raw SLURM job outputs
configs/
  slurm/                 # Reusable SLURM job templates
  module_loading.sh      # Standard module setup
docs/
  HARDWARE.md            # Detailed hardware topology
  OPTIMIZATIONS.md       # Optimization techniques & results
```

## Quick Start

```bash
# On BigRed200
module load python/gpu/3.12.5 cudatoolkit/12.6

# Run P2P topology test
sbatch benchmarks/01_p2p_topology/run.sh

# Run NCCL benchmark
sbatch benchmarks/02_nccl_collectives/run.sh

# Run training benchmark
sbatch benchmarks/03_training_baselines/run.sh
```

## Key Findings

1. **Always use `python/gpu/3.12.5`** -- has PyTorch 2.8 vs 2.2 in 3.11
2. **TF32 is free performance**: `torch.backends.cuda.matmul.allow_tf32 = True` gives 2.8x
3. **FSDP + BF16 is the sweet spot** for training: 4.2x over naive FP32 DDP
4. **NVLink is fully utilized** at 93% theoretical bandwidth
5. **FlashAttention/SDPA is built-in** -- no extra install needed

## License

MIT
