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

## Advanced Benchmark Results (2026-03-10)

### FlashAttention vs Math SDPA (single GPU, FP16)
| Sequence Length | Flash (ms) | Math (ms) | Flash TFLOPS | Speedup |
|----------------|-----------|-----------|-------------|---------|
| 512 | 0.080 | 2.151 | 107.7 | **27x** |
| 1K | 0.192 | 6.906 | 178.6 | **36x** |
| 2K | 0.463 | 27.72 | 296.6 | **60x** |
| 4K | 0.862 | 55.56 | 318.9 | **64x** |
| 8K | 1.669 | 114.7 | **329.5** | **69x** |

FlashAttention achieves **329 TFLOPS** at 8K seq len (vs 4.8 TFLOPS for math backend). Memory usage is constant (~0.18 GB) regardless of sequence length -- math backend uses 20 GB at 8K.

### CUDA Graphs (Inference Speedup)
| Config | No Graph (ms) | CUDA Graph (ms) | Speedup |
|--------|--------------|-----------------|---------|
| Decode batch=1, seq=1 | 3.24 | 0.86 | **3.8x** |
| Decode batch=1, seq=32 | 3.57 | 0.87 | **4.1x** |
| Prefill batch=1, seq=128 | 3.35 | 0.98 | **3.4x** |
| Prefill batch=4, seq=128 | 3.59 | 1.58 | **2.3x** |
| Prefill batch=16, seq=512 | 18.11 | 17.98 | 1.0x |

CUDA graphs give **3-4x speedup for decode** (small batch/seq) by eliminating kernel launch overhead.

### Quantization (936M param model, batch=4, seq=512)
| Config | tok/s | Speedup | Peak Mem |
|--------|-------|---------|----------|
| FP32 | 9,829 | 1.0x | 4.03 GB |
| FP16 | 112,046 | **11.4x** | 4.02 GB |
| BF16 | 113,174 | **11.5x** | 3.89 GB |
| INT8 (bnb) | 52,940 | 5.4x | 4.50 GB |
| compile+BF16 | 112,512 | 11.5x | 3.89 GB |

**BF16/FP16 give 11.5x** over FP32 on A100 tensor cores. INT8 via bitsandbytes is slower (dequant overhead) but enables larger models.

### Activation Checkpointing (FSDP+BF16, 4 GPUs)
| Model | Mode | tok/s | Mem/GPU | Mem Save |
|-------|------|-------|---------|----------|
| 218M | none | 25,657 | 7.07 GB | -- |
| 218M | selective (every 2nd) | 17,866 | 6.09 GB | 13.9% |
| 218M | full | 13,374 | 5.11 GB | 27.7% |
| 830M | none | 14,933 | 17.04 GB | -- |
| 830M | selective | 13,459 | 16.49 GB | 3.2% |

Selective checkpointing trades ~30% throughput for ~14% memory savings. Full checkpointing saves more memory at higher cost.

### Multi-GPU Model Scaling (FSDP BF16, batch=1, seq=512)
| Model | 4-GPU (tok/s) | 1-GPU (tok/s) | 1-GPU Mem |
|-------|--------------|---------------|-----------|
| ~1.3B | 15,466 | 31,836 | 39.05 GB (near limit!) |
| ~3B | 21,880 | OOM | -- |
| ~7B | OOM | OOM | -- |

FSDP enables running 3B models that don't fit on a single 40GB GPU. For 7B+, activation checkpointing or offloading needed.

## Key Findings

1. **Always use `python/gpu/3.12.5`** -- has PyTorch 2.8 vs 2.2 in 3.11
2. **TF32 is free performance**: `torch.backends.cuda.matmul.allow_tf32 = True` gives 2.8x
3. **BF16 gives 11.5x** over FP32 for inference on A100 tensor cores
4. **FSDP + BF16 is the sweet spot** for training: 4.2x over naive FP32 DDP
5. **CUDA Graphs give 3-4x** for decode-style inference (small batch)
6. **NVLink is fully utilized** at 93% theoretical bandwidth (185 GB/s bidi)
7. **FSDP enables 3B models** that OOM on single 40GB GPU
8. **FlashAttention/SDPA is built-in** -- no extra install needed
9. **Selective activation checkpointing** saves 14% memory with 30% throughput cost
10. See [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) for 13 additional SOTA techniques

## License

MIT
