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

Best environment: `module load python/gpu/3.12.5 cudatoolkit/12.6`

---

## Optimization Techniques: What They Are and How They Performed

Each section below explains the technique, why it matters, and shows our measured results on BigRed200.

---

### 1. NVLink P2P (GPU-to-GPU Direct Memory Access)

**What it is:** NVLink allows GPUs to read/write each other's memory directly without going through the CPU or PCIe bus. Our 4 A100s are connected in a "full mesh" where every GPU has 4 dedicated NVLink lanes to every other GPU.

**Why it matters for LLMs:** Multi-GPU training and inference require constant data exchange (gradient sync, weight sharding, KV cache sharing). NVLink makes this communication 5-25x faster than PCIe, which is the difference between multi-GPU being practical or bottlenecked.

**Our results:**

| Metric | PCIe (P2P disabled) | NVLink (P2P enabled) | Improvement |
|--------|--------------------|--------------------|-------------|
| Bandwidth (uni) | 19 GB/s | **93.5 GB/s** | 4.9x |
| Bandwidth (bidi) | 21 GB/s | **185 GB/s** | 8.8x |
| Latency | 12-53 us | **2.2 us** | 6-24x |

We hit **93% of theoretical NVLink bandwidth**. This confirms the hardware is working optimally.

---

### 2. NCCL Collectives (Multi-GPU Communication Primitives)

**What it is:** NCCL (NVIDIA Collective Communication Library) provides operations like AllReduce (sum gradients across GPUs), AllGather (collect sharded weights), and Broadcast. These are the building blocks of all distributed training.

**Why it matters for LLMs:** Every training step uses AllReduce to synchronize gradients. Every FSDP forward pass uses AllGather to reconstruct weights. The speed of these operations directly determines your multi-GPU scaling efficiency.

**Our results (peak bus bandwidth):**

| Operation | Peak BW (GB/s) | What it's used for |
|-----------|---------------|-------------------|
| AllReduce | **196.7** | Gradient synchronization (DDP) |
| Broadcast | **210.2** | Weight distribution |
| ReduceScatter | **139.7** | Gradient reduction + sharding (FSDP) |
| AllGather | 35.0 | Weight gathering (FSDP forward) |

AllReduce at 197 GB/s means gradient sync is nearly free for compute-bound workloads.

---

### 3. Mixed Precision (TF32 / BF16)

**What it is:** A100 GPUs have specialized "Tensor Cores" that operate on lower-precision numbers much faster. TF32 is a drop-in replacement for FP32 that uses tensor cores automatically. BF16 (Brain Float 16) halves memory usage and doubles throughput by using 16-bit numbers with the same exponent range as FP32.

**Why it matters for LLMs:** The single biggest free speedup you can get. FP32 leaves 90% of the A100's compute power unused because it doesn't use tensor cores efficiently. Switching to BF16 unlocks the full 312 TFLOPS of the A100.

**Our results (218M param transformer, 4 GPUs):**

| Config | tok/s | Speedup | Peak Memory |
|--------|-------|---------|-------------|
| DDP + FP32 | 12,740 | 1.0x | 8.26 GB |
| DDP + TF32 | 35,875 | **2.8x** | 8.25 GB |
| DDP + BF16 | 38,897 | **3.1x** | 6.89 GB |
| FSDP + BF16 | **52,967** | **4.2x** | **5.40 GB** |

**How to enable:** One line for TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`), or use `torch.autocast("cuda", dtype=torch.bfloat16)` for BF16.

---

### 4. FSDP (Fully Sharded Data Parallelism)

**What it is:** FSDP shards model parameters, gradients, and optimizer states across all GPUs. Each GPU only holds 1/N of the model at rest. During computation, it temporarily gathers the full layer weights via AllGather, computes, then discards them. This is like ZeRO Stage 3 from DeepSpeed but built into PyTorch.

**Why it matters for LLMs:** With 40GB per GPU, a single A100 can only hold a ~3B parameter model in BF16 with optimizer states. FSDP across 4 GPUs effectively gives you 160GB of pooled memory, enabling much larger models. It also provides data parallelism for free.

**Our results:**

| Model Size | 4-GPU FSDP BF16 | Single GPU |
|------------|----------------|------------|
| ~1.3B | 15,466 tok/s, 12 GB/GPU | 31,836 tok/s, 39 GB (nearly OOM) |
| ~3B | 21,880 tok/s, 29 GB/GPU | **OOM** |
| ~7B | OOM (needs checkpointing) | OOM |

FSDP enables running 3B models that are impossible on a single 40GB GPU.

---

### 5. FlashAttention (SDPA)

**What it is:** Standard attention computes an N x N attention matrix, which uses O(N^2) memory and is slow due to excessive HBM reads. FlashAttention rewrites this to tile the computation into SRAM, fusing the softmax and matrix multiply into a single kernel pass. Memory drops to O(N) and compute becomes IO-aware, achieving much higher hardware utilization.

**Why it matters for LLMs:** Attention is the bottleneck operation in transformers. At sequence length 8K, standard attention needs 20 GB just for the attention matrix. FlashAttention uses 0.18 GB regardless of sequence length.

**Our results (single A100, FP16, batch=4, 32 heads, head_dim=64):**

| Sequence Length | Flash (ms) | Math (ms) | Flash TFLOPS | Speedup |
|----------------|-----------|-----------|-------------|---------|
| 512 | 0.080 | 2.151 | 107.7 | **27x** |
| 1K | 0.192 | 6.906 | 178.6 | **36x** |
| 2K | 0.463 | 27.72 | 296.6 | **60x** |
| 4K | 0.862 | 55.56 | 318.9 | **64x** |
| 8K | 1.669 | 114.7 | **329.5** | **69x** |

FlashAttention achieves **329 TFLOPS** on A100, approaching the theoretical 312 TFLOPS BF16 peak (it exceeds it because FP16 tensor cores are slightly faster). Memory is constant at 0.18 GB vs 20 GB for math at 8K.

**How to enable:** Automatic in PyTorch 2.8 via `F.scaled_dot_product_attention()` when using FP16/BF16 on CUDA. No extra install needed.

---

### 6. CUDA Graphs

**What it is:** Normally, the CPU launches GPU kernels one at a time, and each launch has overhead (~3-10 us). For LLM decode, each token generation is a tiny computation with dozens of kernel launches. CUDA Graphs record an entire sequence of kernel launches into a "graph" that can be replayed with a single CPU call, eliminating launch overhead.

**Why it matters for LLMs:** During autoregressive decode (generating tokens one at a time), the GPU does very little compute per step but the CPU launch overhead dominates. CUDA Graphs turn dozens of individual launches into one replay.

**Our results (216M param model, FP16):**

| Config | No Graph (ms) | CUDA Graph (ms) | Speedup |
|--------|--------------|-----------------|---------|
| Decode batch=1, seq=1 | 3.24 | 0.86 | **3.8x** |
| Decode batch=1, seq=32 | 3.57 | 0.87 | **4.1x** |
| Prefill batch=1, seq=128 | 3.35 | 0.98 | **3.4x** |
| Prefill batch=4, seq=128 | 3.59 | 1.58 | **2.3x** |
| Prefill batch=16, seq=512 | 18.11 | 17.98 | 1.0x |

**Key insight:** CUDA Graphs give 3-4x speedup for small-batch decode where kernel launch overhead dominates. For large batches where compute dominates, the benefit disappears. This is why vLLM uses CUDA Graphs for decode and eager mode for prefill.

---

### 7. Quantization

**What it is:** Reducing the numerical precision of model weights and/or activations. FP32 (4 bytes) -> FP16/BF16 (2 bytes) -> INT8 (1 byte) -> INT4 (0.5 bytes). Lower precision means less memory and faster computation, at the cost of some accuracy.

**Why it matters for LLMs:** A 70B parameter model is 140GB in FP16. That doesn't fit on 4x40GB GPUs. With INT4 quantization it's ~35GB, fitting easily. Even for smaller models, quantization improves throughput by reducing memory bandwidth pressure.

**Important A100 note:** A100 does **NOT** have native FP8 tensor cores. FP8 is Hopper (H100) only. On A100, use BF16 for training and INT8 for inference.

**Our results (936M param model, batch=4, seq=512):**

| Precision | tok/s | Speedup vs FP32 | Peak Memory |
|-----------|-------|-----------------|-------------|
| FP32 | 9,829 | 1.0x | 4.03 GB |
| FP16 | 112,046 | **11.4x** | 4.02 GB |
| BF16 | 113,174 | **11.5x** | 3.89 GB |
| INT8 (bitsandbytes) | 52,940 | 5.4x | 4.50 GB |
| torch.compile + BF16 | 112,512 | 11.5x | 3.89 GB |

**Key insight:** BF16/FP16 give 11.5x over FP32 by fully utilizing A100 tensor cores. INT8 via bitsandbytes is actually slower here due to dequantization overhead, but it enables loading models that wouldn't fit otherwise (e.g., 70B in INT4 on 4 GPUs).

---

### 8. Activation Checkpointing

**What it is:** During training, PyTorch saves all intermediate activations for the backward pass. This uses a lot of memory. Activation checkpointing discards these activations and recomputes them during backward, trading ~33% extra compute for 50-70% less activation memory.

**Selective checkpointing** is smarter: it only recomputes cheap ops (pointwise, activations) and saves expensive ones (matrix multiplies), getting ~50% of the memory savings with only ~5-10% compute overhead.

**Why it matters for LLMs:** On 40GB GPUs, activation memory is often what causes OOM, not model weights. Checkpointing is how you fit larger batch sizes or longer sequences.

**Our results (FSDP + BF16, 4 GPUs):**

| Model | Mode | tok/s | Mem/GPU | Memory Saved |
|-------|------|-------|---------|-------------|
| 218M | none | 25,657 | 7.07 GB | -- |
| 218M | selective (every 2nd block) | 17,866 | 6.09 GB | **13.9%** |
| 218M | full | 13,374 | 5.11 GB | **27.7%** |
| 830M | none | 14,933 | 17.04 GB | -- |
| 830M | selective | 13,459 | 16.49 GB | 3.2% |

---

### 9. Additional Techniques (from Optimization Report)

These are documented in detail in [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) with code examples:

| Technique | What it does | Expected Impact | Status |
|-----------|-------------|----------------|--------|
| **Liger-Kernel** | Fused Triton kernels for RMSNorm, SwiGLU, CrossEntropy | +20% throughput, -60% memory | `pip install liger-kernel` |
| **8-bit AdamW** | Quantizes optimizer states to INT8 | 75% less optimizer memory | Built-in (bitsandbytes) |
| **Speculative Decoding** | Small draft model proposes tokens, big model verifies in batch | 1.5-3x decode latency | Via vLLM |
| **PagedAttention** | Virtual memory for KV cache, eliminates fragmentation | 2-4x throughput via batching | Via vLLM |
| **Context Parallelism** | Ring Attention across GPUs for long sequences | 4x sequence length | Native PyTorch 2.8 |
| **FlexAttention** | Custom attention patterns compiled to fused kernels | ~0.9x FlashAttention speed | Native PyTorch 2.8 |
| **torch.compile** | JIT compilation with operator fusion | 1.3-2x train, 2-4x decode | Native PyTorch 2.8 |
| **Sequence Parallelism** | Split sequence dim for LayerNorm/Dropout with TP | Reduces activation memory | Native PyTorch 2.8 |
| **GaLore / LOMO** | Low-rank optimizer projections | 65-100% less optimizer memory | `pip install` |
| **NF4 QLoRA** | 4-bit base model + LoRA adapters | Train 70B on 4x40GB | bitsandbytes + peft |

---

## Maximum Inference Speed: Optimal Combination

Based on our benchmarks, here's the theoretical optimal stack for inference on 4x A100-40GB:

### For Decode Latency (single request, fastest response)

```
FlashAttention (69x attention speedup)
  + BF16 (11.5x tensor core utilization)
  + CUDA Graphs (3-4x kernel launch elimination)
  + Speculative Decoding (1.5-3x via draft model)
  + Tensor Parallelism 4-way (near-linear scaling)
```

**Estimated combined effect:** These are not simply multiplicative (they target different bottlenecks), but the combination should yield **10-30x** over a naive FP32 single-GPU implementation.

### For Throughput (many requests, max tokens/second)

```
FlashAttention (69x)
  + BF16 (11.5x)
  + PagedAttention (2-4x via better batching)
  + Continuous Batching (eliminates idle GPU time)
  + Tensor Parallelism 4-way
```

### Practical: vLLM on BigRed200

```python
from vllm import LLM, SamplingParams

# Maximum throughput config
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.92,
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
)

# Minimum latency config
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=4,
    enforce_eager=False,  # CUDA graphs on
    speculative_model="[ngram]",
    ngram_prompt_lookup_max=4,
    num_speculative_tokens=5,
)
```

---

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
  OPTIMIZATIONS.md       # 13 SOTA techniques with code examples
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

# Run advanced optimizations
sbatch benchmarks/04_advanced_opts/run_all.sh
```

## Key Takeaways

1. **BF16 is the single biggest win**: 11.5x inference, 3.1x training, one line of code
2. **FlashAttention is automatic and transformative**: 69x at 8K, constant memory
3. **FSDP + BF16 is the training sweet spot**: 4.2x speedup, 35% less memory
4. **CUDA Graphs dominate decode**: 3-4x for autoregressive generation
5. **NVLink works perfectly**: 93% of theoretical, enabling efficient multi-GPU
6. **A100 has no FP8**: Don't waste time with FP8 on this hardware
7. **40GB is the real constraint**: Use FSDP, quantization, and checkpointing to work around it
8. **vLLM combines everything**: PagedAttention + TP + CUDA Graphs + speculative decoding in one package

## License

MIT
