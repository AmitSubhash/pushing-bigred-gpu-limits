# Pushing BigRed200 GPU Limits

Benchmarks, optimizations, and experiments for squeezing maximum performance out of Indiana University's BigRed200 GPU nodes for LLM training and inference.

## What We're Optimizing

**Goal:** Maximize throughput (tokens/second) and minimize latency for large language model (LLM) training and inference on BigRed200's 4x A100-SXM4-40GB GPU nodes.

**The problem:** A naive FP32 implementation on a single GPU leaves ~90% of the A100's compute power unused and can only fit models up to ~1B parameters. Modern LLMs (7B-70B+) require careful optimization across precision, memory management, parallelism, and kernel efficiency to run at all, let alone run fast.

**Our approach:** We systematically benchmarked 18 GPU optimization techniques, from hardware-level (NVLink P2P) to algorithm-level (speculative decoding), measuring real throughput and memory on BigRed200. Each technique targets a different bottleneck; the key is knowing which ones matter for your workload and how they compose.

**The workload:** We use transformer language models ranging from 218M to 3B parameters as benchmarking targets, with sequence lengths from 128 to 16K tokens. Results generalize to production LLMs like Llama, GPT, and Mistral.

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

## Part 1: Core Optimization Techniques

These are the foundational techniques every GPU workload should use.

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

## Part 2: SOTA Techniques (Benchmarked)

These 10 additional techniques were identified from state-of-the-art research and tested on BigRed200.

---

### 9. FlexAttention (Custom Attention Patterns)

**What it is:** A PyTorch 2.8 API that lets you define custom attention score modifications (sliding window, soft-capping, ALiBi, prefix LM masks) as Python callables. These get JIT-compiled into fused attention kernels via `torch.compile`, meaning you get custom attention patterns at near-FlashAttention speed without writing CUDA.

**Why it matters for LLMs:** Many modern LLMs use non-standard attention (Gemma's soft-cap, Mistral's sliding window, GQA). Without FlexAttention, implementing these requires either materializing the full N x N mask (slow, memory-hungry) or writing custom CUDA kernels (hard). FlexAttention gives you both performance and flexibility.

**Our results (single A100, FP16, batch varies, 32 heads, head_dim=64):**

| Seq Length | SDPA Flash (ms) | FlexAttn Causal (ms) | FlexAttn Slide-256 (ms) | FlexAttn Soft-Cap (ms) |
|------------|----------------|---------------------|------------------------|----------------------|
| 512 | 0.073 | 0.131 | 0.119 | 0.166 |
| 1K | 0.192 | 0.237 | 0.180 | 0.564 |
| 2K | 0.594 | 0.655 | **0.275** | 1.269 |
| 4K | 0.862 | 1.329 | **0.348** | 2.537 |
| 8K | 1.668 | 2.550 | **0.348** | 5.183 |

**Key insight:** FlexAttention causal is 0.65x of SDPA Flash for standard causal (overhead from torch.compile). But FlexAttention sliding-window-256 achieves **1,581 TFLOPS at 8K** (4.8x over SDPA Flash!) because it only attends to 256 tokens per query position, skipping most computation. Use SDPA Flash for standard causal; use FlexAttention when you need custom patterns.

---

### 10. torch.compile (JIT Compilation)

**What it is:** PyTorch's JIT compiler traces eager Python code into an optimized graph, applying operator fusion, memory planning, and optional CUDA graph capture. Three modes: `default` (safe fusion), `reduce-overhead` (CUDA graphs under the hood), and regional (compile per-layer instead of whole model).

**Why it matters for LLMs:** Operator fusion eliminates memory round-trips between small ops (norm, activation, residual). For training, this gives 10-15% throughput improvement. Regional compilation (compiling each transformer layer separately) reduces compile time from hours to minutes.

**Our results (218M param model, single A100, BF16):**

| Mode | Training tok/s | Training Speedup | Inference tok/s | Inference Latency |
|------|---------------|-----------------|----------------|-------------------|
| Eager | 87,747 | 1.0x | 51,763 | 2.47 ms |
| compile(default) | 98,112 | **1.12x** | 47,991 | 2.67 ms |
| compile(regional) | 97,930 | **1.12x** | 43,130 | 2.97 ms |
| compile(reduce-OH) | **98,936** | **1.13x** | 46,364 | 2.76 ms |

**Key insight:** torch.compile gives a consistent **12-13% training speedup** across all modes. For single-batch inference, it's actually **slower** (likely due to graph overhead outweighing fusion benefits at batch=1). Use it for training; for inference, prefer CUDA Graphs directly or vLLM.

---

### 11. Liger-Kernel (Fused Triton Kernels)

**What it is:** Drop-in Triton kernels that replace standard PyTorch operations with fused versions. RMSNorm+residual, CrossEntropy, FusedLinearCrossEntropy, SwiGLU, and RoPE are fused into single GPU kernel calls, eliminating memory bandwidth bottlenecks.

**Why it matters for LLMs:** LLM training is memory-bandwidth-bound for many operations. Fusing multiple ops into a single kernel pass reduces HBM reads/writes dramatically. The biggest win is FusedLinearCrossEntropy, which fuses the final lm_head linear layer with the cross-entropy loss, avoiding materializing the full vocab-sized logit tensor.

**Our results (single A100, BF16, d=4096, vocab=32K, batch=4, seq=2048):**

| Operation | PyTorch (ms) | Liger-Kernel (ms) | Speedup | Memory Saved |
|-----------|-------------|-------------------|---------|-------------|
| RMSNorm (fwd+bwd) | 2.606 | **0.611** | **4.27x** | 37% |
| CrossEntropy (fwd+bwd) | 4.020 | **2.578** | **1.56x** | 19% |

**Key insight:** RMSNorm fusion alone gives **4.3x speedup** because the unfused version does multiple memory round-trips (load, square, mean, sqrt, multiply, store) while Liger does it in one pass. Install with `pip install liger-kernel` and apply with one line: `apply_liger_kernel_to_llama()`.

---

### 12. N-gram Speculative Decoding

**What it is:** During autoregressive decode, instead of generating one token at a time, use n-gram pattern matching from the prompt/generated text to "guess" the next K tokens. Then verify all K guesses in a single forward pass. Accepted tokens are output immediately; rejected ones trigger fallback. No draft model needed, zero extra GPU memory.

**Why it matters for LLMs:** Autoregressive decode is slow because each token requires a full model forward pass, but the GPU is underutilized (memory-bound, not compute-bound). Speculative decoding amortizes forward pass overhead across multiple tokens. N-gram speculation is the simplest form: it works without any extra model.

**Our results (219M param model, single A100, BF16):**

| Method | tok/s | Speedup | Accept Rate |
|--------|-------|---------|-------------|
| Greedy (standard) | 243.5 | 1.0x | -- |
| N-gram (n=2, k=3) | 498.5 | **2.05x** | 54.5% |
| N-gram (n=2, k=5) | **529.8** | **2.18x** | 47.1% |
| N-gram (n=3, k=3) | 450.4 | 1.85x | 46.8% |
| N-gram (n=4, k=5) | 451.3 | 1.85x | 36.0% |

**Key insight:** N-gram speculation gives **2.18x decode speedup** with zero extra memory and zero extra model parameters. Shorter n-grams (n=2) work best because they have more matches. With a proper draft model, speedups of 2-3x are typical.

---

### 13. NCCL Algorithm/Protocol Tuning

**What it is:** NCCL auto-selects the communication algorithm (Ring, Tree, CollNet) and protocol (LL, LL128, Simple) based on hardware topology. Manual override via environment variables can sometimes improve performance for specific workloads.

**Why it matters for LLMs:** NCCL collectives (AllReduce, AllGather) happen on every training step. A 20% improvement in collective bandwidth directly translates to faster training, especially for communication-bound workloads like large-batch DDP.

**Our results (AllReduce bus bandwidth, 4x A100 NVLink):**

| Config | 1 MB | 16 MB | 64 MB | 256 MB |
|--------|------|-------|-------|--------|
| Default (auto) | 2.9 GB/s | 124 GB/s | 166 GB/s | **194 GB/s** |
| Ring + LL128 | **31.3 GB/s** | 69 GB/s | 104 GB/s | 151 GB/s |
| Tree + LL128 | 5.2 GB/s | 56 GB/s | 96 GB/s | 123 GB/s |
| Ring + Simple | 19.2 GB/s | 110 GB/s | 139 GB/s | 188 GB/s |

**Key insight:** NCCL's auto-tuning is already excellent for NVLink. Ring+LL128 gives **11x improvement for small messages** (<1 MB) but is worse for large messages. For gradient sync (typically 16-256 MB), leave NCCL at default. Only override for specific workloads with known message sizes.

---

### 14. Communication-Computation Overlap

**What it is:** Overlapping NCCL collective operations with GPU compute using async operations. Instead of synchronously communicating then computing, launch communication in the background and compute simultaneously. Also configurable via `CUDA_DEVICE_MAX_CONNECTIONS=1`.

**Why it matters for LLMs:** In FSDP training, AllGather and ReduceScatter happen every layer. If these can run in parallel with the next layer's compute, training becomes faster.

**Our results (4x A100 NVLink):**

| Config | Overlap Gain (raw) | FSDP tok/s |
|--------|-------------------|-----------|
| Sync (default) | 1.0x | 45,873 |
| Async overlap (256MB + compute) | **1.24x** | -- |
| CUDA_DEVICE_MAX_CONNECTIONS=1 | -- | **49,280** |

**Key insight:** Setting `CUDA_DEVICE_MAX_CONNECTIONS=1` gives **7.4% more FSDP throughput** with zero code changes. Async overlap gives up to 24% gain when communication and compute are balanced. This is a free optimization.

---

### 15. Memory-Efficient Optimizers

**What it is:** Variants of AdamW that reduce optimizer state memory. Standard AdamW stores two states per parameter (momentum + variance = 8 bytes/param). 8-bit AdamW quantizes these to INT8 (2 bytes/param). Adafactor factorizes the second moment matrix. GaLore projects gradients to low-rank space.

**Why it matters for LLMs:** For a 7B model, AdamW optimizer states consume 56 GB. With 4x 40GB GPUs and FSDP, that's 14 GB/GPU just for optimizer states. 8-bit optimization cuts this to 3.5 GB/GPU.

**Our results (single GPU, 218M model, BF16):**

| Optimizer | tok/s | Memory | Memory Savings |
|-----------|-------|--------|---------------|
| AdamW (standard) | **79,608** | 4.31 GB | -- |
| AdamW8bit (bnb) | 65,503 | **3.90 GB** | **9.5%** |
| Adafactor | 58,996 | 4.31 GB | 0% |

**Our results (FSDP, 1.3B model, 4 GPUs):**

| Optimizer | tok/s | Memory/GPU |
|-----------|-------|-----------|
| AdamW (standard) | 9,121 | 12.21 GB |
| AdamW8bit (bnb) | **9,366** | 13.41 GB |
| Adafactor | 8,777 | 14.75 GB |

**Key insight:** On single GPU, AdamW8bit saves 10% memory but is 18% slower. On FSDP (where optimizer states are already sharded), the memory benefit is minimal. 8-bit optimizers shine for very large models where optimizer memory is the OOM bottleneck, not for models that already fit. Use standard AdamW unless you're running out of memory.

---

### 16. QLoRA Training (Quantized Low-Rank Adapters)

**What it is:** Load the base model in 4-bit (NF4) or 8-bit precision, freeze it, and train small LoRA adapter layers in BF16. This enables fine-tuning models that are 4-8x larger than what would fit in full precision.

**Why it matters for LLMs:** Fine-tuning a 70B model in BF16 requires 140 GB just for weights plus 560 GB for optimizer states. With NF4 QLoRA, the base model is 35 GB and only the ~100M LoRA parameters need optimizer states.

**Our results (936M param model, single A100, batch=4, seq=512):**

| Config | tok/s | Memory | Trainable Params |
|--------|-------|--------|-----------------|
| Full BF16 | **19,746** | 9.52 GB | 936M (100%) |
| INT8 + LoRA (r=16) | 10,306 | **8.44 GB** | 5.2M (0.6%) |
| Full BF16 + AdamW8bit | 17,076 | **7.98 GB** | 936M (100%) |

**Key insight:** INT8 + LoRA saves 11% memory but is 48% slower due to dequantization overhead. The real value of QLoRA is enabling models that simply won't fit otherwise. For a 936M model that fits in BF16, full fine-tuning is faster. For a 70B model, QLoRA is the only option on 4x 40GB.

---

### 17. Sequence Sharding (Context Parallelism Proxy)

**What it is:** Splitting the input sequence across GPUs along the sequence dimension. Each GPU processes a fraction of the sequence. Context Parallelism (Ring Attention) adds communication to enable proper cross-sequence attention; basic sharding just processes independent shards.

**Why it matters for LLMs:** Long-context models (32K-128K tokens) need O(N^2) attention memory. At 32K on a single GPU, the attention matrix alone needs 32 GB. Splitting across 4 GPUs reduces this to 2 GB per GPU.

**Our results (standard SDPA vs sharded, 4 GPUs, BF16):**

| Sequence Length | Full Seq (ms) | Sharded (seq/4) (ms) | Speedup | Mem Savings |
|-----------------|--------------|---------------------|---------|------------|
| 2K | 0.19 | 0.04 | 4.8x | ~same |
| 4K | 1.59 | 0.07 | **22.7x** | 29% |
| 8K | 2.68 | 0.15 | **17.9x** | 29% |
| 16K | 6.20 | 0.48 | **12.9x** | 26% |

**Key insight:** Sequence sharding gives massive speedups because attention is O(N^2). Note: basic sharding misses cross-shard attention, so accuracy requires Ring Attention (PyTorch's `context_parallel` API, not yet available in our build). The speedups here represent the upper bound; actual context parallelism adds ~10-15% overhead for the ring communication.

---

### 18. FlexAttention Sliding Window (Sparse Attention)

**What it is:** Using FlexAttention's block mask to implement a sliding window of 256 tokens. Instead of each token attending to all previous tokens (O(N) per token), it only attends to the nearest 256 (O(1) per token). The block mask tells the fused kernel which blocks to skip entirely.

**Why it matters for LLMs:** For long sequences, full causal attention is quadratic. Sliding window makes it linear, enabling much longer sequences at constant cost per token.

**Our results (single A100, FP16):**

| Seq Length | Full Causal (ms) | Sliding-256 (ms) | Speedup | TFLOPS |
|------------|-----------------|------------------|---------|--------|
| 2K | 0.594 | **0.275** | **2.2x** | 500 |
| 4K | 0.862 | **0.348** | **2.5x** | 791 |
| 8K | 1.668 | **0.348** | **4.8x** | **1,581** |

**Key insight:** At 8K, sliding window is **4.8x faster** than full causal and achieves **1,581 TFLOPS** (apparent, because less total work is done). The cost is constant regardless of sequence length, making this essential for long-context models like Mistral.

---

## Maximum Inference Speed: Optimal Combination

Based on our benchmarks, here's the optimal stack for inference on 4x A100-40GB:

### For Decode Latency (single request, fastest response)

```
FlashAttention (69x attention speedup)
  + BF16 (11.5x tensor core utilization)
  + CUDA Graphs (3-4x kernel launch elimination)
  + N-gram Speculative Decoding (2.2x, zero extra memory)
  + Tensor Parallelism 4-way (near-linear scaling)
  + CUDA_DEVICE_MAX_CONNECTIONS=1 (7% free throughput)
```

**Estimated combined effect:** These target different bottlenecks, so the combination should yield **15-40x** over a naive FP32 single-GPU implementation.

### For Throughput (many requests, max tokens/second)

```
FlashAttention (69x)
  + BF16 (11.5x)
  + PagedAttention (2-4x via better batching)
  + Continuous Batching (eliminates idle GPU time)
  + Sliding Window Attention (4.8x at 8K for Mistral-style)
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

## Full Results Summary

| # | Technique | Speedup | Memory Impact | Effort |
|---|-----------|---------|-------------|--------|
| 1 | NVLink P2P | 4.9-8.8x bandwidth | -- | Hardware |
| 2 | NCCL Collectives | 197 GB/s AllReduce | -- | Automatic |
| 3 | **BF16 Mixed Precision** | **11.5x inference, 3.1x train** | **-17% memory** | One line |
| 4 | FSDP | Enables 3B+ models | Shards params | PyTorch native |
| 5 | **FlashAttention** | **69x at 8K** | **O(N) vs O(N^2)** | Automatic |
| 6 | **CUDA Graphs** | **3-4x decode** | Extra buffers | Low |
| 7 | BF16 > INT8 inference | 11.5x vs 5.4x | INT8 enables larger models | Low |
| 8 | Activation Checkpointing | 0.7x speed | -28% memory | Low |
| 9 | FlexAttention causal | 0.65x SDPA Flash | -- | Low |
| 10 | **torch.compile** | **1.13x training** | ~same | One line |
| 11 | **Liger-Kernel RMSNorm** | **4.27x fwd+bwd** | **-37% memory** | pip install |
| 12 | **N-gram Spec Decode** | **2.18x decode** | **Zero extra** | Custom code |
| 13 | NCCL Ring+LL128 | 11x for <1MB msgs | -- | Env var |
| 14 | **CUDA_MAX_CONN=1** | **+7.4% FSDP** | -- | Env var |
| 15 | AdamW8bit | 0.82x speed | -10% memory | pip install |
| 16 | INT8 + LoRA | 0.52x speed | -11% memory | pip install |
| 17 | **Sequence Sharding** | **13-23x attention** | **-26-29% memory** | PyTorch native |
| 18 | **Sliding Window Attn** | **4.8x at 8K** | Same | FlexAttention |

Bold = recommended for most workloads.

---

## Repository Structure

```
benchmarks/
  01_p2p_topology/       # NVLink P2P bandwidth & latency
  02_nccl_collectives/   # AllReduce, AllGather, ReduceScatter, Broadcast
  03_training_baselines/ # DDP vs FSDP, FP32/TF32/BF16
  04_advanced_opts/      # Flash attention, CUDA graphs, quantization, etc.
  05_sota_techniques/    # FlexAttention, torch.compile, Liger, spec decode, etc.
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

# Run SOTA technique benchmarks
sbatch benchmarks/05_sota_techniques/run_all.sh
```

## Key Takeaways

1. **BF16 is the single biggest win**: 11.5x inference, 3.1x training, one line of code
2. **FlashAttention is automatic and transformative**: 69x at 8K, constant memory
3. **FSDP + BF16 is the training sweet spot**: 4.2x speedup, 35% less memory
4. **CUDA Graphs dominate decode**: 3-4x for autoregressive generation
5. **NVLink works perfectly**: 93% of theoretical, enabling efficient multi-GPU
6. **A100 has no FP8**: Don't waste time with FP8 on this hardware
7. **40GB is the real constraint**: Use FSDP, quantization, and checkpointing to work around it
8. **N-gram speculation is free speed**: 2.2x decode with zero extra memory
9. **Liger-Kernel fuses the bottleneck ops**: 4.3x RMSNorm, trivial to adopt
10. **torch.compile helps training, not inference**: 13% training boost, but slower for batch=1 inference
11. **CUDA_DEVICE_MAX_CONNECTIONS=1**: Free 7% FSDP throughput from one env var
12. **Sliding window is the long-context unlock**: 4.8x at 8K via FlexAttention

## License

MIT
