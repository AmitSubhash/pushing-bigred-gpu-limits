# GPU Optimization Techniques for LLM Training & Inference on 4x A100-SXM4-40GB (2025-2026)

**Target Hardware**: 4x NVIDIA A100-SXM4-40GB, NVLink 3.0 (NV4 full mesh, 600 GB/s bidirectional)
**Software Stack**: PyTorch 2.8, Triton 3.4, NCCL 2.27, bitsandbytes, Transformers 4.57, Accelerate
**Date**: March 2026

---

## Table of Contents

1. [Attention Kernel Optimizations (FlashAttention / FlexAttention)](#1-attention-kernel-optimizations)
2. [torch.compile Best Practices](#2-torchcompile-best-practices)
3. [Activation Checkpointing Strategies](#3-activation-checkpointing-strategies)
4. [Communication-Computation Overlap (NCCL Async)](#4-communication-computation-overlap)
5. [KV Cache Optimization](#5-kv-cache-optimization)
6. [Quantization for Training & Inference](#6-quantization-for-training--inference)
7. [Speculative Decoding](#7-speculative-decoding)
8. [Tensor Parallelism for 4-GPU](#8-tensor-parallelism-for-4-gpu)
9. [Sequence & Context Parallelism](#9-sequence--context-parallelism)
10. [Memory-Efficient Optimizers](#10-memory-efficient-optimizers)
11. [Kernel Fusion with Triton](#11-kernel-fusion-with-triton)
12. [CUDA Graphs for Inference](#12-cuda-graphs-for-inference)
13. [Emerging 2025-2026 Techniques](#13-emerging-2025-2026-techniques)

---

## 1. Attention Kernel Optimizations

### FlashAttention-2 (Primary for A100)

**What it does**: Rewrites the attention computation to be IO-aware, tiling Q/K/V to SRAM and fusing softmax + matmul into a single kernel pass. Avoids materializing the full N x N attention matrix in HBM.

**Expected speedup on A100**: Up to 70% of theoretical max FLOPS on Ampere. 2-4x wall-clock speedup over naive attention. Memory usage goes from O(N^2) to O(N).

**A100 note**: FlashAttention-3 targets Hopper (H100) specifically for warp-specialized async TMA and FP8 tensor cores. On A100, FlashAttention-2 remains the optimal choice.

**Native in PyTorch 2.8**: Yes. Built into `torch.nn.functional.scaled_dot_product_attention` (SDPA).

```python
import torch
import torch.nn.functional as F

# Automatically dispatches to FlashAttention-2 on A100
# when inputs are float16/bfloat16 and on CUDA
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
```

### FlexAttention (PyTorch Native)

**What it does**: A compiler-driven API that lets you define custom attention score modifications (soft-capping, ALiBi, sliding window, prefix LM masks, etc.) via Python callables, which get JIT-compiled into the fused attention kernel via torch.compile. No need to write custom CUDA kernels.

**Expected speedup on A100**: Within 90% of hand-written FlashAttention-2 performance for standard patterns; significant speedup over manual attention mask implementations.

**Native in PyTorch 2.8**: Yes. Available as `torch.nn.attention.flex_attention`.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Causal mask via FlexAttention
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Soft-capping (e.g., Gemma-style)
def soft_cap(score, b, h, q_idx, kv_idx):
    softcap = 50.0
    score = score / softcap
    score = torch.tanh(score)
    score = score * softcap
    return score

block_mask = create_block_mask(causal_mask, B=batch, H=heads, Q_LEN=seq, KV_LEN=seq)

# Must be used within torch.compile for kernel fusion
@torch.compile
def attention_fn(q, k, v):
    return flex_attention(q, k, v, score_mod=soft_cap, block_mask=block_mask)

out = attention_fn(query, key, value)
```

**When to use FlexAttention over SDPA**:
- Custom attention patterns (sliding window + global tokens, prefix LM)
- Score modifications (ALiBi, soft-capping, relative position biases)
- Document masking in packed sequences
- Any non-standard mask that would otherwise require materializing the full mask matrix

---

## 2. torch.compile Best Practices

**What it does**: Traces PyTorch eager code into an optimized graph, applying operator fusion, memory planning, and optional CUDA graph capture. Typical speedup of 1.5-2x over eager for training; up to 4x for inference with static KV cache.

**Expected speedup on A100**: 10-30% for training (higher for smaller models); 50-100% for inference decode with `reduce-overhead` mode.

**Native in PyTorch 2.8**: Yes.

### Training Configuration

```python
import torch

model = MyLLM().cuda().bfloat16()

# Option A: Compile entire model (works for simple cases)
compiled_model = torch.compile(model, mode="default")

# Option B: Regional compilation (recommended for LLMs)
# Only compile the transformer block to reduce compile time
for layer in model.transformer.layers:
    layer = torch.compile(layer, mode="default")

# Option C: fullgraph for maximum optimization (no graph breaks)
compiled_model = torch.compile(model, fullgraph=True)
```

### Inference Configuration

```python
# For decode phase: reduce-overhead uses CUDA graphs under the hood
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

# With static KV cache for up to 4x speedup
from transformers import StaticCache
model.generation_config.cache_implementation = "static"
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

### When torch.compile HELPS:
- Autoregressive decode loops (huge win from CUDA graph capture)
- Models with many small ops that can be fused (layer norms, activations, residuals)
- Static shapes (no dynamic sequence lengths between calls)
- Single-GPU or TP workloads without frequent graph breaks

### When torch.compile HURTS:
- First-call compilation overhead (can be minutes to hours for large models)
- Dynamic shapes that trigger recompilation
- Models with heavy Python control flow (many graph breaks)
- Multi-GPU with FSDP: gradient reduce-scatter introduces graph breaks (use `fullgraph=False`)
- When model code uses unsupported operations

### Best Practices:
1. **Regional compilation** over whole-model compilation for faster compile times
2. Use `TORCH_LOGS="+dynamo"` to identify graph breaks
3. `torch._dynamo.config.suppress_errors = True` for graceful fallback during development
4. Cache compiled artifacts: `TORCHINDUCTOR_FX_GRAPH_CACHE=1`
5. For distributed: compile individual layers, not the entire DDP/FSDP wrapper

---

## 3. Activation Checkpointing Strategies

**What it does**: Discards intermediate activations during forward pass and recomputes them during backward. Trades ~33% extra compute for 50-70% memory reduction.

### Full Activation Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        return self.ffn(self.attention(x))

# Wrap each transformer layer
for layer in model.layers:
    original_forward = layer.forward
    layer.forward = lambda *args, _fn=original_forward, **kwargs: checkpoint(
        _fn, *args, use_reentrant=False, **kwargs
    )
```

### Selective Activation Checkpointing (SAC) -- Recommended

**What it does**: Only recomputes cheap ops (pointwise, activations) while saving expensive ops (matmuls). Achieves ~50% of full checkpointing's memory savings with ~5% compute overhead instead of 33%.

**Native in PyTorch 2.8**: Yes (stable since 2.5).

```python
from torch.utils.checkpoint import checkpoint, CheckpointPolicy
from torch.utils.checkpoint import create_selective_checkpoint_contexts
from functools import partial

# Policy: save matmuls, recompute everything else
ops_to_save = {
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.addmm.default,
}

def policy_fn(ctx, op, *args, **kwargs):
    if op in ops_to_save:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

# Apply to each transformer layer
out = checkpoint(
    layer_fn, *args,
    use_reentrant=False,
    context_fn=partial(create_selective_checkpoint_contexts, policy_fn),
)
```

### Memory Budget API (with torch.compile)

```python
import torch._functorch.config

# 0.0 = full activation checkpointing (recompute everything)
# 1.0 = no checkpointing (save everything)
# 0.5 = recompute ~50% of activations (pareto-optimal selection)
torch._functorch.config.activation_memory_budget = 0.5

compiled_model = torch.compile(model)
```

**Expected savings on A100-40GB**:
- Full AC: ~60% memory reduction, ~33% slower training
- SAC: ~30-50% memory reduction, ~5-10% slower training
- Memory Budget 0.5: ~50% memory reduction with automatic pareto-optimal policy

**Recommendation for 4x A100-40GB**: Use SAC or Memory Budget API at 0.5-0.7. The 40GB VRAM is tight for large models; SAC gives the best memory-performance tradeoff.

---

## 4. Communication-Computation Overlap

**What it does**: Overlaps NCCL collective operations (all-reduce, all-gather, reduce-scatter) with GPU compute to hide communication latency behind useful work.

**Expected speedup on A100 NVLink**: 10-25% throughput improvement for distributed training by hiding the ~1-3ms per collective behind compute.

### FSDP2 with Built-in Overlap

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# FSDP2 automatically overlaps all-gather of next layer
# with compute of current layer
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

### Manual Async Operations

```python
import torch.distributed as dist

# Async all-reduce overlapped with compute
handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
# Do other compute while communication happens
result = compute_something_else(other_tensor)
# Synchronize when you need the result
handle.wait()
```

### SymmetricMemory for NVLink (Advanced)

PyTorch SymmetricMemory enables direct P2P GPU memory access over NVLink, avoiding NCCL overhead for intra-node communication.

```python
import torch.distributed.symmetric_memory as symm_mem

# Allocate symmetric memory accessible by all 4 GPUs via NVLink
t = symm_mem.empty_strided_p2p(
    size=(shard_size,), stride=(1,),
    dtype=torch.bfloat16, device=device,
    storage_offset=0,
)

# Fused all-gather + GEMM with NVLink pipelining
symm_mem.fused_all_gather_matmul(
    input_tensor, weight,
    gather_dim=0, group_name="default"
)
```

### NCCL Tuning for 4x A100 NVLink

```bash
# Environment variables for optimal NVLink performance
export NCCL_ALGO=Ring          # Ring algorithm optimal for 4 GPUs
export NCCL_PROTO=LL128        # Low-latency 128-byte protocol for NVLink
export NCCL_MIN_NCHANNELS=4    # Min channels for parallelism
export NCCL_MAX_NCHANNELS=12   # Max channels
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Ensure overlap works properly
```

---

## 5. KV Cache Optimization

### PagedAttention (Inference)

**What it does**: Manages KV cache as fixed-size blocks (like OS virtual memory pages), eliminating memory fragmentation. Reduces KV cache memory waste from 60-80% to under 4%.

**Expected speedup on A100**: 2-4x throughput improvement through higher batch sizes enabled by memory savings.

**Implementation**: Via vLLM or SGLang (not native PyTorch).

```python
# vLLM with PagedAttention (automatic)
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=4,           # Use all 4 A100s
    gpu_memory_utilization=0.90,      # Use 90% of 40GB per GPU
    max_model_len=8192,
    block_size=16,                    # KV cache block size (tokens)
    enable_chunked_prefill=True,      # Enable chunked prefill
    max_num_batched_tokens=2048,      # Limit batch tokens for latency
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

### Chunked Prefill

**What it does**: Splits long input sequences into chunks so prefill runs interleaved with decode, reducing time-to-first-token (TTFT) by up to 30% for long prompts.

```python
# vLLM: Chunked prefill is enabled by default in V1
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,  # Chunk size
)
```

### Static KV Cache (with torch.compile)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = model.to("cuda").to(torch.bfloat16)

# Static cache enables CUDA graph capture via torch.compile
cache = StaticCache(
    config=model.config,
    max_batch_size=8,
    max_cache_len=2048,
    device="cuda",
    dtype=torch.bfloat16,
)

compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

---

## 6. Quantization for Training & Inference

### CRITICAL: A100 FP8 Limitations

**A100 does NOT have native FP8 tensor cores.** FP8 (E4M3/E5M2) is a Hopper (H100) and Ada Lovelace feature. On A100, FP8 runs in emulation mode with NO performance benefit for training. Do not use Transformer Engine FP8 or MS-AMP FP8 for training on A100.

### What WORKS on A100 for Training

| Precision | A100 Support | Use Case | Speedup |
|-----------|-------------|----------|---------|
| BF16 mixed precision | Native tensor cores | Default training | 2x over FP32 |
| TF32 | Native tensor cores | Drop-in FP32 replacement | 1.5-2x over FP32 |
| INT8 (compute) | Native tensor cores | Quantized inference | 1.5-2x over FP16 |
| FP16 mixed precision | Native tensor cores | Training (with loss scaling) | 2x over FP32 |

### BF16 Mixed Precision Training (Recommended for A100)

```python
# PyTorch native AMP
scaler = None  # BF16 does not need gradient scaling
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input_ids)
    loss = criterion(output, labels)
loss.backward()
optimizer.step()
```

### INT8 Inference with bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization for inference (A100 compatible)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",  # Spreads across 4 GPUs
)
```

### 4-bit QLoRA Training (A100 Compatible)

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
# Now trainable parameters are in BF16, base model is in NF4
```

### INT8 SmoothQuant for Inference

```python
# Via vLLM (A100 supports INT8 W8A8 natively)
from vllm import LLM

llm = LLM(
    model="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",
    tensor_parallel_size=4,
    quantization="compressed-tensors",
)
```

---

## 7. Speculative Decoding

**What it does**: A small, fast "draft" model proposes K tokens speculatively. The large "target" model verifies all K tokens in a single forward pass (parallel verification). Accepted tokens are output; rejected ones cause a fallback. The output is mathematically identical to standard decoding.

**Expected speedup on A100**: 1.5-3x decode throughput depending on acceptance rate (alpha). Best when alpha >= 0.6 with K >= 5 draft tokens.

**Implementation**: Via vLLM, SGLang, or custom PyTorch.

### vLLM Speculative Decoding

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",  # Small draft model
    speculative_draft_tensor_parallel_size=1,  # Draft on 1 GPU
    num_speculative_tokens=5,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)
```

### N-gram Speculative Decoding (No Draft Model)

```python
# Uses n-gram matching from prompt as draft -- zero extra GPU memory
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="[ngram]",
    ngram_prompt_lookup_max=4,
    num_speculative_tokens=5,
)
```

### Eagle3 (State-of-the-Art, 2025)

Eagle3 uses hidden states from multiple layers of the target model as draft input, achieving the highest acceptance rates. Requires a pretrained Eagle3 draft head.

```python
# Via vLLM with Eagle3 draft model
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="path/to/eagle3-draft-head",
    speculative_method="eagle3",
    num_speculative_tokens=8,
)
```

### When to Use:
- Latency-sensitive applications (chatbots, real-time inference)
- When GPU is memory-bound during decode (common on A100)
- Temperature=0 or low temperature sampling (higher acceptance rate)

### When NOT to Use:
- High-throughput batch processing (continuous batching is better)
- High temperature / diverse sampling (low acceptance rate)

---

## 8. Tensor Parallelism for 4-GPU

**What it does**: Splits model weight matrices across GPUs. Each GPU holds 1/4 of each layer's weights and computes on 1/4 of the hidden dimension, communicating via all-reduce after each layer.

**Expected speedup on A100 NVLink**: Near-linear scaling (3.5-3.8x for 4 GPUs) due to NVLink's 600 GB/s bandwidth hiding communication latency.

### Optimal Configurations for 4x A100

| Model Size | Recommended Strategy | Details |
|-----------|---------------------|---------|
| < 7B params | FSDP2 only (DP=4) | TP overhead not worth it |
| 7-13B params | FSDP2 (DP=4) or TP=4 | Either works; TP for inference |
| 13-70B params | TP=4 for inference; TP=2 x FSDP=2 for training | 2D parallelism |
| 70B+ params | TP=4 + quantization | 40GB per GPU is tight |

### FSDP2 (Data Parallelism -- Best for Training)

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# Simple 1D mesh for 4-GPU FSDP
mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# Shard each transformer layer
for layer in model.layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

# Shard the root model
fully_shard(model, mesh=mesh, mp_policy=mp_policy)

# Training loop -- FSDP2 handles sharding/gathering automatically
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 2D Parallelism (TP + FSDP)

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed.fsdp import fully_shard

# 2D mesh: TP=2, DP=2 across 4 GPUs
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

# Apply tensor parallelism to attention and MLP
for layer in model.layers:
    parallelize_module(
        layer.attention,
        mesh["tp"],
        {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        },
    )
    parallelize_module(
        layer.mlp,
        mesh["tp"],
        {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )
    # Then apply FSDP on the DP dimension
    fully_shard(layer, mesh=mesh["dp"])

fully_shard(model, mesh=mesh["dp"])
```

### Pure Tensor Parallelism (Best for Inference)

```python
# Via vLLM -- simplest path for inference TP
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4)
```

---

## 9. Sequence & Context Parallelism

### Context Parallelism (Ring Attention)

**What it does**: Splits the input sequence across GPUs along the sequence dimension. Each GPU processes a shard of the sequence and communicates KV blocks in a ring pattern. Enables training with sequences far longer than what fits in single-GPU memory.

**Expected benefit on 4x A100**: Enables 4x longer sequences (e.g., 32K per GPU -> 128K total). Throughput overhead of ~10-15% vs standard attention due to ring communication.

**Native in PyTorch 2.8**: Yes, via `torch.nn.attention.context_parallel`.

```python
from torch.nn.attention import context_parallel
from torch.distributed.device_mesh import init_device_mesh

# Create a mesh for context parallelism across 4 GPUs
cp_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("cp",))

# Wrap the attention region
with context_parallel(cp_mesh):
    # SDPA is automatically replaced with Ring Attention
    out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
```

### Sequence Parallelism (with Tensor Parallelism)

**What it does**: While TP splits the hidden dimension, sequence parallelism splits the sequence dimension for non-tensor-parallel operations (LayerNorm, Dropout). Reduces activation memory by the TP degree.

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)

# SP is applied to LayerNorm and Dropout layers
parallelize_module(
    layer,
    mesh["tp"],
    {
        "attention.q_proj": ColwiseParallel(),
        "attention.o_proj": RowwiseParallel(),
        "norm1": SequenceParallel(),  # LayerNorm runs on sequence shards
        "norm2": SequenceParallel(),
    },
)
```

### When to Use:
- **Context Parallelism**: Long sequences (>32K tokens), especially pretraining or long-context fine-tuning
- **Sequence Parallelism**: Always combine with TP to reduce activation memory; essentially free on NVLink

---

## 10. Memory-Efficient Optimizers

### 8-bit Adam (bitsandbytes)

**What it does**: Quantizes optimizer states (momentum, variance) to 8-bit, reducing optimizer memory by 75% (from 8 bytes/param to 2 bytes/param).

**Expected savings on A100**: For a 7B model, optimizer states go from ~56GB to ~14GB. Enables training models 2-3x larger in the same memory.

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# Or via Transformers TrainingArguments
from transformers import TrainingArguments
args = TrainingArguments(
    optim="adamw_bnb_8bit",
    learning_rate=1e-4,
    # ...
)
```

### Adafactor

**What it does**: Factorizes the second moment matrix into row and column factors, reducing memory from O(mn) to O(m+n) per weight matrix.

**Expected savings**: ~50-60% optimizer memory reduction vs AdamW.

```python
from transformers import Adafactor

optimizer = Adafactor(
    model.parameters(),
    lr=1e-4,
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
)
```

### LOMO / AdaLomo

**What it does**: Fuses gradient computation and parameter update into a single step, never materializing the full gradient tensor. Reduces memory to nearly zero for optimizer states.

**Expected savings**: Optimizer memory -> ~0. Enables full-parameter fine-tuning of 65B models on 4x A100-40GB.

```python
# Via Transformers
from transformers import TrainingArguments
args = TrainingArguments(
    optim="adalomo",
    learning_rate=1e-5,
    # ...
)

# Or directly
from lomo_optim import AdaLomo
optimizer = AdaLomo(model, lr=1e-5)
```

### GaLore (Gradient Low-Rank Projection)

**What it does**: Projects gradients into a low-rank subspace before optimizer state computation. Reduces optimizer memory by up to 65.5% while maintaining full-parameter learning (unlike LoRA).

**Expected savings**: Up to 82.5% optimizer memory reduction with 8-bit GaLore. Pre-trains 7B models on a single 24GB GPU.

```python
# Install: pip install galore-torch
from galore_torch import GaLoreAdamW8bit

param_groups = [
    {"params": non_linear_params, "rank": 0},     # No projection for small params
    {"params": linear_params, "rank": 512,         # Low-rank projection for large params
     "update_proj_gap": 200, "scale": 0.25},
]

optimizer = GaLoreAdamW8bit(param_groups, lr=1e-4)
```

### GaLore 2 (2025)

Addresses SVD overhead in GaLore 1; scales to 7B pretraining with 500B tokens. Improved stability and reduced projection update frequency.

### Recommendation for 4x A100-40GB:
- **Pretraining**: 8-bit AdamW + FSDP2 (best convergence-memory tradeoff)
- **Fine-tuning large models**: AdaLomo or GaLore (enables models that otherwise won't fit)
- **QLoRA fine-tuning**: Standard AdamW is fine (LoRA params are small)

---

## 11. Kernel Fusion with Triton

### Liger-Kernel (Drop-in Fused Kernels)

**What it does**: Collection of Triton kernels that replace standard PyTorch operations with fused versions. RMSNorm + residual, SwiGLU, RoPE, CrossEntropy, and FusedLinearCrossEntropy are fused into single GPU kernel calls.

**Expected speedup on A100**: 20% training throughput increase, 60% memory reduction.

**Installation**: `pip install liger-kernel`

```python
# One-line monkey-patch for HuggingFace models
from liger_kernel.transformers import apply_liger_kernel_to_llama

apply_liger_kernel_to_llama()  # Patches all Llama models globally

# Or selective application
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama(
    rope=True,
    swiglu=True,
    cross_entropy=True,
    fused_linear_cross_entropy=True,  # Biggest win: fuses final linear + CE
    rms_norm=True,
)
```

### Individual Kernel Usage

```python
from liger_kernel.ops.rms_norm import LigerRMSNorm
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

# Replace layers individually
model.norm = LigerRMSNorm(hidden_size, eps=1e-6)
loss_fn = LigerFusedLinearCrossEntropyLoss()  # Fuses lm_head + CE loss
```

### Per-Kernel Performance on A100:

| Kernel | Speed vs PyTorch | Memory vs PyTorch |
|--------|-----------------|-------------------|
| RMSNorm | ~3x faster | ~3x less |
| SwiGLU | ~1x (parity) | ~1.5x less |
| CrossEntropy | ~3x faster | ~5x less |
| FusedLinearCrossEntropy | ~4x faster | ~10x less |
| RoPE | ~2x faster | ~2x less |

### Writing Custom Triton Kernels

```python
import triton
import triton.language as tl

@triton.jit
def fused_residual_rmsnorm_kernel(
    X_ptr, Residual_ptr, Weight_ptr, Out_ptr,
    N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load input and residual
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)
    res = tl.load(Residual_ptr + row * N + cols, mask=mask, other=0.0)

    # Fused residual add + RMSNorm
    x = x + res
    tl.store(Residual_ptr + row * N + cols, x, mask=mask)  # Update residual in-place

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(Weight_ptr + cols, mask=mask, other=0.0)
    out = x * rstd * w

    tl.store(Out_ptr + row * N + cols, out, mask=mask)
```

### Best Practices for Custom Triton Kernels:
1. **Fuse sequences**: Always combine normalization + activation + multiply
2. **Leverage registers**: Cache scalars (e.g., RMS statistics) in registers
3. **Recompute over store**: In backward passes, recompute cheap ops rather than storing intermediates
4. **Tune BLOCK_SIZE**: Profile with `triton.autotune` across `[256, 512, 1024, 2048]`
5. **In-place operations**: Modify residual streams in-place to avoid extra memory allocations

---

## 12. CUDA Graphs for Inference

**What it does**: Records a sequence of GPU operations into a graph during a warmup run, then replays the entire graph with a single CPU launch. Eliminates per-kernel CPU launch overhead (~10-20 microseconds per kernel x hundreds of kernels per forward pass).

**Expected speedup on A100**: 2-4x for decode phase (which is CPU-bound due to small per-kernel compute). Less benefit for prefill (which is compute-bound).

**Native in PyTorch 2.8**: Yes, both directly and via `torch.compile(mode="reduce-overhead")`.

### Direct CUDA Graph Usage

```python
import torch

model.eval()

# Static inputs required for CUDA graphs
static_input = torch.randint(0, vocab_size, (batch_size, 1), device="cuda")
static_cache = StaticKVCache(...)

# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        _ = model(static_input, past_key_values=static_cache)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input, past_key_values=static_cache)

# Inference loop -- replay graph, just copy new inputs
for token_ids in decode_loop:
    static_input.copy_(token_ids)
    g.replay()
    logits = static_output.logits  # Read output from static buffer
```

### Via torch.compile (Recommended)

```python
# reduce-overhead mode automatically captures CUDA graphs
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

# First call triggers compilation + graph capture
output = model(input_ids)  # Slow (compilation)
output = model(input_ids)  # Fast (graph replay)
```

### CUDA Graphs in vLLM

vLLM automatically uses CUDA graphs for the decode phase. Graph-incompatible operations (like cascade attention) are handled by breaking the graph into safe/unsafe segments.

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=False,  # Default: CUDA graphs enabled
    # gpu_memory_utilization controls how much memory is reserved for graphs
    gpu_memory_utilization=0.90,
)
```

### Limitations:
- Requires static tensor shapes (no dynamic sequence lengths within a graph)
- Memory overhead for storing graph buffers
- Not all operations are graph-capturable (data-dependent control flow, CPU sync points)
- Batch size changes require graph recapture

---

## 13. Emerging 2025-2026 Techniques

### TorchTitan Framework

**What it is**: PyTorch-native platform for production LLM pretraining. Composes FSDP2 + TP + PP + CP + Float8 + torch.compile in a modular, tested framework.

**Key results**: 65% acceleration at 128-GPU scale for Llama 3.1 8B; additional 12.6% with 2D parallelism for 70B; additional 30% with 3D parallelism for 405B.

```bash
# Install
pip install torchtitan

# Example: pretrain Llama 3 8B on 4 GPUs
torchrun --nproc_per_node=4 \
    -m torchtitan.train \
    --job.config_file train_configs/llama3_8b.toml
```

### GaLore 2 (April 2025)

Successor to GaLore with reduced SVD overhead. Enables pretraining 7B models with 500B tokens with the same memory as LoRA but full-parameter learning quality.

### Eagle3 Speculative Decoding (2025)

State-of-the-art draft model architecture using multi-layer hidden states from the target model. Highest acceptance rates across model families.

### SymmetricMemory + Fused Collectives

PyTorch's new primitive for NVLink-aware P2P operations. Enables fused all-gather-matmul and reduce-scatter-matmul that overlap communication with compute at the kernel level (not just stream level).

### MLSys 2026 Competition Tracks (Emerging)

- **Fused MoE kernels**: Triton kernels for efficient expert routing + computation in Mixture-of-Experts models
- **Sparse Attention**: Long-context inference with sparse attention patterns
- **Gated Delta Networks**: Efficient state-update operations for linear attention alternatives

### Adapprox (2025)

Adaptive randomized low-rank approximation for optimizer states. Further reduces memory below GaLore with theoretical convergence guarantees.

### Continuous Batching Improvements

Modern inference servers (vLLM V1, SGLang) now default to iteration-level scheduling with automatic preemption, achieving near-optimal GPU utilization without manual batch management.

---

## Quick Reference: Configuration for 4x A100-40GB

### Training (Pretraining/Full Fine-tuning)

```python
# Optimal training setup for 7-13B models on 4x A100-40GB
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# 1. Mixed precision
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

# 2. FSDP2 sharding
for layer in model.layers:
    fully_shard(layer, mp_policy=mp)
fully_shard(model, mp_policy=mp)

# 3. Selective activation checkpointing
torch._functorch.config.activation_memory_budget = 0.5

# 4. Compile transformer blocks
for layer in model.layers:
    torch.compile(layer)

# 5. Liger-Kernel fused ops
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()

# 6. 8-bit optimizer
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

# 7. NCCL tuning (set via env vars before launch)
# NCCL_ALGO=Ring NCCL_PROTO=LL128 CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Inference (Maximum Throughput)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.92,
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
    # Optional: speculative decoding
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
)
```

### Inference (Minimum Latency)

```python
from vllm import LLM, SamplingParams

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

## Summary Table

| Technique | Speedup | Memory Savings | A100 Native | Extra Packages |
|-----------|---------|---------------|-------------|----------------|
| FlashAttention-2 (SDPA) | 2-4x attention | O(N) vs O(N^2) | PyTorch 2.8 | None |
| FlexAttention | ~0.9x FA2 | Same as FA2 | PyTorch 2.8 | None |
| torch.compile | 1.3-2x train, 2-4x infer | Slight | PyTorch 2.8 | None |
| SAC | 0.9-0.95x speed | 30-50% less | PyTorch 2.8 | None |
| Memory Budget API | Tunable | Tunable | PyTorch 2.8 | None |
| FSDP2 | Near-linear DP scaling | Shards params | PyTorch 2.8 | None |
| TP (4-GPU) | 3.5-3.8x | 1/4 per GPU | PyTorch 2.8 | None |
| Context Parallel | Enables 4x seq len | 1/4 per GPU | PyTorch 2.8 | None |
| Liger-Kernel | +20% throughput | -60% peak | pip install | liger-kernel |
| 8-bit AdamW | ~1x | -75% optim | pip install | bitsandbytes |
| GaLore | ~1x | -65% optim | pip install | galore-torch |
| LOMO/AdaLomo | ~1x | ~0 optim | pip install | lomo-optim |
| CUDA Graphs | 2-4x decode | Extra buffers | PyTorch 2.8 | None |
| PagedAttention | 2-4x throughput | -90% KV waste | pip install | vllm |
| Speculative Decoding | 1.5-3x latency | +draft model | pip install | vllm |
| BF16 Mixed Precision | 2x over FP32 | 50% less | PyTorch 2.8 | None |
| INT8 Inference | 1.5-2x over FP16 | 50% less | A100 native | bitsandbytes/vllm |
| NF4 QLoRA | Enables 70B training | ~75% less | pip install | bitsandbytes, peft |

---

## Sources

- [FlashAttention-3 Blog (PyTorch)](https://pytorch.org/blog/flashattention-3/)
- [FlexAttention Blog (PyTorch)](https://pytorch.org/blog/flexattention/)
- [FlexAttention API Docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [Attention Gym (GitHub)](https://github.com/meta-pytorch/attention-gym)
- [State of torch.compile for Training (Aug 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [torch.compile HuggingFace Guide](https://huggingface.co/docs/transformers/en/perf_torch_compile)
- [vLLM torch.compile Integration](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [Activation Checkpointing Techniques (PyTorch Blog)](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [torch.utils.checkpoint Docs](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch SymmetricMemory](https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798)
- [NCCL Analysis (arXiv)](https://arxiv.org/html/2507.04786v1)
- [PagedAttention (vLLM Docs)](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [vLLM Optimization Guide](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [bitsandbytes HuggingFace Integration](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [Speculative Decoding Guide (PyTorch)](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)
- [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [Eagle3 (ICLR 2026)](https://openreview.net/pdf?id=aL1Wnml9Ef)
- [FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [TorchTitan Paper](https://arxiv.org/abs/2410.06511)
- [2D Parallelism (Lightning)](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp_fsdp.html)
- [Context Parallel Tutorial (PyTorch)](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [Ring Attention Blog (Akasa)](https://akasa.com/blog/ring-attention)
- [HuggingFace Optimizers Guide](https://huggingface.co/docs/transformers/en/optimizers)
- [GaLore Paper](https://arxiv.org/abs/2403.03507)
- [GaLore 2 Paper](https://arxiv.org/abs/2504.20437)
- [Liger-Kernel (GitHub)](https://github.com/linkedin/Liger-Kernel)
- [Fused Kernels Deep Dive (TDS)](https://towardsdatascience.com/cutting-llm-memory-by-84-a-deep-dive-into-fused-kernels/)
- [Custom Triton Kernels for Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [CUDA Graphs (PyTorch Blog)](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [PyGraph: CUDA Graph Compiler Support](https://arxiv.org/html/2503.19779v2)
- [vLLM Anatomy Blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Speculators v0.3.0 (vLLM Blog)](https://blog.vllm.ai/2025/12/13/speculators-v030.html)
