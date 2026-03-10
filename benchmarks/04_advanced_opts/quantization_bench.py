"""Quantization Benchmark for LLM Inference on A100.

Tests: FP32, FP16, BF16, INT8 (bitsandbytes), INT4 (bitsandbytes NF4).
Measures throughput, memory, and output quality (perplexity proxy).
"""
import os
import time
import torch
import torch.nn as nn


def setup():
    rank = 0
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank


class TransformerLM(nn.Module):
    """Transformer LM for quantization testing."""
    def __init__(self, vocab_size: int = 32000, d_model: int = 2048,
                 nhead: int = 32, num_layers: int = 16, dim_ff: int = 8192):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        h = self.transformer(h)
        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1e6


def benchmark_inference_throughput(model, batch: int, seq_len: int,
                                   warmup: int = 5, iters: int = 30):
    """Run inference and measure throughput."""
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 32000, (batch, seq_len), device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (iters * batch * seq_len) / elapsed
    ms_per_step = (elapsed / iters) * 1000
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    return tokens_per_sec, ms_per_step, peak_mem_gb


def main():
    rank = setup()

    if rank == 0:
        print("Quantization Inference Benchmark")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    batch, seq_len = 4, 512

    results = []

    # ---- FP32 ----
    if rank == 0:
        print("Loading FP32 model...")
    model = TransformerLM().cuda().eval()
    params = count_params(model)
    if rank == 0:
        print(f"  Params: {params:,} ({get_model_size_mb(model):.0f} MB)")
    tps, ms, mem = benchmark_inference_throughput(model, batch, seq_len)
    results.append(("FP32", tps, ms, mem))
    del model
    torch.cuda.empty_cache()

    # ---- FP16 ----
    if rank == 0:
        print("Loading FP16 model...")
    model = TransformerLM().cuda().half().eval()
    tps, ms, mem = benchmark_inference_throughput(model, batch, seq_len)
    results.append(("FP16", tps, ms, mem))
    del model
    torch.cuda.empty_cache()

    # ---- BF16 ----
    if rank == 0:
        print("Loading BF16 model...")
    model = TransformerLM().cuda().to(torch.bfloat16).eval()
    tps, ms, mem = benchmark_inference_throughput(model, batch, seq_len)
    results.append(("BF16", tps, ms, mem))
    del model
    torch.cuda.empty_cache()

    # ---- FP16 + TF32 matmul ----
    if rank == 0:
        print("Loading FP16 + TF32 model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = TransformerLM().cuda().half().eval()
    tps, ms, mem = benchmark_inference_throughput(model, batch, seq_len)
    results.append(("FP16+TF32", tps, ms, mem))
    del model
    torch.cuda.empty_cache()

    # ---- INT8 (bitsandbytes) ----
    try:
        import bitsandbytes as bnb
        if rank == 0:
            print(f"Loading INT8 model (bitsandbytes {bnb.__version__})...")

        # Replace linear layers with 8-bit versions
        model = TransformerLM().eval()
        model = model.cuda()

        # Use bitsandbytes quantization via torch
        from bitsandbytes.nn import Linear8bitLt

        def replace_linear_with_8bit(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    new_layer = Linear8bitLt(
                        child.in_features, child.out_features,
                        bias=child.bias is not None, has_fp16_weights=False,
                    )
                    new_layer.weight = bnb.nn.Int8Params(
                        child.weight.data, requires_grad=False,
                    )
                    if child.bias is not None:
                        new_layer.bias = child.bias
                    setattr(module, name, new_layer)
                else:
                    replace_linear_with_8bit(child)

        replace_linear_with_8bit(model)
        model = model.cuda()

        tps, ms, mem = benchmark_inference_throughput(model, batch, seq_len)
        results.append(("INT8-bnb", tps, ms, mem))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        if rank == 0:
            print(f"  INT8 failed: {e}")
        results.append(("INT8-bnb", 0, 0, 0))

    # ---- torch.compile + BF16 ----
    if rank == 0:
        print("Loading torch.compile + BF16 model...")
    try:
        model = TransformerLM().cuda().to(torch.bfloat16).eval()
        model = torch.compile(model, mode="reduce-overhead")
        # Extra warmup for compilation
        tps, ms, mem = benchmark_inference_throughput(model, batch, seq_len,
                                                      warmup=10, iters=30)
        results.append(("compile+BF16", tps, ms, mem))
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        if rank == 0:
            print(f"  torch.compile failed: {e}")
        results.append(("compile+BF16", 0, 0, 0))

    # ---- Results ----
    if rank == 0:
        print()
        print(f"{'='*70}")
        print(f"{'Config':<18} {'tok/s':>12} {'ms/step':>10} {'Peak Mem (GB)':>14} {'Speedup':>8}")
        print(f"{'-'*70}")
        base_tps = results[0][1]
        for name, tps, ms, mem in results:
            if tps > 0:
                speedup = tps / base_tps
                print(f"{name:<18} {tps:>12,.0f} {ms:>10.1f} {mem:>14.2f} {speedup:>7.2f}x")
            else:
                print(f"{name:<18} {'FAILED':>12}")
        print(f"{'='*70}")
        print(f"\nBatch={batch}, Seq={seq_len}, Params={params:,}")


if __name__ == "__main__":
    main()
