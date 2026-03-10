"""Benchmark SDPA/FlashAttention backends on A100.

Compares: Math, FlashAttention, MemoryEfficient backends
across different sequence lengths and head dimensions.
"""
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist


def setup():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return local_rank


def benchmark_sdpa(
    batch: int, heads: int, seq_len: int, head_dim: int,
    backend: str, dtype: torch.dtype, warmup: int = 10, iters: int = 50,
) -> dict:
    """Benchmark a specific SDPA backend."""
    device = torch.cuda.current_device()
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # Select backend - configure via enable/disable flags
    def set_backend(name):
        torch.backends.cuda.enable_flash_sdp(name == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(name == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(name == "math")

    set_backend(backend)

    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    # Calculate FLOPS (2 * batch * heads * seq^2 * head_dim for attn + same for value)
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = flops / (avg_ms / 1000) / 1e12

    # Re-enable all backends
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "tflops": tflops,
        "mem_gb": mem_gb,
    }


def main():
    rank = setup()

    if rank == 0:
        print("SDPA / FlashAttention Benchmark")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Flash SDPA enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"MemEfficient SDPA: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print()

    configs = [
        # (batch, heads, seq_len, head_dim, dtype)
        (4, 32, 512, 64, torch.float16, "Small seq (512)"),
        (4, 32, 1024, 64, torch.float16, "Medium seq (1K)"),
        (4, 32, 2048, 64, torch.float16, "Long seq (2K)"),
        (2, 32, 4096, 64, torch.float16, "Very long seq (4K)"),
        (1, 32, 8192, 64, torch.float16, "Ultra seq (8K)"),
        (4, 32, 2048, 128, torch.float16, "Large head_dim (128)"),
        (4, 32, 2048, 64, torch.bfloat16, "BF16 (2K)"),
    ]

    backends = ["flash", "mem_efficient", "math"]

    if rank == 0:
        print(f"{'Config':<25} {'Backend':<15} {'Avg (ms)':>10} {'Min (ms)':>10} "
              f"{'TFLOPS':>10} {'Mem (GB)':>10}")
        print("-" * 85)

    for batch, heads, seq_len, head_dim, dtype, label in configs:
        for backend in backends:
            try:
                result = benchmark_sdpa(batch, heads, seq_len, head_dim, backend, dtype)
                if rank == 0:
                    print(f"{label:<25} {backend:<15} {result['avg_ms']:>10.3f} "
                          f"{result['min_ms']:>10.3f} {result['tflops']:>10.2f} "
                          f"{result['mem_gb']:>10.2f}")
            except Exception as e:
                if rank == 0:
                    print(f"{label:<25} {backend:<15} {'FAILED':>10} -- {str(e)[:40]}")

        if rank == 0:
            print()

    if rank == 0:
        print("Benchmark complete.")


if __name__ == "__main__":
    main()
