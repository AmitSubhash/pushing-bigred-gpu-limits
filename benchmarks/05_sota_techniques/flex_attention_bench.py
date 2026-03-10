"""FlexAttention vs SDPA FlashAttention Benchmark on A100.

Compares PyTorch 2.8 FlexAttention (compiled custom masks)
against standard SDPA FlashAttention backend.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def benchmark_attention(fn, warmup: int = 10, iters: int = 50) -> dict:
    """Benchmark an attention function."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times) / len(times)
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return {"avg_ms": avg_ms, "mem_gb": mem_gb}


def main():
    torch.cuda.set_device(0)
    print("FlexAttention vs SDPA Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Define FlexAttention mask functions
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def sliding_window_mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= 256)

    def soft_cap_score_mod(score, b, h, q_idx, kv_idx):
        cap = 50.0
        return cap * torch.tanh(score / cap)

    configs = [
        (4, 32, 512, 64, "seq=512"),
        (4, 32, 1024, 64, "seq=1K"),
        (4, 32, 2048, 64, "seq=2K"),
        (2, 32, 4096, 64, "seq=4K"),
        (1, 32, 8192, 64, "seq=8K"),
    ]

    print(f"{'Config':<15} {'Method':<25} {'Avg (ms)':>10} {'TFLOPS':>10} {'Mem (GB)':>10}")
    print("-" * 75)

    for batch, heads, seq_len, head_dim, label in configs:
        q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        flops = 4 * batch * heads * seq_len * seq_len * head_dim

        # 1. SDPA Flash (baseline)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            r = benchmark_attention(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
            tflops = flops / (r["avg_ms"] / 1000) / 1e12
            print(f"{label:<15} {'SDPA Flash':<25} {r['avg_ms']:>10.3f} {tflops:>10.1f} {r['mem_gb']:>10.2f}")
            sdpa_ms = r["avg_ms"]
        except Exception as e:
            print(f"{label:<15} {'SDPA Flash':<25} {'FAIL':>10} -- {str(e)[:30]}")
            sdpa_ms = None
        finally:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        # 2. FlexAttention causal
        try:
            block_mask = create_block_mask(causal_mask, B=batch, H=heads, Q_LEN=seq_len, KV_LEN=seq_len)
            compiled_flex = torch.compile(
                lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask)
            )
            r = benchmark_attention(lambda: compiled_flex(q, k, v))
            tflops = flops / (r["avg_ms"] / 1000) / 1e12
            speedup = f" ({sdpa_ms/r['avg_ms']:.2f}x)" if sdpa_ms else ""
            print(f"{label:<15} {'FlexAttn causal':<25} {r['avg_ms']:>10.3f} {tflops:>10.1f} {r['mem_gb']:>10.2f}{speedup}")
        except Exception as e:
            print(f"{label:<15} {'FlexAttn causal':<25} {'FAIL':>10} -- {str(e)[:40]}")

        # 3. FlexAttention sliding window (256)
        try:
            sw_mask = create_block_mask(sliding_window_mask, B=batch, H=heads, Q_LEN=seq_len, KV_LEN=seq_len)
            compiled_sw = torch.compile(
                lambda q, k, v: flex_attention(q, k, v, block_mask=sw_mask)
            )
            r = benchmark_attention(lambda: compiled_sw(q, k, v))
            tflops = flops / (r["avg_ms"] / 1000) / 1e12
            print(f"{label:<15} {'FlexAttn slide-256':<25} {r['avg_ms']:>10.3f} {tflops:>10.1f} {r['mem_gb']:>10.2f}")
        except Exception as e:
            print(f"{label:<15} {'FlexAttn slide-256':<25} {'FAIL':>10} -- {str(e)[:40]}")

        # 4. FlexAttention with soft-cap score mod
        try:
            block_mask2 = create_block_mask(causal_mask, B=batch, H=heads, Q_LEN=seq_len, KV_LEN=seq_len)
            compiled_sc = torch.compile(
                lambda q, k, v: flex_attention(q, k, v, score_mod=soft_cap_score_mod, block_mask=block_mask2)
            )
            r = benchmark_attention(lambda: compiled_sc(q, k, v))
            tflops = flops / (r["avg_ms"] / 1000) / 1e12
            print(f"{label:<15} {'FlexAttn soft-cap':<25} {r['avg_ms']:>10.3f} {tflops:>10.1f} {r['mem_gb']:>10.2f}")
        except Exception as e:
            print(f"{label:<15} {'FlexAttn soft-cap':<25} {'FAIL':>10} -- {str(e)[:40]}")

        print()
        del q, k, v
        torch.cuda.empty_cache()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
