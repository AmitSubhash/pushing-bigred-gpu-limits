"""Liger-Kernel Fused Ops Benchmark on A100.

Compares standard PyTorch ops vs Liger-Kernel fused Triton kernels
for RMSNorm, CrossEntropy, and FusedLinearCrossEntropy.
"""
import time
import torch
import torch.nn as nn


def bench_op(fn, warmup: int = 20, iters: int = 100) -> dict:
    """Benchmark a single operation."""
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


class PyTorchRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def main():
    torch.cuda.set_device(0)
    print("Liger-Kernel Fused Ops Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import liger_kernel
        print(f"Liger-Kernel: installed")
        HAS_LIGER = True
    except ImportError:
        print("Liger-Kernel NOT installed -- showing PyTorch baselines only")
        HAS_LIGER = False

    print()

    hidden = 4096
    vocab = 32000
    batch, seq = 4, 2048

    print(f"{'Operation':<30} {'PyTorch (ms)':>14} {'Liger (ms)':>14} {'Speedup':>10} {'PT Mem':>10} {'LG Mem':>10}")
    print("-" * 95)

    # 1. RMSNorm
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pt_norm = PyTorchRMSNorm(hidden).cuda().bfloat16()
    r_pt = bench_op(lambda: pt_norm(x).sum().backward(retain_graph=True))

    if HAS_LIGER:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        lg_norm = LigerRMSNorm(hidden).cuda().bfloat16()
        x2 = x.detach().clone().requires_grad_(True)
        r_lg = bench_op(lambda: lg_norm(x2).sum().backward(retain_graph=True))
        speedup = r_pt["avg_ms"] / r_lg["avg_ms"]
        print(f"{'RMSNorm (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {r_lg['avg_ms']:>14.3f} {speedup:>10.2f}x {r_pt['mem_gb']:>10.2f} {r_lg['mem_gb']:>10.2f}")
    else:
        print(f"{'RMSNorm (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {'N/A':>14} {'N/A':>10} {r_pt['mem_gb']:>10.2f} {'N/A':>10}")

    del x
    torch.cuda.empty_cache()

    # 2. CrossEntropy
    logits = torch.randn(batch * seq, vocab, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, vocab, (batch * seq,), device="cuda")
    ce = nn.CrossEntropyLoss()
    r_pt = bench_op(lambda: ce(logits, targets).backward(retain_graph=True))

    if HAS_LIGER:
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
        lg_ce = LigerCrossEntropyLoss()
        logits2 = logits.detach().clone().requires_grad_(True)
        r_lg = bench_op(lambda: lg_ce(logits2, targets).backward(retain_graph=True))
        speedup = r_pt["avg_ms"] / r_lg["avg_ms"]
        print(f"{'CrossEntropy (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {r_lg['avg_ms']:>14.3f} {speedup:>10.2f}x {r_pt['mem_gb']:>10.2f} {r_lg['mem_gb']:>10.2f}")
    else:
        print(f"{'CrossEntropy (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {'N/A':>14} {'N/A':>10} {r_pt['mem_gb']:>10.2f} {'N/A':>10}")

    del logits, targets
    torch.cuda.empty_cache()

    # 3. FusedLinearCrossEntropy (lm_head + CE in one kernel)
    hidden_states = torch.randn(batch * seq, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lm_head = nn.Linear(hidden, vocab, bias=False).cuda().bfloat16()
    targets3 = torch.randint(0, vocab, (batch * seq,), device="cuda")

    def pt_linear_ce():
        logits = lm_head(hidden_states)
        return nn.functional.cross_entropy(logits, targets3)

    r_pt = bench_op(lambda: pt_linear_ce().backward(retain_graph=True))

    if HAS_LIGER:
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            lg_flce = LigerFusedLinearCrossEntropyLoss()
            hs2 = hidden_states.detach().clone().requires_grad_(True)
            w = lm_head.weight.detach().clone().requires_grad_(True)

            def lg_fused():
                return lg_flce(hs2, w, targets3)

            r_lg = bench_op(lambda: lg_fused().backward(retain_graph=True))
            speedup = r_pt["avg_ms"] / r_lg["avg_ms"]
            print(f"{'FusedLinearCE (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {r_lg['avg_ms']:>14.3f} {speedup:>10.2f}x {r_pt['mem_gb']:>10.2f} {r_lg['mem_gb']:>10.2f}")
        except Exception as e:
            print(f"{'FusedLinearCE (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {'FAIL':>14} -- {str(e)[:40]}")
    else:
        print(f"{'FusedLinearCE (fwd+bwd)':<30} {r_pt['avg_ms']:>14.3f} {'N/A':>14} {'N/A':>10} {r_pt['mem_gb']:>10.2f} {'N/A':>10}")

    # 4. End-to-end training step comparison (with monkey-patch)
    if HAS_LIGER:
        print(f"\n{'='*60}")
        print("End-to-end training step (218M param transformer)")
        print(f"{'Config':<30} {'tok/s':>10} {'Mem (GB)':>10}")
        print("-" * 55)

        from liger_kernel.transformers import _apply_liger_kernel

        for label, use_liger in [("Standard PyTorch", False), ("+ Liger-Kernel", True)]:
            try:
                model = nn.Sequential(
                    nn.Embedding(vocab, 1024),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(1024, 16, 4096, batch_first=True, dropout=0.0),
                        num_layers=12,
                    ),
                    nn.Linear(1024, vocab, bias=False),
                ).cuda().bfloat16()

                if use_liger:
                    # Replace norms with Liger versions
                    from liger_kernel.transformers.rms_norm import LigerRMSNorm
                    for layer in model[1].layers:
                        layer.norm1 = LigerRMSNorm(1024).cuda().bfloat16()
                        layer.norm2 = LigerRMSNorm(1024).cuda().bfloat16()

                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                ids = torch.randint(0, vocab, (8, 512), device="cuda")
                tgt = torch.randint(0, vocab, (8, 512), device="cuda")

                for _ in range(3):
                    out = model(ids)
                    loss = nn.functional.cross_entropy(out.view(-1, vocab), tgt.view(-1))
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(15):
                    out = model(ids)
                    loss = nn.functional.cross_entropy(out.view(-1, vocab), tgt.view(-1))
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                tok_s = 15 * ids.numel() / elapsed
                mem = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                print(f"{label:<30} {tok_s:>10,.0f} {mem:>10.2f}")

                del model, opt
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{label:<30} {'FAIL':>10} -- {str(e)[:50]}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
