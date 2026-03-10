"""CUDA Graphs for LLM Inference on A100.

CUDA Graphs capture a sequence of GPU operations and replay them
without CPU overhead. This eliminates kernel launch latency which
dominates small-batch inference.

Tests: with/without CUDA graphs at various batch sizes.
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


class LLMBlock(nn.Module):
    """Simplified transformer block for benchmarking."""
    def __init__(self, d_model: int = 1024, nhead: int = 16, dim_ff: int = 4096):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class SmallLLM(nn.Module):
    """Stack of transformer blocks for inference benchmarking."""
    def __init__(self, num_layers: int = 12, d_model: int = 1024,
                 nhead: int = 16, vocab_size: int = 32000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([LLMBlock(d_model, nhead) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def benchmark_inference(model, input_ids, label, warmup=10, iters=100):
    """Benchmark inference without CUDA graphs."""
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = (elapsed / iters) * 1000
    tokens_per_sec = (iters * input_ids.numel()) / elapsed
    return ms_per_step, tokens_per_sec


def benchmark_cuda_graphs(model, input_ids, label, warmup=10, iters=100):
    """Benchmark inference WITH CUDA graphs."""
    # Warmup and capture
    static_input = input_ids.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                _ = model(static_input)
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_output = model(static_input)

    # Warmup replay
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    # Benchmark replay
    start = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = (elapsed / iters) * 1000
    tokens_per_sec = (iters * input_ids.numel()) / elapsed
    return ms_per_step, tokens_per_sec


def main():
    rank = setup()

    if rank == 0:
        print("CUDA Graphs Inference Benchmark")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    model = SmallLLM(num_layers=12, d_model=1024).cuda().eval().half()
    param_count = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"Model params: {param_count:,}")
        print()

    configs = [
        (1, 1, "Decode: batch=1, seq=1"),
        (1, 32, "Decode: batch=1, seq=32"),
        (1, 128, "Prefill: batch=1, seq=128"),
        (4, 128, "Prefill: batch=4, seq=128"),
        (8, 256, "Prefill: batch=8, seq=256"),
        (16, 512, "Prefill: batch=16, seq=512"),
    ]

    if rank == 0:
        print(f"{'Config':<35} {'No Graph (ms)':>14} {'CUDA Graph (ms)':>16} "
              f"{'Speedup':>8} {'tok/s (graph)':>14}")
        print("-" * 92)

    for batch, seq, label in configs:
        input_ids = torch.randint(0, 32000, (batch, seq), device="cuda")

        ms_no_graph, tps_no_graph = benchmark_inference(model, input_ids, label)

        try:
            ms_graph, tps_graph = benchmark_cuda_graphs(model, input_ids, label)
            speedup = ms_no_graph / ms_graph
            if rank == 0:
                print(f"{label:<35} {ms_no_graph:>14.3f} {ms_graph:>16.3f} "
                      f"{speedup:>7.2f}x {tps_graph:>14,.0f}")
        except Exception as e:
            if rank == 0:
                print(f"{label:<35} {ms_no_graph:>14.3f} {'FAILED':>16} "
                      f"{'--':>8} {'--':>14}  ({str(e)[:30]})")

        torch.cuda.empty_cache()

    if rank == 0:
        print()
        print("Benchmark complete.")


if __name__ == "__main__":
    main()
