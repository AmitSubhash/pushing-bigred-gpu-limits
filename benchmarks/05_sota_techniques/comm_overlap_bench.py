"""Communication-Computation Overlap Benchmark on 4x A100.

Tests whether overlapping NCCL all-reduce with GPU compute
provides throughput improvement over synchronous communication.
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def bench_sync_vs_async(tensor_size_mb: float, compute_iters: int = 100) -> dict:
    """Compare synchronous vs async all-reduce with compute overlap."""
    rank = dist.get_rank()
    numel = int(tensor_size_mb * 1024 * 1024 / 4)
    comm_tensor = torch.randn(numel, device=f"cuda:{rank}")
    compute_a = torch.randn(1024, 1024, device=f"cuda:{rank}", dtype=torch.bfloat16)
    compute_b = torch.randn(1024, 1024, device=f"cuda:{rank}", dtype=torch.bfloat16)

    warmup = 5
    iters = 20

    # Synchronous: all-reduce then compute
    for _ in range(warmup):
        dist.all_reduce(comm_tensor)
        for _ in range(compute_iters):
            _ = torch.matmul(compute_a, compute_b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(comm_tensor)
        for _ in range(compute_iters):
            _ = torch.matmul(compute_a, compute_b)
    torch.cuda.synchronize()
    sync_time = (time.perf_counter() - start) / iters

    # Async: overlap all-reduce with compute
    for _ in range(warmup):
        handle = dist.all_reduce(comm_tensor, async_op=True)
        for _ in range(compute_iters):
            _ = torch.matmul(compute_a, compute_b)
        handle.wait()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        handle = dist.all_reduce(comm_tensor, async_op=True)
        for _ in range(compute_iters):
            _ = torch.matmul(compute_a, compute_b)
        handle.wait()
    torch.cuda.synchronize()
    async_time = (time.perf_counter() - start) / iters

    return {
        "sync_ms": sync_time * 1000,
        "async_ms": async_time * 1000,
        "speedup": sync_time / async_time,
    }


class TransformerLM(nn.Module):
    def __init__(self, vocab: int = 32000, d: int = 1024, heads: int = 16,
                 layers: int = 12, ff: int = 4096):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        return self.head(self.transformer(self.embed(x)))


def bench_fsdp_with_overlap(rank, use_cuda_max_conn=False):
    """Benchmark FSDP training with/without CUDA_DEVICE_MAX_CONNECTIONS=1."""
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = TransformerLM().to(rank)
    model = FSDP(model, mixed_precision=bf16_policy)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch, seq = 8, 512
    input_ids = torch.randint(0, 32000, (batch, seq), device=f"cuda:{rank}")
    labels = torch.randint(0, 32000, (batch, seq), device=f"cuda:{rank}")
    loss_fn = nn.CrossEntropyLoss()

    warmup, iters = 5, 20

    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tok_s = iters * input_ids.numel() / elapsed
    mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9
    torch.cuda.reset_peak_memory_stats(rank)

    del model, opt
    torch.cuda.empty_cache()
    return tok_s, mem_gb


def main():
    rank = setup()

    if rank == 0:
        print("Communication-Computation Overlap Benchmark")
        print(f"World size: {dist.get_world_size()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_DEVICE_MAX_CONNECTIONS: {os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'default')}")
        print()

    # Part 1: Raw async vs sync all-reduce with compute
    if rank == 0:
        print("Part 1: Raw AllReduce + Compute Overlap")
        print(f"{'Comm Size':<12} {'Sync (ms)':>10} {'Async (ms)':>10} {'Overlap Gain':>14}")
        print("-" * 50)

    for size_mb in [1, 16, 64, 256]:
        for compute_iters in [50, 200]:
            r = bench_sync_vs_async(size_mb, compute_iters)
            if rank == 0:
                label = f"{size_mb}MB+{compute_iters}mm"
                print(f"{label:<12} {r['sync_ms']:>10.2f} {r['async_ms']:>10.2f} {r['speedup']:>13.2f}x")

    # Part 2: FSDP training throughput
    if rank == 0:
        print(f"\nPart 2: FSDP Training Throughput")
        print(f"{'Config':<35} {'tok/s':>10} {'Mem (GB)':>10}")
        print("-" * 60)

    tok_s, mem = bench_fsdp_with_overlap(rank)
    if rank == 0:
        conn_val = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "default")
        label = f"FSDP+BF16 (MAX_CONN={conn_val})"
        print(f"{label:<35} {tok_s:>10,.0f} {mem:>10.2f}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
