"""Context Parallelism (Ring Attention) Benchmark on 4x A100.

Tests PyTorch 2.8 native context parallelism for long sequences.
Compares standard attention vs context-parallel attention.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def bench_attention(fn, warmup: int = 5, iters: int = 20):
    """Benchmark attention operation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return avg_ms, mem_gb


def main():
    rank = setup()
    world_size = dist.get_world_size()

    if rank == 0:
        print("Context Parallelism Benchmark")
        print(f"World size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    # Check if context_parallel is available
    try:
        from torch.distributed.device_mesh import init_device_mesh
        cp_available = True
    except ImportError:
        cp_available = False

    try:
        from torch.nn.attention import context_parallel
        has_cp_api = True
    except ImportError:
        has_cp_api = False
        if rank == 0:
            print("torch.nn.attention.context_parallel not available")
            print("Testing standard distributed attention only")

    batch = 1
    heads = 32
    head_dim = 64
    seq_lengths = [2048, 4096, 8192, 16384]

    if rank == 0:
        print(f"{'Seq Len':<10} {'Method':<25} {'Avg (ms)':>10} {'Mem/GPU (GB)':>14}")
        print("-" * 65)

    for seq_len in seq_lengths:
        # Standard SDPA (each GPU runs full sequence independently)
        try:
            q = torch.randn(batch, heads, seq_len, head_dim, device=f"cuda:{rank}", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            avg_ms, mem_gb = bench_attention(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
            )
            if rank == 0:
                print(f"{seq_len:<10} {'Standard SDPA':<25} {avg_ms:>10.2f} {mem_gb:>14.2f}")
            del q, k, v
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"{seq_len:<10} {'Standard SDPA':<25} {'OOM':>10}")
            torch.cuda.empty_cache()

        # Context Parallel (split sequence across GPUs)
        if has_cp_api:
            try:
                cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
                # Each GPU gets seq_len/world_size of the sequence
                local_seq = seq_len // world_size
                q = torch.randn(batch, heads, local_seq, head_dim, device=f"cuda:{rank}", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                def run_cp():
                    with context_parallel(cp_mesh):
                        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

                avg_ms, mem_gb = bench_attention(run_cp)
                if rank == 0:
                    print(f"{seq_len:<10} {'Context Parallel (CP)':<25} {avg_ms:>10.2f} {mem_gb:>14.2f}")
                del q, k, v
                torch.cuda.empty_cache()
            except Exception as e:
                if rank == 0:
                    print(f"{seq_len:<10} {'Context Parallel (CP)':<25} {'FAIL':>10} -- {str(e)[:40]}")
                torch.cuda.empty_cache()
        else:
            # Simulate CP by just running on local shard
            try:
                local_seq = seq_len // world_size
                q = torch.randn(batch, heads, local_seq, head_dim, device=f"cuda:{rank}", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                avg_ms, mem_gb = bench_attention(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                )
                if rank == 0:
                    print(f"{seq_len:<10} {'Sharded SDPA (seq/{world_size})':<25} {avg_ms:>10.2f} {mem_gb:>14.2f}")
                del q, k, v
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"{seq_len:<10} {'Sharded SDPA':<25} {'OOM':>10}")
                torch.cuda.empty_cache()

        if rank == 0:
            print()
        dist.barrier()

    # Try very long sequences with CP
    if has_cp_api and rank == 0:
        print(f"\n{'='*60}")
        print("Long sequence test (CP only):")

    for seq_len in [32768, 65536]:
        if has_cp_api:
            try:
                cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
                local_seq = seq_len // world_size
                q = torch.randn(batch, heads, local_seq, head_dim, device=f"cuda:{rank}", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                def run_cp_long():
                    with context_parallel(cp_mesh):
                        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

                avg_ms, mem_gb = bench_attention(run_cp_long, warmup=3, iters=10)
                if rank == 0:
                    print(f"{seq_len:<10} {'Context Parallel':<25} {avg_ms:>10.2f} {mem_gb:>14.2f}")
                del q, k, v
                torch.cuda.empty_cache()
            except Exception as e:
                if rank == 0:
                    print(f"{seq_len:<10} {'Context Parallel':<25} {'FAIL':>10} -- {str(e)[:40]}")
                torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()
    if rank == 0:
        print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
