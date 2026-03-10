"""NCCL Algorithm/Protocol Tuning Benchmark.

Tests different NCCL_ALGO and NCCL_PROTO combinations
to find optimal settings for 4x A100 NVLink.
"""
import os
import time
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def bench_allreduce(size_mb: float, warmup: int = 10, iters: int = 50) -> dict:
    """Benchmark AllReduce at a given message size."""
    rank = dist.get_rank()
    numel = int(size_mb * 1024 * 1024 / 4)  # float32
    tensor = torch.randn(numel, device=f"cuda:{rank}")

    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_s = elapsed / iters
    # Bus bandwidth = data_size * 2 * (n-1) / n / time (for ring allreduce)
    n = dist.get_world_size()
    data_bytes = numel * 4
    algbw = data_bytes / avg_s / 1e9
    busbw = algbw * 2 * (n - 1) / n

    return {"algbw_gbs": algbw, "busbw_gbs": busbw, "avg_us": avg_s * 1e6}


def main():
    rank = setup()

    # Read current NCCL settings
    algo = os.environ.get("NCCL_ALGO", "default")
    proto = os.environ.get("NCCL_PROTO", "default")
    max_conn = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "default")

    if rank == 0:
        print("NCCL Tuning Benchmark")
        print(f"World size: {dist.get_world_size()}")
        print(f"NCCL_ALGO={algo}, NCCL_PROTO={proto}")
        print(f"CUDA_DEVICE_MAX_CONNECTIONS={max_conn}")
        print(f"PyTorch: {torch.__version__}")
        print()

    sizes_mb = [0.001, 0.01, 0.1, 1, 4, 16, 64, 256]

    if rank == 0:
        print(f"{'Size':<10} {'AlgBW (GB/s)':>14} {'BusBW (GB/s)':>14} {'Latency (us)':>14}")
        print("-" * 56)

    for size in sizes_mb:
        try:
            r = bench_allreduce(size)
            if rank == 0:
                label = f"{size:.3f} MB" if size < 1 else f"{size:.0f} MB"
                print(f"{label:<10} {r['algbw_gbs']:>14.2f} {r['busbw_gbs']:>14.2f} {r['avg_us']:>14.1f}")
        except Exception as e:
            if rank == 0:
                print(f"{size} MB: FAIL -- {e}")

    dist.destroy_process_group()
    if rank == 0:
        print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
