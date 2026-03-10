"""NCCL Collectives Benchmark on 4x A100 NVLink.

Tests AllReduce, AllGather, ReduceScatter at various tensor sizes.
Reports bandwidth and latency for each operation.
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


def benchmark_collective(name, fn, tensor_sizes, rank, warmup=5, iters=50):
    """Benchmark a collective operation at various tensor sizes."""
    results = []
    for size_mb in tensor_sizes:
        numel = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
        tensor = torch.randn(numel, device=f"cuda:{rank}", dtype=torch.float32)

        # Warmup
        for _ in range(warmup):
            fn(tensor)
        torch.cuda.synchronize()

        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            start_events[i].record()
            fn(tensor)
            end_events[i].record()

        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_ms = sum(times_ms) / len(times_ms)
        min_ms = min(times_ms)

        # Calculate bus bandwidth
        world_size = dist.get_world_size()
        data_bytes = numel * 4
        if name == "AllReduce":
            algo_bw = data_bytes * 2 * (world_size - 1) / world_size
        elif name == "AllGather":
            algo_bw = data_bytes * (world_size - 1) / world_size
        elif name == "ReduceScatter":
            algo_bw = data_bytes * (world_size - 1) / world_size
        elif name == "Broadcast":
            algo_bw = data_bytes
        else:
            algo_bw = data_bytes

        bus_bw_gbps = algo_bw / (avg_ms / 1000) / 1e9

        results.append({
            "size_mb": size_mb,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "bus_bw_gbps": bus_bw_gbps,
        })

    return results


def main():
    rank = setup()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"NCCL Collectives Benchmark")
        print(f"World size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"NCCL: {torch.cuda.nccl.version()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*80}")

    tensor_sizes_mb = [0.001, 0.01, 0.1, 1, 8, 32, 64, 128, 256, 512]

    # AllReduce
    def allreduce_fn(t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    results = benchmark_collective("AllReduce", allreduce_fn, tensor_sizes_mb, rank)
    if rank == 0:
        print(f"\n{'AllReduce':=^80}")
        print(f"{'Size (MB)':>12} {'Avg (ms)':>12} {'Min (ms)':>12} {'Bus BW (GB/s)':>15}")
        for r in results:
            print(f"{r['size_mb']:>12.3f} {r['avg_ms']:>12.3f} {r['min_ms']:>12.3f} {r['bus_bw_gbps']:>15.2f}")

    # AllGather
    def allgather_fn(t):
        output = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(output, t)

    results = benchmark_collective("AllGather", allgather_fn, tensor_sizes_mb, rank)
    if rank == 0:
        print(f"\n{'AllGather':=^80}")
        print(f"{'Size (MB)':>12} {'Avg (ms)':>12} {'Min (ms)':>12} {'Bus BW (GB/s)':>15}")
        for r in results:
            print(f"{r['size_mb']:>12.3f} {r['avg_ms']:>12.3f} {r['min_ms']:>12.3f} {r['bus_bw_gbps']:>15.2f}")

    # ReduceScatter
    def reducescatter_fn(t):
        chunk = t.numel() // world_size
        output = torch.empty(chunk, device=t.device, dtype=t.dtype)
        input_list = list(t.chunk(world_size))
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

    # Only test sizes divisible by world_size
    rs_sizes = [s for s in tensor_sizes_mb if int(s * 1024 * 1024 / 4) % world_size == 0]
    results = benchmark_collective("ReduceScatter", reducescatter_fn, rs_sizes, rank)
    if rank == 0:
        print(f"\n{'ReduceScatter':=^80}")
        print(f"{'Size (MB)':>12} {'Avg (ms)':>12} {'Min (ms)':>12} {'Bus BW (GB/s)':>15}")
        for r in results:
            print(f"{r['size_mb']:>12.3f} {r['avg_ms']:>12.3f} {r['min_ms']:>12.3f} {r['bus_bw_gbps']:>15.2f}")

    # Broadcast
    def broadcast_fn(t):
        dist.broadcast(t, src=0)

    results = benchmark_collective("Broadcast", broadcast_fn, tensor_sizes_mb, rank)
    if rank == 0:
        print(f"\n{'Broadcast':=^80}")
        print(f"{'Size (MB)':>12} {'Avg (ms)':>12} {'Min (ms)':>12} {'Bus BW (GB/s)':>15}")
        for r in results:
            print(f"{r['size_mb']:>12.3f} {r['avg_ms']:>12.3f} {r['min_ms']:>12.3f} {r['bus_bw_gbps']:>15.2f}")

    if rank == 0:
        print(f"\n{'='*80}")
        print("Benchmark complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
