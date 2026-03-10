"""Tensor Parallelism Inference Benchmark on 4x A100 NVLink.

Tests model inference with tensor parallelism using PyTorch's
DTensor and DeviceMesh APIs (PyTorch 2.8+).

Measures: 1-GPU vs 2-GPU vs 4-GPU TP for various model sizes.
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


class LargeTransformerLM(nn.Module):
    """Transformer LM sized for TP testing."""
    def __init__(self, vocab_size: int = 32000, d_model: int = 4096,
                 nhead: int = 32, num_layers: int = 32, dim_ff: int = 11008):
        super().__init__()
        self.d_model = d_model
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


def benchmark_inference(model, input_ids, rank, warmup=5, iters=20):
    """Benchmark inference throughput."""
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

    tok_s = (iters * input_ids.numel()) / elapsed
    ms_step = (elapsed / iters) * 1000
    mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9
    torch.cuda.reset_peak_memory_stats(rank)
    return tok_s, ms_step, mem_gb


def main():
    rank = setup()
    world_size = dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = True

    if rank == 0:
        print("Tensor Parallelism / FSDP Inference Benchmark")
        print(f"World size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Test different model sizes
    configs = [
        # (d_model, nhead, layers, dim_ff, label)
        (2048, 32, 24, 8192, "~1.3B"),
        (4096, 32, 16, 11008, "~3B"),
        (4096, 32, 32, 11008, "~7B"),
    ]

    batch_size = 1
    seq_len = 512

    for d_model, nhead, layers, dim_ff, label in configs:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Model: {label} (d={d_model}, L={layers})")

        try:
            model = LargeTransformerLM(
                d_model=d_model, nhead=nhead,
                num_layers=layers, dim_ff=dim_ff,
            ).to(rank)

            params = sum(p.numel() for p in model.parameters())
            param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

            if rank == 0:
                print(f"  Params: {params:,} ({param_gb:.2f} GB in FP32)")

            # Wrap with FSDP (shards across all GPUs)
            model = FSDP(model, mixed_precision=bf16_policy)
            model.eval()

            input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")
            tok_s, ms_step, mem_gb = benchmark_inference(model, input_ids, rank)

            if rank == 0:
                print(f"  FSDP ({world_size}-GPU) BF16:")
                print(f"    {tok_s:,.0f} tok/s | {ms_step:.1f} ms/step | {mem_gb:.2f} GB/GPU")

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"  OOM at {label}!")
        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")
        finally:
            if "model" in dir():
                del model
            torch.cuda.empty_cache()
            dist.barrier()

    # ---- Single GPU comparison (rank 0 only) ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("Single GPU baselines (rank 0 only):")
        for d_model, nhead, layers, dim_ff, label in configs:
            try:
                model = LargeTransformerLM(
                    d_model=d_model, nhead=nhead,
                    num_layers=layers, dim_ff=dim_ff,
                ).cuda().to(torch.bfloat16).eval()
                input_ids = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")
                tok_s, ms_step, mem_gb = benchmark_inference(model, input_ids, 0)
                print(f"  {label} single GPU: {tok_s:,.0f} tok/s | "
                      f"{ms_step:.1f} ms | {mem_gb:.2f} GB")
                del model
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  {label} single GPU: OOM")
                torch.cuda.empty_cache()

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        print("Benchmark complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
