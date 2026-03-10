"""PyTorch FSDP + DDP Training Benchmark on 4x A100.

Compares: single GPU, DDP, FSDP on a transformer model.
Tests: BF16, TF32, torch.compile, FlashAttention (if available).
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class SimpleTransformer(nn.Module):
    """A transformer model sized to stress 4x40GB A100s."""
    def __init__(self, vocab_size=32000, d_model=1024, nhead=16,
                 num_layers=12, dim_ff=4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        x = self.transformer(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_training(model, batch_size, seq_len, rank, label, warmup=3, iters=20):
    """Run training steps and measure throughput."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(warmup + iters):
        data = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")
        target = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")

        if step == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.view(-1, 32000), target.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (iters * batch_size * seq_len) / elapsed
    samples_per_sec = (iters * batch_size) / elapsed
    ms_per_step = (elapsed / iters) * 1000

    if rank == 0:
        mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9
        print(f"  [{label}]")
        print(f"    Throughput: {tokens_per_sec:,.0f} tok/s | {samples_per_sec:.1f} samples/s")
        print(f"    Step time:  {ms_per_step:.1f} ms")
        print(f"    Peak mem:   {mem_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats(rank)

    return tokens_per_sec


def main():
    rank = setup()
    world_size = dist.get_world_size()
    batch_size = 8
    seq_len = 512

    if rank == 0:
        print(f"FSDP/DDP Training Benchmark")
        print(f"World size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
        print(f"Batch size: {batch_size}, Seq len: {seq_len}")

        # Check for flash attention
        try:
            from torch.nn.functional import scaled_dot_product_attention
            print(f"FlashAttention/SDPA: available (PyTorch native)")
        except ImportError:
            print(f"FlashAttention/SDPA: not available")

        # Check TF32
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    # ---- Test 1: DDP with FP32 ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 1: DDP + FP32")
    model = SimpleTransformer().to(rank)
    if rank == 0:
        print(f"  Model params: {count_params(model):,}")
    model = DDP(model, device_ids=[rank])
    benchmark_training(model, batch_size, seq_len, rank, "DDP-FP32")
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- Test 2: DDP with TF32 ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 2: DDP + TF32")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = SimpleTransformer().to(rank)
    model = DDP(model, device_ids=[rank])
    benchmark_training(model, batch_size, seq_len, rank, "DDP-TF32")
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- Test 3: DDP with BF16 autocast ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 3: DDP + BF16 (autocast)")
    model = SimpleTransformer().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    warmup, iters = 3, 20

    for step in range(warmup + iters):
        data = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")
        target = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")
        if step == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(data)
            loss = loss_fn(output.view(-1, 32000), target.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tok_s = (iters * batch_size * seq_len) / elapsed

    if rank == 0:
        mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9
        print(f"  [DDP-BF16]")
        print(f"    Throughput: {tok_s:,.0f} tok/s")
        print(f"    Step time:  {(elapsed / iters) * 1000:.1f} ms")
        print(f"    Peak mem:   {mem_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats(rank)

    del model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- Test 4: FSDP with BF16 ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 4: FSDP + BF16 mixed precision")
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = SimpleTransformer().to(rank)
    model = FSDP(model, mixed_precision=bf16_policy)
    benchmark_training(model, batch_size, seq_len, rank, "FSDP-BF16")
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- Test 5: FSDP + BF16 + torch.compile ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 5: FSDP + BF16 + torch.compile")
    try:
        model = SimpleTransformer().to(rank)
        model = torch.compile(model)
        model = FSDP(model, mixed_precision=bf16_policy)
        benchmark_training(model, batch_size, seq_len, rank, "FSDP-BF16-compiled",
                          warmup=5, iters=20)
    except Exception as e:
        if rank == 0:
            print(f"  torch.compile failed: {e}")
    finally:
        del model
        torch.cuda.empty_cache()
        dist.barrier()

    # ---- Summary ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
