"""Activation Checkpointing Benchmark for LLM Training.

Compares: No checkpointing, full checkpointing, selective checkpointing.
Shows memory vs compute tradeoff at various model sizes.
"""
import os
import time
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int):
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


class CheckpointableLLM(nn.Module):
    """LLM with optional activation checkpointing per block."""
    def __init__(self, vocab_size: int = 32000, d_model: int = 1024,
                 nhead: int = 16, num_layers: int = 24, dim_ff: int = 4096,
                 checkpoint_mode: str = "none"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.checkpoint_mode = checkpoint_mode

    def forward(self, x):
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            if self.checkpoint_mode == "full":
                h = checkpoint(block, h, use_reentrant=False)
            elif self.checkpoint_mode == "selective":
                # Checkpoint every other block
                if i % 2 == 0:
                    h = checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            else:
                h = block(h)
        return self.head(h)


def benchmark_training(model, batch_size, seq_len, rank, label, warmup=3, iters=15):
    """Run training steps and measure throughput + memory."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(warmup + iters):
        data = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")
        target = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{rank}")

        if step == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(rank)
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
    ms_step = (elapsed / iters) * 1000
    mem_gb = torch.cuda.max_memory_allocated(rank) / 1e9

    if rank == 0:
        print(f"  [{label}]")
        print(f"    Throughput: {tok_s:,.0f} tok/s | Step: {ms_step:.1f} ms | Mem: {mem_gb:.2f} GB")

    return tok_s, ms_step, mem_gb


def main():
    rank = setup()
    torch.backends.cuda.matmul.allow_tf32 = True

    if rank == 0:
        print("Activation Checkpointing Benchmark")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Model configs: (layers, d_model, nhead, dim_ff, label)
    model_configs = [
        (24, 1024, 16, 4096, "218M"),
        (24, 2048, 32, 8192, "830M"),
    ]

    batch_size = 4
    seq_len = 1024

    for num_layers, d_model, nhead, dim_ff, size_label in model_configs:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Model: {size_label} ({num_layers}L, d={d_model})")
            print(f"Batch: {batch_size}, Seq: {seq_len}")

        results = []

        for ckpt_mode in ["none", "selective", "full"]:
            if rank == 0:
                print(f"\n  Checkpoint mode: {ckpt_mode}")

            try:
                model = CheckpointableLLM(
                    d_model=d_model, nhead=nhead, num_layers=num_layers,
                    dim_ff=dim_ff, checkpoint_mode=ckpt_mode,
                ).to(rank)
                model = FSDP(model, mixed_precision=bf16_policy)

                tok_s, ms_step, mem_gb = benchmark_training(
                    model, batch_size, seq_len, rank,
                    f"FSDP+BF16+ckpt={ckpt_mode}",
                )
                results.append((ckpt_mode, tok_s, ms_step, mem_gb))
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"    OOM! Skipping.")
                results.append((ckpt_mode, 0, 0, 0))
            finally:
                if "model" in dir():
                    del model
                torch.cuda.empty_cache()
                dist.barrier()

        if rank == 0:
            print(f"\n  Summary for {size_label}:")
            print(f"  {'Mode':<12} {'tok/s':>10} {'ms/step':>10} {'Mem (GB)':>10} {'Mem Save':>10}")
            base_mem = results[0][3] if results[0][3] > 0 else 1
            for mode, tps, ms, mem in results:
                if tps > 0:
                    save = (1 - mem / base_mem) * 100 if base_mem > 0 else 0
                    print(f"  {mode:<12} {tps:>10,.0f} {ms:>10.1f} {mem:>10.2f} {save:>9.1f}%")
                else:
                    print(f"  {mode:<12} {'OOM':>10}")

    if rank == 0:
        print(f"\n{'='*60}")
        print("Benchmark complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
