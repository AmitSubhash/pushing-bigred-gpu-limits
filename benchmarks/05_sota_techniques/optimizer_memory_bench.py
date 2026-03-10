"""Memory-Efficient Optimizer Benchmark: AdamW vs 8-bit AdamW vs Adafactor.

Tests optimizer memory savings and training throughput on FSDP + BF16.
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


class TransformerLM(nn.Module):
    def __init__(self, vocab: int = 32000, d: int = 2048, heads: int = 32,
                 layers: int = 24, ff: int = 8192):
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


def train_loop(model, optimizer, input_ids, labels, warmup=3, iters=15):
    """Training loop measuring throughput and memory."""
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = loss.item()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tok_s = iters * input_ids.numel() / elapsed
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return tok_s, mem_gb, final_loss


def main():
    rank = setup()
    world_size = dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = True

    if rank == 0:
        print("Optimizer Memory Benchmark (FSDP + BF16)")
        print(f"World size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    batch, seq = 4, 512
    input_ids = torch.randint(0, 32000, (batch, seq), device=f"cuda:{rank}")
    labels = torch.randint(0, 32000, (batch, seq), device=f"cuda:{rank}")

    optimizers = [
        ("AdamW (standard)", lambda p: torch.optim.AdamW(p, lr=1e-4)),
    ]

    # 8-bit AdamW
    try:
        import bitsandbytes as bnb
        optimizers.append(
            ("AdamW8bit (bnb)", lambda p: bnb.optim.AdamW8bit(p, lr=1e-4))
        )
        if rank == 0:
            print(f"bitsandbytes {bnb.__version__} loaded")
    except ImportError:
        if rank == 0:
            print("bitsandbytes not available, skipping 8-bit AdamW")

    # Adafactor
    try:
        from transformers import Adafactor
        optimizers.append(
            ("Adafactor", lambda p: Adafactor(p, lr=1e-4, scale_parameter=False,
                                               relative_step=False, warmup_init=False))
        )
    except ImportError:
        if rank == 0:
            print("transformers Adafactor not available")

    # GaLore
    try:
        from galore_torch import GaLoreAdamW8bit
        def make_galore(params):
            param_groups = [
                {"params": [p for p in params if p.dim() < 2], "rank": 0},
                {"params": [p for p in params if p.dim() >= 2], "rank": 128,
                 "update_proj_gap": 200, "scale": 0.25},
            ]
            return GaLoreAdamW8bit(param_groups, lr=1e-4)
        optimizers.append(("GaLore-8bit", make_galore))
        if rank == 0:
            print("galore-torch loaded")
    except ImportError:
        if rank == 0:
            print("galore-torch not available, skipping GaLore")

    if rank == 0:
        print(f"\nModel: ~1.3B params (d=2048, L=24, ff=8192)")
        print(f"{'Optimizer':<25} {'tok/s':>10} {'Mem/GPU (GB)':>14} {'Final Loss':>12}")
        print("-" * 65)

    for name, opt_fn in optimizers:
        try:
            model = TransformerLM().to(rank)
            model = FSDP(model, mixed_precision=bf16_policy)
            model.train()

            params = list(model.parameters())
            if "GaLore" in name:
                optimizer = opt_fn(params)
            else:
                optimizer = opt_fn(params)

            tok_s, mem_gb, final_loss = train_loop(model, optimizer, input_ids, labels)

            if rank == 0:
                print(f"{name:<25} {tok_s:>10,.0f} {mem_gb:>14.2f} {final_loss:>12.4f}")

            del model, optimizer
            torch.cuda.empty_cache()
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"{name:<25} {'FAIL':>10} -- {str(e)[:50]}")
            torch.cuda.empty_cache()
            dist.barrier()

    dist.destroy_process_group()
    if rank == 0:
        print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
