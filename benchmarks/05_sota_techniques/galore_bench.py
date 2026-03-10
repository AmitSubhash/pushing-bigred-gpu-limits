"""GaLore (Gradient Low-Rank Projection) Optimizer Benchmark.

Compares GaLore optimizer memory usage and throughput against
standard AdamW and 8-bit AdamW.
"""
import time
import torch
import torch.nn as nn


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


def train_loop(model, optimizer, input_ids, labels, warmup=3, iters=15):
    """Training loop."""
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
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("GaLore Optimizer Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch, seq = 8, 512
    input_ids = torch.randint(0, 32000, (batch, seq), device="cuda")
    labels = torch.randint(0, 32000, (batch, seq), device="cuda")

    print(f"\nModel: ~218M params (d=1024, L=12)")
    print(f"{'Optimizer':<30} {'tok/s':>10} {'Mem (GB)':>10} {'Loss':>10}")
    print("-" * 65)

    # 1. Standard AdamW
    try:
        model = TransformerLM().cuda().bfloat16()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tok_s, mem, loss = train_loop(model, opt, input_ids, labels)
        print(f"{'AdamW (standard)':<30} {tok_s:>10,.0f} {mem:>10.2f} {loss:>10.4f}")
        del model, opt
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{'AdamW (standard)':<30} {'FAIL':>10} -- {str(e)[:40]}")

    # 2. 8-bit AdamW
    try:
        import bitsandbytes as bnb
        model = TransformerLM().cuda().bfloat16()
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
        tok_s, mem, loss = train_loop(model, opt, input_ids, labels)
        print(f"{'AdamW8bit (bnb)':<30} {tok_s:>10,.0f} {mem:>10.2f} {loss:>10.4f}")
        del model, opt
        torch.cuda.empty_cache()
    except ImportError:
        print(f"{'AdamW8bit':<30} {'SKIP':>10} -- bitsandbytes not installed")

    # 3. GaLore AdamW
    try:
        from galore_torch import GaLoreAdamW
        model = TransformerLM().cuda().bfloat16()

        # Separate params: small params get standard treatment, large linear params get GaLore
        linear_params = []
        other_params = []
        for name, p in model.named_parameters():
            if p.dim() >= 2 and p.size(0) >= 256 and p.size(1) >= 256:
                linear_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {"params": other_params, "rank": 0},
            {"params": linear_params, "rank": 128, "update_proj_gap": 200, "scale": 0.25},
        ]
        opt = GaLoreAdamW(param_groups, lr=1e-4)
        tok_s, mem, loss = train_loop(model, opt, input_ids, labels)
        print(f"{'GaLore AdamW (r=128)':<30} {tok_s:>10,.0f} {mem:>10.2f} {loss:>10.4f}")
        del model, opt
        torch.cuda.empty_cache()
    except ImportError:
        print(f"{'GaLore AdamW':<30} {'SKIP':>10} -- galore-torch not installed")
    except Exception as e:
        print(f"{'GaLore AdamW':<30} {'FAIL':>10} -- {str(e)[:50]}")

    # 4. GaLore 8-bit
    try:
        from galore_torch import GaLoreAdamW8bit
        model = TransformerLM().cuda().bfloat16()

        linear_params = []
        other_params = []
        for name, p in model.named_parameters():
            if p.dim() >= 2 and p.size(0) >= 256 and p.size(1) >= 256:
                linear_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {"params": other_params, "rank": 0},
            {"params": linear_params, "rank": 128, "update_proj_gap": 200, "scale": 0.25},
        ]
        opt = GaLoreAdamW8bit(param_groups, lr=1e-4)
        tok_s, mem, loss = train_loop(model, opt, input_ids, labels)
        print(f"{'GaLore AdamW8bit (r=128)':<30} {tok_s:>10,.0f} {mem:>10.2f} {loss:>10.4f}")
        del model, opt
        torch.cuda.empty_cache()
    except ImportError:
        print(f"{'GaLore AdamW8bit':<30} {'SKIP':>10} -- galore-torch not installed")
    except Exception as e:
        print(f"{'GaLore AdamW8bit':<30} {'FAIL':>10} -- {str(e)[:50]}")

    # 5. Adafactor
    try:
        from transformers import Adafactor
        model = TransformerLM().cuda().bfloat16()
        opt = Adafactor(model.parameters(), lr=1e-4, scale_parameter=False,
                        relative_step=False, warmup_init=False)
        tok_s, mem, loss = train_loop(model, opt, input_ids, labels)
        print(f"{'Adafactor':<30} {tok_s:>10,.0f} {mem:>10.2f} {loss:>10.4f}")
        del model, opt
        torch.cuda.empty_cache()
    except ImportError:
        print(f"{'Adafactor':<30} {'SKIP':>10} -- transformers not installed")
    except Exception as e:
        print(f"{'Adafactor':<30} {'FAIL':>10} -- {str(e)[:50]}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
