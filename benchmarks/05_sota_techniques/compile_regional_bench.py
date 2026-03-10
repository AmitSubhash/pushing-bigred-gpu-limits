"""torch.compile Regional vs Whole-Model Benchmark on A100.

Compares eager, whole-model compile, regional (per-layer) compile,
and reduce-overhead mode for training and inference.
"""
import time
import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """Small transformer for compile benchmarking."""
    def __init__(self, vocab: int = 32000, d: int = 1024, heads: int = 16,
                 layers: int = 12, ff: int = 4096):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        return self.head(self.transformer(self.embed(x)))


def bench_train(model, optimizer, input_ids, labels, warmup=5, iters=20):
    """Measure training throughput."""
    for _ in range(warmup):
        out = model(input_ids)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = model(input_ids)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tok_s = iters * input_ids.numel() / elapsed
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return tok_s, mem_gb


def bench_infer(model, input_ids, warmup=10, iters=50):
    """Measure inference latency."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = model(input_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    ms = (elapsed / iters) * 1000
    tok_s = iters * input_ids.numel() / elapsed
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return tok_s, ms, mem_gb


def main():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("torch.compile Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    batch, seq = 8, 512

    modes = []

    # 1. Eager baseline
    print("Building eager model...")
    m = TransformerLM().cuda().to(torch.bfloat16)
    modes.append(("Eager", m))

    # 2. Whole-model compile (default)
    print("Compiling whole model (default)...")
    m2 = TransformerLM().cuda().to(torch.bfloat16)
    m2 = torch.compile(m2, mode="default")
    modes.append(("compile(default)", m2))

    # 3. Regional compile (per-layer)
    print("Compiling regional (per-layer)...")
    m3 = TransformerLM().cuda().to(torch.bfloat16)
    for i in range(len(m3.transformer.layers)):
        m3.transformer.layers[i] = torch.compile(m3.transformer.layers[i], mode="default")
    modes.append(("compile(regional)", m3))

    # 4. Whole-model reduce-overhead (for inference)
    print("Compiling reduce-overhead...")
    m4 = TransformerLM().cuda().to(torch.bfloat16)
    m4 = torch.compile(m4, mode="reduce-overhead")
    modes.append(("compile(reduce-OH)", m4))

    input_ids = torch.randint(0, 32000, (batch, seq), device="cuda")
    labels = torch.randint(0, 32000, (batch, seq), device="cuda")

    # Training benchmark
    print(f"\n{'='*60}")
    print(f"TRAINING: batch={batch}, seq={seq}, BF16")
    print(f"{'Mode':<25} {'tok/s':>12} {'Mem (GB)':>10}")
    print("-" * 50)

    for name, model in modes:
        try:
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            tok_s, mem = bench_train(model, opt, input_ids, labels)
            print(f"{name:<25} {tok_s:>12,.0f} {mem:>10.2f}")
            del opt
        except Exception as e:
            print(f"{name:<25} {'FAIL':>12} -- {str(e)[:40]}")
        torch.cuda.empty_cache()

    # Inference benchmark
    infer_input = torch.randint(0, 32000, (1, 128), device="cuda")
    print(f"\n{'='*60}")
    print(f"INFERENCE: batch=1, seq=128, BF16")
    print(f"{'Mode':<25} {'tok/s':>12} {'Latency (ms)':>14} {'Mem (GB)':>10}")
    print("-" * 65)

    for name, model in modes:
        try:
            tok_s, ms, mem = bench_infer(model, infer_input)
            print(f"{name:<25} {tok_s:>12,.0f} {ms:>14.2f} {mem:>10.2f}")
        except Exception as e:
            print(f"{name:<25} {'FAIL':>12} -- {str(e)[:40]}")
        torch.cuda.empty_cache()

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
