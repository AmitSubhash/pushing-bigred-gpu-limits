"""NF4 QLoRA vs Full BF16 Training Benchmark.

Compares memory usage and throughput of NF4 quantized base model
with LoRA adapters vs full BF16 fine-tuning.
"""
import time
import torch
import torch.nn as nn


class LargeTransformerLM(nn.Module):
    """~830M param model for QLoRA comparison."""
    def __init__(self, vocab: int = 32000, d: int = 2048, heads: int = 32,
                 layers: int = 16, ff: int = 8192):
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


def train_loop(model, optimizer, input_ids, labels, warmup=3, iters=10):
    """Training loop with BF16 autocast."""
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
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tok_s = iters * input_ids.numel() / elapsed
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return tok_s, mem_gb


def main():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("QLoRA vs Full BF16 Training Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    batch, seq = 4, 512
    input_ids = torch.randint(0, 32000, (batch, seq), device="cuda")
    labels = torch.randint(0, 32000, (batch, seq), device="cuda")

    print(f"{'Config':<35} {'tok/s':>10} {'Mem (GB)':>10} {'Trainable Params':>18}")
    print("-" * 78)

    # 1. Full BF16 training
    try:
        model = LargeTransformerLM().cuda().bfloat16()
        total_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tok_s, mem_gb = train_loop(model, opt, input_ids, labels)
        print(f"{'Full BF16':<35} {tok_s:>10,.0f} {mem_gb:>10.2f} {total_params:>18,}")
        del model, opt
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{'Full BF16':<35} {'FAIL':>10} -- {str(e)[:40]}")
        torch.cuda.empty_cache()

    # 2. INT8 base + LoRA (if peft available)
    try:
        import bitsandbytes as bnb
        from peft import LoraConfig, get_peft_model, TaskType

        # INT8 quantized base model + LoRA
        model = LargeTransformerLM().cuda()

        # Quantize linear layers to INT8
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                quantized = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                )
                quantized.weight = bnb.nn.Int8Params(
                    module.weight.data, requires_grad=False, has_fp16_weights=False,
                )
                if module.bias is not None:
                    quantized.bias = module.bias
                setattr(parent, child_name, quantized)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["linear1", "linear2"],  # TransformerEncoderLayer's FFN
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        tok_s, mem_gb = train_loop(model, opt, input_ids, labels)
        print(f"{'INT8 + LoRA (r=16)':<35} {tok_s:>10,.0f} {mem_gb:>10.2f} {trainable:>18,}")
        del model, opt
        torch.cuda.empty_cache()
    except ImportError:
        print(f"{'INT8 + LoRA':<35} {'SKIP':>10} -- peft or bitsandbytes not installed")
    except Exception as e:
        print(f"{'INT8 + LoRA':<35} {'FAIL':>10} -- {str(e)[:50]}")
        torch.cuda.empty_cache()

    # 3. NF4 QLoRA (if available)
    try:
        import bitsandbytes as bnb
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # NF4 quantized model
        model = LargeTransformerLM()

        # Manual NF4 quantization of linear layers
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and module.in_features >= 1024:
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                nf4_linear = bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
                setattr(parent, child_name, nf4_linear)

        model = model.cuda()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["linear1", "linear2"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        tok_s, mem_gb = train_loop(model, opt, input_ids, labels)
        print(f"{'NF4 QLoRA (r=16)':<35} {tok_s:>10,.0f} {mem_gb:>10.2f} {trainable:>18,}")
        del model, opt
        torch.cuda.empty_cache()
    except ImportError:
        print(f"{'NF4 QLoRA':<35} {'SKIP':>10} -- peft or bitsandbytes not installed")
    except Exception as e:
        print(f"{'NF4 QLoRA':<35} {'FAIL':>10} -- {str(e)[:50]}")
        torch.cuda.empty_cache()

    # 4. Full BF16 + 8-bit optimizer
    try:
        import bitsandbytes as bnb
        model = LargeTransformerLM().cuda().bfloat16()
        total_params = sum(p.numel() for p in model.parameters())
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
        tok_s, mem_gb = train_loop(model, opt, input_ids, labels)
        print(f"{'Full BF16 + AdamW8bit':<35} {tok_s:>10,.0f} {mem_gb:>10.2f} {total_params:>18,}")
        del model, opt
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{'Full BF16 + AdamW8bit':<35} {'FAIL':>10} -- {str(e)[:50]}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
