"""N-gram Speculative Decoding Benchmark on A100.

Implements N-gram speculation for autoregressive decoding
and compares against standard greedy decoding.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallTransformerLM(nn.Module):
    """~216M param model for decoding benchmarks."""
    def __init__(self, vocab: int = 32000, d: int = 1024, heads: int = 16,
                 layers: int = 12, ff: int = 4096):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.pos_embed = nn.Embedding(2048, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, start_pos: int = 0):
        B, T = x.shape
        positions = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.transformer(h)
        return self.head(h)


def greedy_decode(model, prompt_ids: torch.Tensor, max_new: int = 128) -> tuple:
    """Standard autoregressive greedy decoding."""
    generated = prompt_ids.clone()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(generated)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return generated, elapsed


def ngram_speculative_decode(
    model, prompt_ids: torch.Tensor, max_new: int = 128,
    ngram_max: int = 4, num_spec: int = 5,
) -> tuple:
    """N-gram speculative decoding.

    Uses n-gram matching from the prompt/generated text as a draft.
    The model verifies all speculative tokens in a single forward pass.
    """
    generated = prompt_ids.clone()
    total_accepted = 0
    total_proposed = 0

    start = time.perf_counter()
    with torch.no_grad():
        tokens_generated = 0
        while tokens_generated < max_new:
            # Build n-gram lookup from generated text
            seq = generated[0].tolist()
            ngram_table = {}
            for n in range(ngram_max, 0, -1):
                for i in range(len(seq) - n):
                    key = tuple(seq[i:i + n])
                    # Store the tokens that followed this n-gram
                    if i + n < len(seq):
                        ngram_table[key] = seq[i + n:min(i + n + num_spec, len(seq))]

            # Try to find speculative tokens via n-gram match
            draft_tokens = []
            for n in range(ngram_max, 0, -1):
                if len(seq) >= n:
                    key = tuple(seq[-n:])
                    if key in ngram_table:
                        draft_tokens = ngram_table[key][:num_spec]
                        break

            if not draft_tokens:
                # No n-gram match, fall back to standard decoding
                logits = model(generated)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1
                continue

            # Create verification input: current sequence + draft tokens
            draft_tensor = torch.tensor([draft_tokens], device=generated.device)
            verify_input = torch.cat([generated, draft_tensor], dim=1)

            # Single forward pass to verify all draft tokens
            logits = model(verify_input)

            # Check which draft tokens match the model's predictions
            # Compare model's prediction at position i with draft token at position i+1
            verify_start = generated.size(1) - 1
            accepted = 0
            for i in range(len(draft_tokens)):
                model_token = logits[:, verify_start + i, :].argmax(dim=-1).item()
                if model_token == draft_tokens[i]:
                    accepted += 1
                else:
                    # Accept the model's prediction instead
                    next_token = torch.tensor([[model_token]], device=generated.device)
                    draft_accepted = torch.tensor([draft_tokens[:i]], device=generated.device) if i > 0 else None
                    if draft_accepted is not None and draft_accepted.numel() > 0:
                        generated = torch.cat([generated, draft_accepted, next_token], dim=1)
                    else:
                        generated = torch.cat([generated, next_token], dim=1)
                    tokens_generated += i + 1
                    break
            else:
                # All draft tokens accepted, add one more from model
                all_accepted = torch.tensor([draft_tokens], device=generated.device)
                bonus_token = logits[:, verify_start + len(draft_tokens), :].argmax(dim=-1).unsqueeze(0)
                generated = torch.cat([generated, all_accepted, bonus_token], dim=1)
                tokens_generated += len(draft_tokens) + 1
                accepted = len(draft_tokens)

            total_accepted += accepted
            total_proposed += len(draft_tokens)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    accept_rate = total_accepted / max(total_proposed, 1)
    return generated, elapsed, accept_rate


def main():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("N-gram Speculative Decoding Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    model = SmallTransformerLM().cuda().bfloat16().eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.0f}M params")

    # Create a prompt with some repetitive content (helps n-gram matching)
    prompt_len = 256
    prompt = torch.randint(0, 1000, (1, prompt_len), device="cuda")  # small vocab range for more repeats

    max_new_tokens = 128

    print(f"\nPrompt length: {prompt_len}, Generate: {max_new_tokens} tokens")
    print(f"{'Method':<35} {'Time (s)':>10} {'tok/s':>10} {'Speedup':>10} {'Accept Rate':>12}")
    print("-" * 82)

    # Warmup
    with torch.no_grad():
        _ = model(prompt)
    torch.cuda.synchronize()

    # Standard greedy
    gen_std, t_std = greedy_decode(model, prompt, max_new_tokens)
    tps_std = max_new_tokens / t_std
    print(f"{'Greedy (standard)':<35} {t_std:>10.3f} {tps_std:>10.1f} {'1.00x':>10} {'N/A':>12}")

    # N-gram speculative with different n values
    for ngram_max in [2, 3, 4]:
        for num_spec in [3, 5]:
            label = f"N-gram(n={ngram_max}, k={num_spec})"
            try:
                gen_spec, t_spec, accept_rate = ngram_speculative_decode(
                    model, prompt, max_new_tokens,
                    ngram_max=ngram_max, num_spec=num_spec,
                )
                tps_spec = max_new_tokens / t_spec
                speedup = t_std / t_spec
                print(f"{label:<35} {t_spec:>10.3f} {tps_spec:>10.1f} {speedup:>9.2f}x {accept_rate:>11.1%}")
            except Exception as e:
                print(f"{label:<35} {'FAIL':>10} -- {str(e)[:40]}")

    # Test with longer, more repetitive prompt (better for n-gram matching)
    print(f"\n{'='*60}")
    print("With repetitive prompt (better n-gram matches):")
    # Create a prompt with repeated patterns
    pattern = torch.randint(0, 500, (1, 32), device="cuda")
    prompt_rep = pattern.repeat(1, 8)  # 256 tokens, very repetitive

    gen_std2, t_std2 = greedy_decode(model, prompt_rep, max_new_tokens)
    tps_std2 = max_new_tokens / t_std2
    print(f"{'Greedy (standard)':<35} {t_std2:>10.3f} {tps_std2:>10.1f} {'1.00x':>10}")

    for ngram_max in [3, 4]:
        label = f"N-gram(n={ngram_max}, k=5, repetitive)"
        try:
            gen_spec, t_spec, accept_rate = ngram_speculative_decode(
                model, prompt_rep, max_new_tokens,
                ngram_max=ngram_max, num_spec=5,
            )
            tps_spec = max_new_tokens / t_spec
            speedup = t_std2 / t_spec
            print(f"{label:<35} {t_spec:>10.3f} {tps_spec:>10.1f} {speedup:>9.2f}x {accept_rate:>11.1%}")
        except Exception as e:
            print(f"{label:<35} {'FAIL':>10} -- {str(e)[:40]}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
