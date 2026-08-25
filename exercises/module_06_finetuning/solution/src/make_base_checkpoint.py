"""
Offline generator for data/base_model.pt.

Trains a tiny GPT on the bundled tiny Shakespeare corpus from Module 5,
seeds everything for reproducibility, and saves the checkpoint so Module 6
can focus purely on finetuning without re-pretraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPTConfig, TinyGPT  # noqa: E402


BLOCK_SIZE = 128
BATCH_SIZE = 32
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
MAX_STEPS = 2000
WARMUP_STEPS = 100
MAX_LR = 3e-3
MIN_LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
SEED = 1337

_THIS_DIR = Path(__file__).resolve().parent
DATA_FILE = _THIS_DIR.parent.parent.parent / "module_05_pretraining" / "data" / "tinyshakespeare.txt"
OUTPUT_FILE = _THIS_DIR.parent.parent / "data" / "base_model.pt"


def _find_data_file() -> Path:
    for parent in _THIS_DIR.parents:
        candidate = parent / "module_05_pretraining" / "data" / "tinyshakespeare.txt"
        if candidate.exists():
            return candidate
    if DATA_FILE.exists():
        return DATA_FILE
    raise FileNotFoundError("Could not locate tinyshakespeare.txt")


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


def lr_at_step(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    import math
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    text = _find_data_file().read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    data = encode(text, stoi)
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]

    cfg = GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT)
    model = TinyGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    batch_gen = torch.Generator().manual_seed(SEED)

    print(f"Training base model: {model.num_params():,} params, {len(text):,} chars")
    for step in range(MAX_STEPS + 1):
        lr = lr_at_step(step, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % 250 == 0 or step == MAX_STEPS:
            model.eval()
            with torch.no_grad():
                x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, batch_gen)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            model.train()
            print(f"  step {step:>4}: loss {loss.item():.4f}")

        if step == MAX_STEPS:
            break

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, batch_gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    torch.save({
        "config": {
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "dropout": cfg.dropout,
        },
        "state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
    }, OUTPUT_FILE)
    print(f"Saved base checkpoint to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
