"""
Offline generator for data/instruct_model.pt (solution-only build script).

Module 7 starts from a small instruct model that can *partly* reverse strings: it
usually has the right answer as its top guess, but its sampling distribution is
broad, so sampled completions are correct only ~30% of the time. That gap is exactly
what GRPO closes, and it mirrors the module's thesis (RL sharpens the base model's
distribution rather than inventing new ability).

To produce that checkpoint we take the Module 5/6 base model, expand its vocabulary
by the four special tokens, and supervised-finetune it on the chat-template reverse
task over the `sft` word split (held out from the RL prompts). We stop while the task
is still being learned (mid-transition), which leaves the policy in the uncertain
regime RL needs.

Run with:
    uv run python module_07_rl/solution/src/make_instruct_checkpoint.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
# Pull the word splits from the exercise's data module so SFT words and RL prompts
# never overlap (src/ is added so `import data` resolves to the provided module).
sys.path.insert(0, str(_THIS_DIR.parent.parent / "src"))

from model import GPTConfig, TinyGPT, generate  # noqa: E402
from tokenizer import build_vocab, encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import word_splits  # noqa: E402


BATCH_SIZE = 64
MAX_SFT_STEPS = 300      # train past the competence peak, then keep the BEST checkpoint
CHECK_EVERY = 5
LR = 3e-4               # lr at which the tiny model groks the reverse algorithm
GRAD_CLIP = 1.0
PAD_LEN = 48
SAMPLE_TEMP = 1.0        # the temperature GRPO (and the eval) will sample at
SEED = 1
DATA_SEED = 0

OUTPUT_FILE = _THIS_DIR.parent.parent / "data" / "instruct_model.pt"


def _find(rel_parts: tuple[str, ...]) -> Path:
    for parent in _THIS_DIR.parents:
        candidate = parent.joinpath(*rel_parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("/".join(rel_parts))


def _load_base(new_vocab_size: int):
    ckpt_path = _find(("module_06_finetuning", "data", "base_model.pt"))
    ckpt = torch.load(ckpt_path, weights_only=False)
    raw_cfg = ckpt["config"]
    old_cfg = GPTConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    old_vocab = old_cfg.vocab_size
    cfg = GPTConfig(
        vocab_size=new_vocab_size,
        block_size=old_cfg.block_size,
        n_layer=old_cfg.n_layer,
        n_head=old_cfg.n_head,
        n_embd=old_cfg.n_embd,
        dropout=0.0,
    )
    model = TinyGPT(cfg)
    sd = ckpt["state_dict"]
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    with torch.no_grad():
        model.token_embed.weight[:old_vocab, :].copy_(sd["token_embed.weight"])
    return model, cfg, ckpt["stoi"]


def _fmt(p, r, sp, enc):
    return [sp["<|user|>"]] + enc(p) + [sp["<|end|>"]] + [sp["<|assistant|>"]] + enc(r) + [sp["<|end|>"]]


def _targets(ids, ps):
    return [-100] * (ps - 1) + ids[ps:] + [-100]


def _pad(s, n, p):
    return s[:n] if len(s) >= n else s + [p] * (n - len(s))


def main() -> None:
    torch.manual_seed(SEED)
    new_vocab = 65 + len(SPECIAL_TOKENS)

    model, cfg, base_stoi = _load_base(new_vocab)
    stoi, itos = build_vocab(base_stoi)
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    enc = lambda s: encode(s, stoi)  # noqa: E731

    splits = word_splits(DATA_SEED)
    prompts = [(f"reverse: {w}", w[::-1]) for w in splits["sft"]]
    probe_words = splits["eval"]

    def batch(g):
        idxs = torch.randperm(len(prompts), generator=g)[:BATCH_SIZE].tolist()
        xs, ys = [], []
        for i in idxs:
            p, r = prompts[i]
            ids = _fmt(p, r, special, enc)
            ps = 1 + len(enc(p)) + 1 + 1
            xs.append(_pad(ids, PAD_LEN, special["<|pad|>"]))
            ys.append(_pad(_targets(ids, ps), PAD_LEN, -100))
        return torch.tensor(xs), torch.tensor(ys)

    def per_sample_reward(k=8):
        """Mean fraction of sampled (temperature) completions that verify."""
        model.eval()
        total = 0.0
        for w in probe_words:
            ids = [special["<|user|>"]] + enc(f"reverse: {w}") + [special["<|end|>"]] + [special["<|assistant|>"]]
            gg = torch.Generator().manual_seed(7)
            hit = 0
            for _ in range(k):
                out = generate(model, torch.tensor([ids]), len(w) + 2, cfg.block_size,
                               temperature=SAMPLE_TEMP, generator=gg)
                resp = decode(out[0], itos).split("<|assistant|>", 1)[-1].split("<|end|>", 1)[0]
                if resp == w[::-1]:
                    hit += 1
            total += hit / k
        model.train()
        return total / len(probe_words)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)

    print(f"Finetuning instruct model: {model.num_params():,} params, {len(prompts)} reverse pairs")
    print("Keeping the checkpoint with the highest held-out per-sample reward")
    print("(competence peaks mid-training, then overfitting HURTS generalization)")
    model.train()
    best_reward = -1.0
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    for step in range(1, MAX_SFT_STEPS + 1):
        xb, yb = batch(g)
        loss = F.cross_entropy(model(xb).view(-1, new_vocab), yb.view(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if step % CHECK_EVERY == 0:
            r = per_sample_reward()
            flag = ""
            if r > best_reward:
                best_reward, best_step = r, step
                best_state = copy.deepcopy(model.state_dict())
                flag = "  <- new best"
            print(f"  step {step:>4}: loss {loss.item():.4f}  per-sample reward {r:.2f}{flag}")
    model.load_state_dict(best_state)
    print(f"  best checkpoint: step {best_step}, per-sample reward {best_reward:.2f}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "dropout": 0.0,
        },
        "state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
    }, OUTPUT_FILE)
    print(f"Saved instruct checkpoint to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
