"""
Offline generator for the two checkpoints this module evaluates (solution-only).

Module 9 does not train anything; it *measures* two models that the previous two
modules produced. This script rebuilds them from scratch so the comparison in the
exercise is reproducible:

  data/instruct_model.pt  the Module 6 story: take the Module 5 base model, expand
                          the vocabulary with the four chat special tokens, and
                          supervised-finetune it on four toy tasks (uppercase,
                          repeat, qa, reverse).

  data/rl_model.pt        the Module 7 story: take that instruct model and run GRPO
                          on the reverse task only, with a verifiable reward. It
                          gets much better at reverse. What it does to the other
                          three tasks is the question Module 9 exists to answer.

Run with:
    uv run python module_09_evaluation/solution/src/make_checkpoints.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_MODULE_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_MODULE_ROOT / "src"))

from model import GPTConfig, TinyGPT, generate  # noqa: E402
from tokenizer import build_vocab, encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import sft_pairs, rl_prompts, reverse_splits  # noqa: E402


# --- SFT hyperparameters --------------------------------------------------
SFT_BATCH = 64
SFT_STEPS = 1200
SFT_LR = 3e-4
PAD_LEN = 48

# --- GRPO hyperparameters (the Module 7 loop, condensed) ------------------
GROUP_SIZE = 8
PROMPTS_PER_STEP = 4
MAX_NEW_TOKENS = 8
GRPO_STEPS = 150
GRPO_LR = 1e-4
BETA = 0.08              # KL-to-reference penalty weight. Note what it can and
                         # cannot do: the KL term is summed over the *reverse*
                         # completions, so it keeps the policy near the reference
                         # on reverse prompts and does nothing at all to protect
                         # uppercase, repeat, or qa. Raising it does not shrink
                         # the alignment tax, which is exactly why the tax needs
                         # its own evaluation instead of a training-side fix.
TEMPERATURE = 1.0
GRAD_CLIP = 1.0

SEED = 1337
DATA_DIR = _MODULE_ROOT / "data"


def _find_base_checkpoint() -> Path:
    for parent in _THIS_DIR.parents:
        candidate = parent / "module_06_finetuning" / "data" / "base_model.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("module_06_finetuning/data/base_model.pt")


def _load_base(new_vocab_size: int):
    """Load the Module 5/6 base checkpoint and widen its embedding table."""
    ckpt = torch.load(_find_base_checkpoint(), weights_only=False)
    raw_cfg = ckpt["config"]
    old_cfg = GPTConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
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
    model.load_state_dict(
        {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape},
        strict=False,
    )
    with torch.no_grad():
        model.token_embed.weight[: old_cfg.vocab_size, :].copy_(sd["token_embed.weight"])
    return model, cfg, ckpt["stoi"]


def _save(model, cfg, stoi, itos, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "note": note,
    }, path)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Stage 1: multi-task supervised finetuning (the Module 6 checkpoint)
# ---------------------------------------------------------------------------


def _chat_ids(prompt: str, response: str, special, enc) -> list[int]:
    return (
        [special["<|user|>"]] + enc(prompt) + [special["<|end|>"]]
        + [special["<|assistant|>"]] + enc(response) + [special["<|end|>"]]
    )


def _prompt_span(prompt: str, enc) -> int:
    return 1 + len(enc(prompt)) + 1 + 1


def _pad(seq: list[int], n: int, fill: int) -> list[int]:
    return seq[:n] if len(seq) >= n else seq + [fill] * (n - len(seq))


def sft(model, cfg, special, enc, itos) -> None:
    """Finetune the base model on the four toy instruction tasks."""
    pairs = sft_pairs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR)
    g = torch.Generator().manual_seed(SEED)

    def batch():
        idxs = torch.randperm(len(pairs), generator=g)[:SFT_BATCH].tolist()
        xs, ys = [], []
        for i in idxs:
            ids = _chat_ids(pairs[i]["prompt"], pairs[i]["response"], special, enc)
            span = _prompt_span(pairs[i]["prompt"], enc)
            targets = [-100] * (span - 1) + ids[span:] + [-100]
            xs.append(_pad(ids, PAD_LEN, special["<|pad|>"]))
            ys.append(_pad(targets, PAD_LEN, -100))
        return torch.tensor(xs), torch.tensor(ys)

    print(f"SFT: {len(pairs)} pairs, {SFT_STEPS} steps")
    model.train()
    for step in range(1, SFT_STEPS + 1):
        xb, yb = batch()
        loss = F.cross_entropy(
            model(xb).view(-1, cfg.vocab_size), yb.view(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if step % 100 == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")
    model.eval()


# ---------------------------------------------------------------------------
# Stage 2: GRPO on reverse only (the Module 7 checkpoint)
# ---------------------------------------------------------------------------


def _gen_prefix(prompt: str, special, enc) -> torch.Tensor:
    ids = ([special["<|user|>"]] + enc(prompt) + [special["<|end|>"]]
           + [special["<|assistant|>"]])
    return torch.tensor([ids], dtype=torch.long)


def _response(seq: torch.Tensor, prompt_len: int, itos) -> str:
    return decode(seq[prompt_len:], itos).split("<|end|>", 1)[0]


def grpo(policy, reference, cfg, special, enc, itos) -> None:
    """Run GRPO on the reverse task with a verifiable (exact-match) reward."""
    prompts = rl_prompts()
    end_id = special["<|end|>"]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=GRPO_LR)
    gen = torch.Generator().manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    print(f"GRPO: {len(prompts)} reverse prompts, {GRPO_STEPS} steps, G={GROUP_SIZE}")
    recent: list[float] = []
    for step in range(1, GRPO_STEPS + 1):
        losses = []
        for _ in range(PROMPTS_PER_STEP):
            item = prompts[torch.randint(len(prompts), (1,), generator=rng).item()]
            prefix = _gen_prefix(item["prompt"], special, enc)
            prompt_len = prefix.shape[1]

            policy.eval()
            seqs = [generate(policy, prefix, MAX_NEW_TOKENS, cfg.block_size,
                             temperature=TEMPERATURE, generator=gen)[0]
                    for _ in range(GROUP_SIZE)]
            rewards = torch.tensor(
                [1.0 if _response(s, prompt_len, itos) == item["answer"] else 0.0 for s in seqs])
            recent.append(rewards.mean().item())
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            policy.train()
            for seq, adv in zip(seqs, advantages):
                gen_ids = seq[prompt_len:].tolist()
                if end_id in gen_ids:
                    seq = seq[: prompt_len + gen_ids.index(end_id) + 1]
                if seq.shape[0] - prompt_len < 1:
                    continue
                inp, targets = seq[:-1].unsqueeze(0), seq[1:]
                mask = torch.arange(seq.shape[0] - 1) >= prompt_len - 1
                lp = F.log_softmax(policy(inp)[0], dim=-1).gather(
                    -1, targets.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    ref_lp = F.log_softmax(reference(inp)[0], dim=-1).gather(
                        -1, targets.unsqueeze(-1)).squeeze(-1)
                losses.append(-adv.item() * (lp * mask).sum()
                              + BETA * ((lp - ref_lp) * mask).sum())

        loss = torch.stack(losses).mean() if losses else torch.zeros((), requires_grad=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % 25 == 0:
            print(f"  step {step:>4}  mean reward {sum(recent) / len(recent):.3f}")
            recent = []
    policy.eval()


# ---------------------------------------------------------------------------


def _report(model, cfg, special, enc, itos, label: str) -> None:
    """Quick greedy spot check on one held-out case per task."""
    probes = [
        ("uppercase: metric", "METRIC"),
        ("repeat: fact", "fact"),
        ("color of sky?", "it is blue"),
        (f"reverse: {reverse_splits()['eval'][0]}", reverse_splits()["eval"][0][::-1]),
    ]
    print(f"  {label} greedy spot check:")
    for prompt, want in probes:
        prefix = _gen_prefix(prompt, special, enc)
        out = generate(model, prefix, 12, cfg.block_size, greedy=True)
        got = _response(out[0], prefix.shape[1], itos)
        flag = "ok " if got == want else "MISS"
        print(f"    [{flag}] {prompt!r} -> {got!r}  (want {want!r})")


def main() -> None:
    torch.manual_seed(SEED)
    new_vocab = 65 + len(SPECIAL_TOKENS)

    model, cfg, base_stoi = _load_base(new_vocab)
    stoi, itos = build_vocab(base_stoi)
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    enc = lambda s: encode(s, stoi)  # noqa: E731

    # `--rl-only` reuses the already-saved instruct checkpoint (SFT is deterministic)
    # so the GRPO hyperparameters can be re-tuned without re-running the finetune.
    reuse = "--rl-only" in sys.argv and (DATA_DIR / "instruct_model.pt").exists()

    print("=" * 60)
    print("STAGE 1: supervised finetuning (the Module 6 instruct model)")
    print("=" * 60)
    if reuse:
        model.load_state_dict(
            torch.load(DATA_DIR / "instruct_model.pt", weights_only=False)["state_dict"])
        model.eval()
        print("  reusing the saved instruct checkpoint (--rl-only)")
    else:
        sft(model, cfg, special, enc, itos)
        _report(model, cfg, special, enc, itos, "instruct")
        _save(model, cfg, stoi, itos, DATA_DIR / "instruct_model.pt",
              "Module 6 instruct model: multi-task SFT on uppercase/repeat/qa/reverse")

    print()
    print("=" * 60)
    print("STAGE 2: GRPO on reverse only (the Module 7 RL model)")
    print("=" * 60)
    policy = copy.deepcopy(model)
    reference = copy.deepcopy(model)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    grpo(policy, reference, cfg, special, enc, itos)
    _report(policy, cfg, special, enc, itos, "rl")
    _save(policy, cfg, stoi, itos, DATA_DIR / "rl_model.pt",
          "Module 7 RL model: GRPO on the reverse task with a verifiable reward")


if __name__ == "__main__":
    main()
