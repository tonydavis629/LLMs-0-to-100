"""
Module 7 Exercise runner: GRPO on the instruct model with a verifiable reward

Run with:
    uv run python module_07_rl/src/main.py

Loads the bundled Module 6 instruct checkpoint as both the policy and the frozen
reference, then improves it with GRPO on a task it can verify itself: reversing a
string. The runner makes the RL loop visible: sample a group of completions, score
each with a Python verifier, turn rewards into group-relative advantages, and take
one policy-gradient step that pushes up the winners and down the losers. It reports
held-out accuracy before and after, the mean reward over training, and a sample
completion before and after, and it saves a reward-curve image.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one step at a time and re-run immediately.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided model / tokenizer / data / plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (  # noqa: E402  (import after sys.path edits)
    sample_group,
    verifiable_reward,
    score_group,
    group_relative_advantages,
    completion_mask,
    gather_token_log_probs,
    pg_loss,
    kl_penalty,
    grpo_step,
    mean_reward,
)
from model import load_instruct_model, generate  # noqa: E402
from tokenizer import encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import load_prompts  # noqa: E402
from visualization import save_reward_curve  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (small enough to run on a laptop CPU in a couple of minutes)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128         # context length
GROUP_SIZE = 8           # completions sampled per prompt (G)
PROMPTS_PER_STEP = 4     # prompts in each optimizer step (a batch of groups)
MAX_NEW_TOKENS = 8       # tokens to generate per completion (answers are short)
MAX_STEPS = 400          # GRPO steps
EVAL_INTERVAL = 20       # record reward / report every this many steps
LR = 1e-4               # policy learning rate
GRAD_CLIP = 1.0
BETA = 0.01              # KL-to-reference penalty weight
TEMPERATURE = 1.0        # sampling temperature for the group (exploration)
EVAL_SAMPLES = 8         # samples per held-out prompt when measuring sampled accuracy
SEED = 1337

_THIS_DIR = Path(__file__).resolve().parent
END_TOKEN = "<|end|>"


def _find_data_file(name: str) -> Path:
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def _is_implemented(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True


def _probe_steps(policy, prompt_ids, special) -> dict[str, bool]:
    """Detect which exercise.py steps are implemented, using throwaway inputs."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    logits = torch.randn(5, len(special) + 65)
    target_ids = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.tensor([False, True, True, True, True])
    lp = torch.randn(5)
    scratch = copy.deepcopy(policy)
    opt = torch.optim.AdamW(scratch.parameters(), lr=1e-4)
    dummy_loss = (scratch(prompt_ids).sum()) * 0.0
    gen = torch.Generator().manual_seed(0)
    return {
        "sample_group": _is_implemented(
            sample_group, policy, prompt_ids, 2, MAX_NEW_TOKENS, BLOCK_SIZE, TEMPERATURE, generate, gen),
        "verifiable_reward": _is_implemented(verifiable_reward, "tac", "tac"),
        "score_group": _is_implemented(score_group, ["tac", "cat"], "tac"),
        "group_relative_advantages": _is_implemented(group_relative_advantages, rewards),
        "completion_mask": _is_implemented(completion_mask, 3, 6),
        "gather_token_log_probs": _is_implemented(gather_token_log_probs, logits, target_ids),
        "pg_loss": _is_implemented(pg_loss, lp, 1.0, mask),
        "kl_penalty": _is_implemented(kl_penalty, lp, lp, mask),
        "grpo_step": _is_implemented(grpo_step, opt, dummy_loss, scratch, GRAD_CLIP),
        "mean_reward": _is_implemented(mean_reward, rewards),
    }


def _heading(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _gen_prompt_ids(prompt: str, special, enc) -> torch.Tensor:
    """Build the generation prefix [user] prompt [end] [assistant] as a (1, P) tensor."""
    ids = (
        [special["<|user|>"]]
        + enc(prompt)
        + [special["<|end|>"]]
        + [special["<|assistant|>"]]
    )
    return torch.tensor([ids], dtype=torch.long)


def _response_text(seq: torch.Tensor, prompt_len: int, itos) -> str:
    """Decode the generated portion, cut at the first <|end|>."""
    gen_ids = seq[prompt_len:]
    text = decode(gen_ids, itos)
    return text.split(END_TOKEN, 1)[0]


def _truncate_at_end(seq: torch.Tensor, prompt_len: int, end_id: int) -> torch.Tensor:
    """Keep prompt + generated tokens up to and including the first <|end|>."""
    gen_ids = seq[prompt_len:].tolist()
    if end_id in gen_ids:
        cut = gen_ids.index(end_id) + 1
        return seq[: prompt_len + cut]
    return seq


def _eval_greedy_accuracy(policy, prompts, special, enc, itos) -> float:
    """Greedy (argmax) held-out accuracy: fraction whose top-guess answer verifies."""
    rewards = []
    for item in prompts:
        prompt_ids = _gen_prompt_ids(item["prompt"], special, enc)
        out = generate(policy, prompt_ids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
        resp = _response_text(out[0], prompt_ids.shape[1], itos)
        rewards.append(verifiable_reward(resp, item["answer"]))
    return mean_reward(torch.tensor(rewards))


def _eval_sampled_accuracy(policy, prompts, special, enc, itos) -> float:
    """Sampled held-out accuracy: fraction of sampled completions that verify.

    This is the metric GRPO actually optimizes (the policy samples at TEMPERATURE),
    and the one that moves most: it measures how *reliable* the model is, not just
    whether its single top guess happens to be right.
    """
    rewards = []
    for item in prompts:
        prompt_ids = _gen_prompt_ids(item["prompt"], special, enc)
        gen = torch.Generator().manual_seed(SEED)
        for _ in range(EVAL_SAMPLES):
            out = generate(policy, prompt_ids, MAX_NEW_TOKENS, BLOCK_SIZE,
                           temperature=TEMPERATURE, generator=gen)
            resp = _response_text(out[0], prompt_ids.shape[1], itos)
            rewards.append(verifiable_reward(resp, item["answer"]))
    return mean_reward(torch.tensor(rewards))


def _completion_losses(policy, reference, seqs, advantages, prompt_len, end_id) -> list[torch.Tensor]:
    """Per-completion policy-gradient loss + KL penalty for one prompt's group."""
    losses = []
    for seq, adv in zip(seqs, advantages):
        seq = _truncate_at_end(seq, prompt_len, end_id)
        if seq.shape[0] - prompt_len < 1:
            continue  # nothing was generated before <|end|>
        inp = seq[:-1].unsqueeze(0)
        targets = seq[1:]
        mask = completion_mask(prompt_len, seq.shape[0])
        policy_lp = gather_token_log_probs(policy(inp)[0], targets)
        with torch.no_grad():
            ref_lp = gather_token_log_probs(reference(inp)[0], targets)
        losses.append(pg_loss(policy_lp, adv.item(), mask) + BETA * kl_penalty(policy_lp, ref_lp, mask))
    return losses


def _show_sample(policy, prompt: str, answer: str, special, enc, itos, label: str) -> None:
    prompt_ids = _gen_prompt_ids(prompt, special, enc)
    out = generate(policy, prompt_ids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
    resp = _response_text(out[0], prompt_ids.shape[1], itos)
    ok = "correct" if resp == answer else "wrong"
    print(f"  {label}: {prompt!r} -> {resp!r}  (want {answer!r}: {ok})")


def main() -> None:
    torch.manual_seed(SEED)

    # data/instruct_model.pt ships with the repo. It is the Module 6 story finished for
    # us: the Module 5 base model after supervised finetuning, built by
    # solution/src/make_instruct_checkpoint.py.
    ckpt = _find_data_file("instruct_model.pt")
    policy, stoi, itos = load_instruct_model(ckpt)
    reference, _, _ = load_instruct_model(ckpt)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    enc = lambda s: encode(s, stoi)  # noqa: E731
    end_id = special[END_TOKEN]

    prompts = load_prompts(_find_data_file("verify_prompts.jsonl"))
    train_prompts = [p for p in prompts if p["split"] == "train"]
    eval_prompts = [p for p in prompts if p["split"] == "eval"]

    example_prompt = eval_prompts[0]

    steps = _probe_steps(policy, _gen_prompt_ids(example_prompt["prompt"], special, enc), special)

    _heading("MODULE 7: GRPO with a verifiable reward")
    print(f"TinyGPT: {policy.cfg.n_layer} layers, {policy.cfg.n_head} heads, width {policy.cfg.n_embd}")
    print(f"Policy parameters: {policy.num_params():,}   (reference is a frozen copy)")
    print(f"Task: reverse a string, verified by a Python function (no reward model)")
    print(f"Train prompts: {len(train_prompts)}   Held-out prompts: {len(eval_prompts)}")
    print(f"Group size G={GROUP_SIZE}, temperature={TEMPERATURE}, beta(KL)={BETA}, lr={LR}")
    print()

    # ------------------------------------------------------------------
    # BEFORE: held-out accuracy and one sample completion.
    # ------------------------------------------------------------------
    eval_ready = all(steps[s] for s in ("verifiable_reward", "mean_reward"))

    # For the before/after sample, prefer a prompt the model currently gets WRONG
    # (greedy), so the flip is visible. Fall back to the first eval prompt.
    if eval_ready:
        for it in eval_prompts:
            pids = _gen_prompt_ids(it["prompt"], special, enc)
            out = generate(policy, pids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
            if verifiable_reward(_response_text(out[0], pids.shape[1], itos), it["answer"]) == 0.0:
                example_prompt = it
                break

    _heading("BEFORE GRPO")
    acc_before = None
    if eval_ready:
        acc_before = _eval_sampled_accuracy(policy, eval_prompts, special, enc, itos)
        greedy_before = _eval_greedy_accuracy(policy, eval_prompts, special, enc, itos)
        print(f"  Held-out accuracy, sampled (temp {TEMPERATURE}): {acc_before:.1%}   <- what GRPO optimizes")
        print(f"  Held-out accuracy, greedy (argmax):     {greedy_before:.1%}")
        _show_sample(policy, example_prompt["prompt"], example_prompt["answer"],
                     special, enc, itos, "sample")
    else:
        print("  [skipped: implement verifiable_reward() and mean_reward()]")
    print()

    # ------------------------------------------------------------------
    # TRAINING: the full GRPO loop needs every step.
    # ------------------------------------------------------------------
    core = ["sample_group", "verifiable_reward", "score_group", "group_relative_advantages",
            "completion_mask", "gather_token_log_probs", "pg_loss", "kl_penalty",
            "grpo_step", "mean_reward"]
    missing = [s for s in core if not steps[s]]

    _heading("GRPO TRAINING")
    curve_steps: list[int] = []
    curve_rewards: list[float] = []
    if missing:
        print(f"  [skipped: implement {', '.join(missing)} to train]")
        print()
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
        gen = torch.Generator().manual_seed(SEED)
        rng = torch.Generator().manual_seed(SEED)
        print(f"{'step':>6}  {'mean reward':>12}")
        interval_rewards: list[float] = []
        for step in range(MAX_STEPS + 1):
            if (step % EVAL_INTERVAL == 0 or step == MAX_STEPS) and interval_rewards:
                curve_steps.append(step)
                curve_rewards.append(sum(interval_rewards) / len(interval_rewards))
                print(f"{step:>6}  {curve_rewards[-1]:>12.3f}")
                interval_rewards = []
            if step == MAX_STEPS:
                break

            # One optimizer step over a batch of prompts, each with its own group.
            step_losses: list[torch.Tensor] = []
            for _ in range(PROMPTS_PER_STEP):
                item = train_prompts[torch.randint(len(train_prompts), (1,), generator=rng).item()]
                prompt_ids = _gen_prompt_ids(item["prompt"], special, enc)
                prompt_len = prompt_ids.shape[1]

                policy.eval()
                seqs = sample_group(policy, prompt_ids, GROUP_SIZE, MAX_NEW_TOKENS,
                                    BLOCK_SIZE, TEMPERATURE, generate, gen)
                responses = [_response_text(s, prompt_len, itos) for s in seqs]
                rewards = score_group(responses, item["answer"])
                advantages = group_relative_advantages(rewards)
                interval_rewards.append(mean_reward(rewards))

                policy.train()
                step_losses.extend(
                    _completion_losses(policy, reference, seqs, advantages, prompt_len, end_id))

            loss = torch.stack(step_losses).mean() if step_losses else torch.zeros((), requires_grad=True)
            grpo_step(optimizer, loss, policy, GRAD_CLIP)
        print()

    # ------------------------------------------------------------------
    # AFTER: held-out accuracy, the same sample, and the reward curve image.
    # ------------------------------------------------------------------
    _heading("AFTER GRPO")
    if not missing:
        acc_after = _eval_sampled_accuracy(policy, eval_prompts, special, enc, itos)
        greedy_after = _eval_greedy_accuracy(policy, eval_prompts, special, enc, itos)
        print(f"  Held-out accuracy, sampled (temp {TEMPERATURE}): {acc_after:.1%}"
              + (f"   (was {acc_before:.1%})" if acc_before is not None else ""))
        print(f"  Held-out accuracy, greedy (argmax):     {greedy_after:.1%}")
        _show_sample(policy, example_prompt["prompt"], example_prompt["answer"],
                     special, enc, itos, "sample")
        out_img = _THIS_DIR.parent / "output" / "reward_curve.png"
        save_reward_curve(curve_steps, curve_rewards, out_img, acc_before, acc_after)
        print(f"  Reward curve saved to {out_img}")
    else:
        print("  [skipped: complete the steps above and re-run]")
    print()

    _heading("Done")
    print("Run after each step; unfinished steps are skipped automatically.")


if __name__ == "__main__":
    main()
