"""
Module 7 Exercise runner: GRPO on the instruct model with a verifiable reward

Run with:
    uv run python module_07_rl/src/main.py

Each step below runs one piece of GRPO that you write in exercise.py, then
tests it. Read top to bottom, the steps follow one round of GRPO:

    1. sample a group of G answers to one prompt
    2. check one answer against the known solution (reward 1 or 0)
    3. score the whole group
    4. turn rewards into advantages: how much better than the group average?
    5. mark which tokens the model generated (only those get trained)
    6. read off the model's log-probability for each of those tokens
    7. the policy-gradient loss: push up good answers, push down bad ones
    8. the KL penalty: stay close to the starting model
    9. one optimizer step
    10. the mean reward, for the training curve

After the ten steps comes the payoff: GRPO training with all ten pieces
together. The bundled Module 6 instruct model learns to reverse strings, a
task a Python function can verify, and the runner reports held-out accuracy
before and after. Training takes a few minutes on a laptop CPU.

Add --step N to run one step (1-10, or "train" for the training run).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
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
from src.chat import END_TOKEN, data_file, prompt_ids, response_text, truncate_at_end
from src.data import load_prompts
from src.model import generate, load_instruct_model
from src.prerequisites import STEP_NAMES, require
from src.reporting import format_numbers, progress, run_step
from src.visualization import save_reward_curve

# One test file per step lives in tests/
from tests.test_step1_sample_group import check_sample_group
from tests.test_step2_verifiable_reward import check_verifiable_reward
from tests.test_step3_score_group import check_score_group
from tests.test_step4_advantages import check_group_relative_advantages
from tests.test_step5_completion_mask import check_completion_mask
from tests.test_step6_log_probs import check_gather_token_log_probs
from tests.test_step7_pg_loss import check_pg_loss
from tests.test_step8_kl_penalty import check_kl_penalty
from tests.test_step9_grpo_step import check_grpo_step
from tests.test_step10_mean_reward import check_mean_reward
from tests.test_training import check_training

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Hyperparameters (small enough to run on a laptop CPU in a few minutes)
BLOCK_SIZE = 128         # context length
GROUP_SIZE = 8           # completions sampled per prompt (G)
PROMPTS_PER_STEP = 4     # prompts in each optimizer step (a batch of groups)
MAX_NEW_TOKENS = 8       # tokens to generate per completion (answers are short)
MAX_STEPS = 400          # GRPO steps
EVAL_INTERVAL = 20       # record reward / report every this many steps
LR = 1e-4                # policy learning rate
GRAD_CLIP = 1.0          # largest allowed gradient norm
BETA = 0.01              # KL-to-reference penalty weight
TEMPERATURE = 1.0        # sampling temperature for the group (exploration)
EVAL_SAMPLES = 8         # samples per held-out prompt when measuring sampled accuracy
SEED = 1337

# ---------------------------------------------------------------------------
# The models, the prompts, and asking the model a question
# ---------------------------------------------------------------------------


def greedy_answer(policy, prompt: str, setup: dict) -> str:
    """The model's single most likely answer (argmax at every token)."""
    ids = prompt_ids(prompt, setup["stoi"])
    out = generate(policy, ids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
    return response_text(out[0], ids.shape[1], setup["itos"])


@cache  # load once, then every step reuses the same models and prompts
def load_setup() -> dict:
    """Load the policy, the frozen reference, the tokenizer maps, and the prompts."""
    # data/instruct_model.pt ships with the repo. It is the Module 6 story finished for
    # us: the Module 5 base model after supervised finetuning, built by
    # src/make_instruct_checkpoint.py.
    ckpt = data_file("instruct_model.pt")
    policy, stoi, itos = load_instruct_model(ckpt)     # the model GRPO trains
    reference, _, _ = load_instruct_model(ckpt)        # an identical copy that never trains
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    prompts = load_prompts(data_file("verify_prompts.jsonl"))
    setup = {
        "policy": policy,
        "reference": reference,
        "stoi": stoi,
        "itos": itos,
        "end_id": stoi[END_TOKEN],
        "train": [p for p in prompts if p["split"] == "train"],  # GRPO learns from these
        "eval": [p for p in prompts if p["split"] == "eval"],    # held out, only measured
    }

    # The running example: the first held-out prompt the model currently gets WRONG
    # (greedy), so the before/after flip is visible. Fall back to the first one.
    setup["example"] = setup["eval"][0]
    for item in setup["eval"]:
        if greedy_answer(policy, item["prompt"], setup) != item["answer"]:
            setup["example"] = item
            break
    return setup


# ---------------------------------------------------------------------------
# Two running examples: a sampled group for the first held-out prompt (Steps
# 1, 3, 4, 10) and the greedy answer to the example prompt (Steps 5, 6)
# ---------------------------------------------------------------------------


@cache  # draw the group once, then reuse it in later steps
def example_group() -> list[str]:
    """The decoded group of G completions for the first held-out prompt."""
    setup = load_setup()
    ids = prompt_ids(setup["eval"][0]["prompt"], setup["stoi"])
    gen = torch.Generator().manual_seed(SEED)  # same seed, same group, every run
    seqs = sample_group(setup["policy"], ids, GROUP_SIZE, MAX_NEW_TOKENS,
                        BLOCK_SIZE, TEMPERATURE, generate, gen)
    return [response_text(s, ids.shape[1], setup["itos"]) for s in seqs]


@cache
def example_rewards() -> torch.Tensor:
    """The verifier's reward for each completion in the example group, as floats."""
    rewards = score_group(example_group(), load_setup()["eval"][0]["answer"])
    return torch.as_tensor(rewards, dtype=torch.float32)


@cache
def example_sequence() -> tuple[torch.Tensor, int]:
    """prompt + the model's greedy answer for the example, cut after <|end|>.

    Returns the sequence and the number of prompt tokens at its start.
    """
    setup = load_setup()
    ids = prompt_ids(setup["example"]["prompt"], setup["stoi"])
    out = generate(setup["policy"], ids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
    return truncate_at_end(out[0], ids.shape[1], setup["end_id"]), ids.shape[1]


# ---------------------------------------------------------------------------
# Step 1: sample a group of G completions
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        item = load_setup()["eval"][0]
        print(f"Held-out prompt {item['prompt']!r}; the verifier wants {item['answer']!r}")
        print(f"A group of G={GROUP_SIZE} completions sampled at temperature {TEMPERATURE}:")
        print("  " + "  ".join(repr(text) for text in example_group()))

    return run_step("Step 1: sample_group()", show, lambda: check_sample_group(sample_group))


# ---------------------------------------------------------------------------
# Step 2: the verifiable reward (1.0 for a correct answer, else 0.0)
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        setup = load_setup()
        print("The starting model's greedy answers on 4 held-out prompts:")
        for item in setup["eval"][:4]:
            answer = greedy_answer(setup["policy"], item["prompt"], setup)
            reward = verifiable_reward(answer, item["answer"])
            print(f"  {item['prompt']!r} -> {answer!r}  want {item['answer']!r}  reward {reward}")

    return run_step("Step 2: verifiable_reward()", show, lambda: check_verifiable_reward(verifiable_reward))


# ---------------------------------------------------------------------------
# Step 3: score the whole group
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        require(3, needs=[1, 2], why="to score a real group")
        rewards = score_group(example_group(), load_setup()["eval"][0]["answer"])
        print(f"Rewards for the Step 1 group: {format_numbers(rewards, 'g')}")

    return run_step("Step 3: score_group()", show, lambda: check_score_group(score_group))


# ---------------------------------------------------------------------------
# Step 4: rewards -> group-relative advantages
# ---------------------------------------------------------------------------


def step_4() -> str:
    def show():
        require(4, needs=[1, 2, 3], why="to score the group it standardizes")
        rewards = example_rewards()
        advantages = group_relative_advantages(rewards)
        print(f"Rewards for the Step 1 group: {format_numbers(rewards, '5.2f')}")
        print(f"Advantages:                   {format_numbers(advantages, '+5.2f')}")

    return run_step("Step 4: group_relative_advantages()", show,
                    lambda: check_group_relative_advantages(group_relative_advantages))


# ---------------------------------------------------------------------------
# Step 5: which positions predict a generated token?
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        setup = load_setup()
        seq, prompt_len = example_sequence()
        mask = completion_mask(prompt_len, seq.shape[0])
        answer = response_text(seq, prompt_len, setup["itos"])
        print(f"Greedy answer to {setup['example']['prompt']!r}: {answer!r} + <|end|>, "
              f"so {prompt_len} prompt tokens + {seq.shape[0] - prompt_len} generated")
        print(f"Mask over the {len(mask)} next-token positions: {''.join('1' if m else '0' for m in mask)}")

    return run_step("Step 5: completion_mask()", show, lambda: check_completion_mask(completion_mask))


# ---------------------------------------------------------------------------
# Step 6: how confident is the policy in each token of its answer?
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        setup = load_setup()
        seq, prompt_len = example_sequence()
        n_generated = seq.shape[0] - prompt_len
        with torch.no_grad():
            # Position t predicts token t + 1, so feed seq[:-1] and score against seq[1:]
            log_probs = gather_token_log_probs(setup["policy"](seq[:-1].unsqueeze(0))[0], seq[1:])
        # The last n_generated positions are the ones that predicted the answer's tokens
        tokens = [setup["itos"][int(t)] for t in seq[-n_generated:]]
        pairs = [f"{tok} {float(lp):.2f}" for tok, lp in zip(tokens, log_probs[-n_generated:])]
        item = setup["example"]
        print(f"Log-prob of each token of the greedy answer to {item['prompt']!r} (want {item['answer']!r}):")
        print("  " + "   ".join(pairs))

    return run_step("Step 6: gather_token_log_probs()", show,
                    lambda: check_gather_token_log_probs(gather_token_log_probs))


# ---------------------------------------------------------------------------
# Steps 7-9: the loss, the penalty, and the optimizer step (tests only here;
# they run on the real model in the training loop below)
# ---------------------------------------------------------------------------


def step_7() -> str:
    return run_step("Step 7: pg_loss()", lambda: None, lambda: check_pg_loss(pg_loss))


def step_8() -> str:
    return run_step("Step 8: kl_penalty()", lambda: None, lambda: check_kl_penalty(kl_penalty))


def step_9() -> str:
    return run_step("Step 9: grpo_step()", lambda: None, lambda: check_grpo_step(grpo_step))


# ---------------------------------------------------------------------------
# Step 10: the mean reward (the training-curve metric)
# ---------------------------------------------------------------------------


def step_10() -> str:
    def show():
        require(10, needs=[1, 2, 3], why="to score the group it averages")
        print(f"Mean reward of the Step 1 group: {mean_reward(example_rewards()):.3f}")

    return run_step("Step 10: mean_reward()", show, lambda: check_mean_reward(mean_reward))


# ---------------------------------------------------------------------------
# The payoff: GRPO training with all ten pieces together
# ---------------------------------------------------------------------------


def greedy_accuracy(policy, setup: dict) -> float:
    """Greedy (argmax) held-out accuracy: fraction whose top-guess answer verifies."""
    rewards = [verifiable_reward(greedy_answer(policy, item["prompt"], setup), item["answer"])
               for item in setup["eval"]]
    return mean_reward(torch.tensor(rewards))


def sampled_accuracy(policy, setup: dict) -> float:
    """Sampled held-out accuracy: fraction of sampled completions that verify.

    This is the metric GRPO actually optimizes (the policy samples at TEMPERATURE),
    and the one that moves most: it measures how *reliable* the model is, not just
    whether its single top guess happens to be right.
    """
    rewards = []
    for item in setup["eval"]:
        ids = prompt_ids(item["prompt"], setup["stoi"])
        gen = torch.Generator().manual_seed(SEED)
        for _ in range(EVAL_SAMPLES):
            out = generate(policy, ids, MAX_NEW_TOKENS, BLOCK_SIZE, temperature=TEMPERATURE, generator=gen)
            answer = response_text(out[0], ids.shape[1], setup["itos"])
            rewards.append(verifiable_reward(answer, item["answer"]))
    return mean_reward(torch.tensor(rewards))


def show_sample(policy, setup: dict) -> None:
    """Print the running example's greedy answer and whether it verifies."""
    item = setup["example"]
    answer = greedy_answer(policy, item["prompt"], setup)
    verdict = "correct" if answer == item["answer"] else "wrong"
    print(f"  sample: {item['prompt']!r} -> {answer!r}  (want {item['answer']!r}: {verdict})")


def completion_losses(policy, reference, seqs, advantages, prompt_len, end_id) -> list[torch.Tensor]:
    """Per-completion policy-gradient loss + KL penalty for one prompt's group (Steps 5-8)."""
    losses = []
    for seq, adv in zip(seqs, advantages):
        seq = truncate_at_end(seq, prompt_len, end_id)
        if seq.shape[0] - prompt_len < 1:
            continue  # nothing was generated before <|end|>
        inp = seq[:-1].unsqueeze(0)  # what the model reads
        targets = seq[1:]            # the token it should predict at each position
        mask = completion_mask(prompt_len, seq.shape[0])
        policy_lp = gather_token_log_probs(policy(inp)[0], targets)
        with torch.no_grad():
            ref_lp = gather_token_log_probs(reference(inp)[0], targets)
        losses.append(pg_loss(policy_lp, adv.item(), mask) + BETA * kl_penalty(policy_lp, ref_lp, mask))
    return losses


def train(setup: dict) -> tuple[list[int], list[float]]:
    """Run MAX_STEPS GRPO steps on the training prompts; return the reward curve."""
    policy, reference = setup["policy"], setup["reference"]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)  # for sampling completions
    rng = torch.Generator().manual_seed(SEED)  # for picking prompts
    curve_steps: list[int] = []
    curve_rewards: list[float] = []

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
        progress(f"  GRPO training: step {step + 1}/{MAX_STEPS}")

        # One optimizer step over a batch of prompts, each with its own group.
        step_losses: list[torch.Tensor] = []
        for _ in range(PROMPTS_PER_STEP):
            item = setup["train"][torch.randint(len(setup["train"]), (1,), generator=rng).item()]
            ids = prompt_ids(item["prompt"], setup["stoi"])
            prompt_len = ids.shape[1]

            # Steps 1-4 and 10: sample a group, score it, compare each answer to the group
            policy.eval()
            seqs = sample_group(policy, ids, GROUP_SIZE, MAX_NEW_TOKENS,
                                BLOCK_SIZE, TEMPERATURE, generate, gen)
            responses = [response_text(s, prompt_len, setup["itos"]) for s in seqs]
            rewards = score_group(responses, item["answer"])
            advantages = group_relative_advantages(rewards)
            interval_rewards.append(mean_reward(rewards))

            # Steps 5-8: the loss for every completion in the group
            policy.train()
            step_losses.extend(
                completion_losses(policy, reference, seqs, advantages, prompt_len, setup["end_id"]))

        # Step 9: one optimizer step on the average loss
        loss = torch.stack(step_losses).mean() if step_losses else torch.zeros((), requires_grad=True)
        grpo_step(optimizer, loss, policy, GRAD_CLIP)
    return curve_steps, curve_rewards


def step_train() -> str:
    """No new code: improve the policy with GRPO and measure it before and after."""
    outcome: dict = {}  # filled in by show(), then judged by the tests

    def show():
        require(None, needs=list(STEP_NAMES), why="to train the policy")
        setup = load_setup()
        policy = setup["policy"]
        cfg = policy.cfg
        print(f"TinyGPT: {cfg.n_layer} layers, {cfg.n_head} heads, width {cfg.n_embd}, "
              f"{policy.num_params():,} parameters (the reference is a frozen copy)")
        print("Task: reverse a string, verified by a Python function (no reward model)")
        print(f"Train prompts: {len(setup['train'])}   Held-out prompts: {len(setup['eval'])}")
        print(f"Group size G={GROUP_SIZE}, temperature={TEMPERATURE}, beta(KL)={BETA}, lr={LR}")
        print()

        # BEFORE: held-out accuracy and the example's greedy answer
        progress("  GRPO training: measuring the starting policy")
        acc_before = sampled_accuracy(policy, setup)
        greedy_before = greedy_accuracy(policy, setup)
        print("Before GRPO:")
        print(f"  Held-out accuracy, sampled (temp {TEMPERATURE}): {acc_before:.1%}   <- what GRPO optimizes")
        print(f"  Held-out accuracy, greedy (argmax):     {greedy_before:.1%}")
        show_sample(policy, setup)
        print()

        # TRAINING: sample, score, advantages, loss, step, MAX_STEPS times
        curve_steps, curve_rewards = train(setup)
        print()

        # AFTER: the same measurements, and the reward-curve image
        progress("  GRPO training: measuring the trained policy")
        acc_after = sampled_accuracy(policy, setup)
        greedy_after = greedy_accuracy(policy, setup)
        progress("")  # clear the progress line
        print("After GRPO:")
        print(f"  Held-out accuracy, sampled (temp {TEMPERATURE}): {acc_after:.1%}   (was {acc_before:.1%})")
        print(f"  Held-out accuracy, greedy (argmax):     {greedy_after:.1%}   (was {greedy_before:.1%})")
        show_sample(policy, setup)
        save_reward_curve(curve_steps, curve_rewards, OUTPUT_DIR / "reward_curve.png", acc_before, acc_after)
        print("  Reward curve saved to output/reward_curve.png")

        outcome.update(curve_rewards=curve_rewards, acc_before=acc_before, acc_after=acc_after,
                       greedy_before=greedy_before, greedy_after=greedy_after)

    return run_step("GRPO training (Steps 1-10 together)", show, lambda: check_training(outcome))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Each --step choice and the function that runs it
STEPS = {
    "1": step_1, "2": step_2, "3": step_3, "4": step_4, "5": step_5,
    "6": step_6, "7": step_7, "8": step_8, "9": step_9, "10": step_10,
    "train": step_train,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO with a verifiable reward")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run: 1-10, or train for the GRPO training run (default: all)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    load_setup()  # load the models and prompts up front, before any step runs
    for name, step in STEPS.items():
        if args.step in ("all", name):
            step()


if __name__ == "__main__":
    main()
