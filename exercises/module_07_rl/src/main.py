"""
Module 7 Exercise runner: GRPO on the instruct model with a verifiable reward

Run with:
    uv run python module_07_rl/src/main.py

Every step is tagged on its header line, then its output follows: what your
code produced on the real model and the result of each test in tests/.
The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

After the ten steps comes the payoff: GRPO training with all ten pieces
together. It loads the bundled Module 6 instruct checkpoint as both the policy
and the frozen reference, then improves it on a task it can verify itself:
reversing a string. It reports held-out accuracy before and after, the mean
reward over training, and a sample completion before and after, and it saves a
reward-curve image. Training takes a few minutes on a laptop CPU.

Add --step N to run one step (1-10, or "train" for the training run).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided model / tokenizer / data / plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

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

# One test file per step lives in tests/
from tests.test_step1_sample_group import check_sample_group  # noqa: E402
from tests.test_step2_verifiable_reward import check_verifiable_reward  # noqa: E402
from tests.test_step3_score_group import check_score_group  # noqa: E402
from tests.test_step4_advantages import check_group_relative_advantages  # noqa: E402
from tests.test_step5_completion_mask import check_completion_mask  # noqa: E402
from tests.test_step6_log_probs import check_gather_token_log_probs  # noqa: E402
from tests.test_step7_pg_loss import check_pg_loss  # noqa: E402
from tests.test_step8_kl_penalty import check_kl_penalty  # noqa: E402
from tests.test_step9_grpo_step import check_grpo_step  # noqa: E402
from tests.test_step10_mean_reward import check_mean_reward  # noqa: E402
from tests.test_training import check_training  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (small enough to run on a laptop CPU in a few minutes)
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
OUTPUT_DIR = _THIS_DIR.parent / "output"
END_TOKEN = "<|end|>"


# ---------------------------------------------------------------------------
# Loading the model and data, and turning text into token ids and back
# ---------------------------------------------------------------------------


def _find_data_file(name: str) -> Path:
    """Walk up from src/ until we find data/<name>."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


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


def _greedy_answer(policy, prompt: str, setup: dict) -> str:
    """The model's single most likely answer (argmax at every token)."""
    prompt_ids = _gen_prompt_ids(prompt, setup["special"], setup["enc"])
    out = generate(policy, prompt_ids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
    return _response_text(out[0], prompt_ids.shape[1], setup["itos"])


def load_setup() -> dict:
    """Load the policy, the frozen reference, the tokenizer maps, and the prompts."""
    # data/instruct_model.pt ships with the repo. It is the Module 6 story finished for
    # us: the Module 5 base model after supervised finetuning, built by
    # src/make_instruct_checkpoint.py.
    ckpt = _find_data_file("instruct_model.pt")
    policy, stoi, itos = load_instruct_model(ckpt)
    reference, _, _ = load_instruct_model(ckpt)
    for p in reference.parameters():
        p.requires_grad = False  # the reference never trains
    reference.eval()

    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    prompts = load_prompts(_find_data_file("verify_prompts.jsonl"))
    setup = {
        "policy": policy,
        "reference": reference,
        "itos": itos,
        "special": special,
        "enc": lambda s: encode(s, stoi),
        "end_id": special[END_TOKEN],
        "train": [p for p in prompts if p["split"] == "train"],
        "eval": [p for p in prompts if p["split"] == "eval"],
    }

    # The running example: the first held-out prompt the model currently gets WRONG
    # (greedy), so the before/after flip is visible. Fall back to the first one.
    setup["example"] = setup["eval"][0]
    for item in setup["eval"]:
        if _greedy_answer(policy, item["prompt"], setup) != item["answer"]:
            setup["example"] = item
            break
    return setup


# ---------------------------------------------------------------------------
# Measuring the policy and building the GRPO loss (these call your functions)
# ---------------------------------------------------------------------------


def _eval_greedy_accuracy(policy, setup: dict) -> float:
    """Greedy (argmax) held-out accuracy: fraction whose top-guess answer verifies."""
    rewards = [verifiable_reward(_greedy_answer(policy, item["prompt"], setup), item["answer"])
               for item in setup["eval"]]
    return mean_reward(torch.tensor(rewards))


def _eval_sampled_accuracy(policy, setup: dict) -> float:
    """Sampled held-out accuracy: fraction of sampled completions that verify.

    This is the metric GRPO actually optimizes (the policy samples at TEMPERATURE),
    and the one that moves most: it measures how *reliable* the model is, not just
    whether its single top guess happens to be right.
    """
    rewards = []
    for item in setup["eval"]:
        prompt_ids = _gen_prompt_ids(item["prompt"], setup["special"], setup["enc"])
        gen = torch.Generator().manual_seed(SEED)
        for _ in range(EVAL_SAMPLES):
            out = generate(policy, prompt_ids, MAX_NEW_TOKENS, BLOCK_SIZE,
                           temperature=TEMPERATURE, generator=gen)
            resp = _response_text(out[0], prompt_ids.shape[1], setup["itos"])
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


def _show_sample(policy, setup: dict) -> None:
    """Print the running example's greedy answer and whether it verifies."""
    item = setup["example"]
    resp = _greedy_answer(policy, item["prompt"], setup)
    verdict = "correct" if resp == item["answer"] else "wrong"
    print(f"  sample: {item['prompt']!r} -> {resp!r}  (want {item['answer']!r}: {verdict})")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


def _fmt(values, spec: str) -> str:
    """Format a tensor or list of numbers as [a, b, c] with one format spec."""
    return "[" + ", ".join(format(float(v), spec) for v in values) + "]"


# ---------------------------------------------------------------------------
# Which steps are still TODOs? (so a step that builds on another can say so)
# ---------------------------------------------------------------------------

STEP_NAMES = {
    1: "sample_group",
    2: "verifiable_reward",
    3: "score_group",
    4: "group_relative_advantages",
    5: "completion_mask",
    6: "gather_token_log_probs",
    7: "pg_loss",
    8: "kl_penalty",
    9: "grpo_step",
    10: "mean_reward",
}


def _probe_grpo_step() -> None:
    """Call grpo_step once on a throwaway one-weight layer."""
    layer = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.0)  # lr 0: nothing actually changes
    grpo_step(optimizer, layer(torch.zeros(1, 1)).sum(), layer, GRAD_CLIP)


# One call per step on small throwaway inputs; only a NotImplementedError matters
_PROBES = {
    1: lambda: sample_group(None, torch.zeros(1, 1, dtype=torch.long), 1, 1, BLOCK_SIZE, TEMPERATURE,
                            lambda *args, **kwargs: torch.zeros(1, 2, dtype=torch.long), None),
    2: lambda: verifiable_reward("tac", "tac"),
    3: lambda: score_group(["tac"], "tac"),
    4: lambda: group_relative_advantages(torch.tensor([1.0, 0.0])),
    5: lambda: completion_mask(3, 6),
    6: lambda: gather_token_log_probs(torch.zeros(2, 3), torch.tensor([0, 1])),
    7: lambda: pg_loss(torch.zeros(3), 1.0, torch.ones(3, dtype=torch.bool)),
    8: lambda: kl_penalty(torch.zeros(3), torch.zeros(3), torch.ones(3, dtype=torch.bool)),
    9: _probe_grpo_step,
    10: lambda: mean_reward(torch.tensor([1.0, 0.0])),
}


# score_group() calls verifiable_reward(), so an unfinished Step 2 can raise from
# inside a finished Step 3. That TODO belongs to Step 2, not Step 3.
_CALLS_INTO = {3: [2]}


def _todo_text(step: int) -> str | None:
    """The step's own TODO message if its blank is unfinished, else None."""
    try:
        _PROBES[step]()
    except NotImplementedError as e:
        if any(str(e) == _todo_text(n) for n in _CALLS_INTO.get(step, [])):
            return None  # an earlier step's TODO, raised from inside this one
        return str(e)
    except Exception:  # noqa: BLE001 - it ran, so it is not a TODO (its own tests judge it)
        return None
    return None


def _is_todo(step: int) -> bool:
    """True if the step's own blank still raises NotImplementedError."""
    return _todo_text(step) is not None


def _own_todo_first(step: int) -> None:
    """Raise the step's own TODO if its blank is unfinished.

    Called before _require() so an unfinished step always shows its own TODO,
    and "needs Step N" appears only once the step's own function works.
    """
    message = _todo_text(step)
    if message is not None:
        raise NotImplementedError(message)


def _require(steps: list[int], why: str) -> None:
    """Raise a pointed NotImplementedError if any of these earlier steps is a TODO."""
    missing = [n for n in steps if _is_todo(n)]
    if len(missing) == 1:
        raise NotImplementedError(f"needs Step {missing[0]} ({STEP_NAMES[missing[0]]}) {why}")
    if missing:
        numbers = ", ".join(str(n) for n in missing[:-1]) + f" and {missing[-1]}"
        if len(missing) == len(STEP_NAMES):
            numbers = f"1-{len(STEP_NAMES)}"  # all of them
        raise NotImplementedError(f"needs Steps {numbers} {why}")


# ---------------------------------------------------------------------------
# Two running examples: a sampled group for the first held-out prompt (Steps
# 1, 3, 4, 10) and the greedy answer to the example prompt (Steps 5, 6)
# ---------------------------------------------------------------------------


def _example_group(setup: dict, results: dict) -> list[str]:
    """The decoded group of G completions for the first held-out prompt (drawn once, then reused)."""
    if "group" not in results:
        item = setup["eval"][0]
        prompt_ids = _gen_prompt_ids(item["prompt"], setup["special"], setup["enc"])
        gen = torch.Generator().manual_seed(SEED)  # same seed, same group, every run
        seqs = sample_group(setup["policy"], prompt_ids, GROUP_SIZE, MAX_NEW_TOKENS,
                            BLOCK_SIZE, TEMPERATURE, generate, gen)
        results["group"] = [_response_text(s, prompt_ids.shape[1], setup["itos"]) for s in seqs]
    return results["group"]


def _example_rewards(setup: dict, results: dict, why: str) -> torch.Tensor:
    """The verifier's reward for each completion in the example group (needs Steps 1-3)."""
    if "rewards" not in results:
        _require([1, 2, 3], why)
        group = _example_group(setup, results)
        results["rewards"] = score_group(group, setup["eval"][0]["answer"])
    return results["rewards"]


def _example_sequence(setup: dict) -> torch.Tensor:
    """prompt + the model's greedy answer for the example, cut after <|end|>."""
    if "sequence" not in setup:
        prompt_ids = _gen_prompt_ids(setup["example"]["prompt"], setup["special"], setup["enc"])
        out = generate(setup["policy"], prompt_ids, MAX_NEW_TOKENS, BLOCK_SIZE, greedy=True)
        setup["sequence"] = _truncate_at_end(out[0], prompt_ids.shape[1], setup["end_id"])
        setup["prompt_len"] = prompt_ids.shape[1]
    return setup["sequence"]


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1(setup: dict, results: dict) -> str:
    """Sample a group of G completions for the first held-out prompt."""

    def show():
        item = setup["eval"][0]
        group = _example_group(setup, results)
        print(f"Held-out prompt {item['prompt']!r}; the verifier wants {item['answer']!r}")
        print(f"A group of G={GROUP_SIZE} completions sampled at temperature {TEMPERATURE}:")
        print("  " + "  ".join(repr(text) for text in group))

    return run_step("Step 1: sample_group()", show, lambda: check_sample_group(sample_group))


def step_2(setup: dict, results: dict) -> str:
    """Verify the instruct model's greedy answers on a few held-out prompts."""

    def show():
        print("The starting model's greedy answers on 4 held-out prompts:")
        for item in setup["eval"][:4]:
            answer = _greedy_answer(setup["policy"], item["prompt"], setup)
            reward = verifiable_reward(answer, item["answer"])
            print(f"  {item['prompt']!r} -> {answer!r}  want {item['answer']!r}  reward {reward}")

    return run_step("Step 2: verifiable_reward()", show, lambda: check_verifiable_reward(verifiable_reward))


def step_3(setup: dict, results: dict) -> str:
    """Score the Step 1 group with the verifier."""

    def show():
        _own_todo_first(3)
        _require([1, 2], "to score a real group")
        group = _example_group(setup, results)
        rewards = score_group(group, setup["eval"][0]["answer"])
        # Keep a float copy for the later demos, whatever type came back
        results["rewards"] = torch.as_tensor(rewards, dtype=torch.float32)
        print(f"Rewards for the Step 1 group: {_fmt(rewards, 'g')}")

    return run_step("Step 3: score_group()", show, lambda: check_score_group(score_group))


def step_4(setup: dict, results: dict) -> str:
    """Turn the example group's rewards into advantages."""

    def show():
        _own_todo_first(4)
        rewards = _example_rewards(setup, results, "to score the group it standardizes")
        advantages = group_relative_advantages(rewards)
        print(f"Rewards for the Step 1 group: {_fmt(rewards, '5.2f')}")
        print(f"Advantages:                   {_fmt(advantages, '+5.2f')}")

    return run_step("Step 4: group_relative_advantages()", show,
                    lambda: check_group_relative_advantages(group_relative_advantages))


def step_5(setup: dict, results: dict) -> str:
    """Mask the example's greedy sequence: which positions predict a generated token?"""

    def show():
        seq = _example_sequence(setup)
        prompt_len = setup["prompt_len"]
        mask = completion_mask(prompt_len, seq.shape[0])
        answer = _response_text(seq, prompt_len, setup["itos"])
        print(f"Greedy answer to {setup['example']['prompt']!r}: {answer!r} + <|end|>, "
              f"so {prompt_len} prompt tokens + {seq.shape[0] - prompt_len} generated")
        print(f"Mask over the {len(mask)} next-token positions: {''.join('1' if m else '0' for m in mask)}")

    return run_step("Step 5: completion_mask()", show, lambda: check_completion_mask(completion_mask))


def step_6(setup: dict, results: dict) -> str:
    """How confident is the policy in each token of its greedy answer?"""

    def show():
        seq = _example_sequence(setup)
        n_generated = seq.shape[0] - setup["prompt_len"]
        with torch.no_grad():
            log_probs = gather_token_log_probs(setup["policy"](seq[:-1].unsqueeze(0))[0], seq[1:])
        # The last n_generated positions are the ones that predicted the answer's tokens
        tokens = [setup["itos"][int(t)] for t in seq[-n_generated:]]
        pairs = [f"{tok} {float(lp):.2f}" for tok, lp in zip(tokens, log_probs[-n_generated:])]
        item = setup["example"]
        print(f"Log-prob of each token of the greedy answer to {item['prompt']!r} (want {item['answer']!r}):")
        print("  " + "   ".join(pairs))

    return run_step("Step 6: gather_token_log_probs()", show,
                    lambda: check_gather_token_log_probs(gather_token_log_probs))


def step_7(setup: dict, results: dict) -> str:
    return run_step("Step 7: pg_loss()", lambda: None, lambda: check_pg_loss(pg_loss))


def step_8(setup: dict, results: dict) -> str:
    return run_step("Step 8: kl_penalty()", lambda: None, lambda: check_kl_penalty(kl_penalty))


def step_9(setup: dict, results: dict) -> str:
    return run_step("Step 9: grpo_step()", lambda: None, lambda: check_grpo_step(grpo_step))


def step_10(setup: dict, results: dict) -> str:
    """The training-curve metric, on the Step 1 group."""

    def show():
        _own_todo_first(10)
        rewards = _example_rewards(setup, results, "to score the group it averages")
        print(f"Mean reward of the Step 1 group: {mean_reward(rewards):.3f}")

    return run_step("Step 10: mean_reward()", show, lambda: check_mean_reward(mean_reward))


# ---------------------------------------------------------------------------
# The payoff: GRPO training with all ten pieces together
# ---------------------------------------------------------------------------


def _progress(text: str) -> None:
    """Show live progress on a terminal; an empty text clears the line.

    It goes to stderr, so it is not part of the step's output.
    """
    if sys.stderr.isatty():
        print(f"\r{text:<60}\r", end="", file=sys.stderr, flush=True)


def _train(setup: dict) -> tuple[list[int], list[float]]:
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
        _progress(f"  GRPO training: step {step + 1}/{MAX_STEPS}")

        # One optimizer step over a batch of prompts, each with its own group.
        step_losses: list[torch.Tensor] = []
        for _ in range(PROMPTS_PER_STEP):
            item = setup["train"][torch.randint(len(setup["train"]), (1,), generator=rng).item()]
            prompt_ids = _gen_prompt_ids(item["prompt"], setup["special"], setup["enc"])
            prompt_len = prompt_ids.shape[1]

            policy.eval()
            seqs = sample_group(policy, prompt_ids, GROUP_SIZE, MAX_NEW_TOKENS,
                                BLOCK_SIZE, TEMPERATURE, generate, gen)
            responses = [_response_text(s, prompt_len, setup["itos"]) for s in seqs]
            rewards = score_group(responses, item["answer"])
            advantages = group_relative_advantages(rewards)
            interval_rewards.append(mean_reward(rewards))

            policy.train()
            step_losses.extend(
                _completion_losses(policy, reference, seqs, advantages, prompt_len, setup["end_id"]))

        loss = torch.stack(step_losses).mean() if step_losses else torch.zeros((), requires_grad=True)
        grpo_step(optimizer, loss, policy, GRAD_CLIP)
    return curve_steps, curve_rewards


def step_train(setup: dict, results: dict) -> str:
    """No new code: improve the policy with GRPO and measure it before and after."""

    def show():
        _require(list(STEP_NAMES), "to train the policy")
        policy = setup["policy"]
        cfg = policy.cfg
        print(f"TinyGPT: {cfg.n_layer} layers, {cfg.n_head} heads, width {cfg.n_embd}, "
              f"{policy.num_params():,} parameters (the reference is a frozen copy)")
        print("Task: reverse a string, verified by a Python function (no reward model)")
        print(f"Train prompts: {len(setup['train'])}   Held-out prompts: {len(setup['eval'])}")
        print(f"Group size G={GROUP_SIZE}, temperature={TEMPERATURE}, beta(KL)={BETA}, lr={LR}")
        print()

        # BEFORE: held-out accuracy and the example's greedy answer
        _progress("  GRPO training: measuring the starting policy")
        acc_before = _eval_sampled_accuracy(policy, setup)
        greedy_before = _eval_greedy_accuracy(policy, setup)
        print("Before GRPO:")
        print(f"  Held-out accuracy, sampled (temp {TEMPERATURE}): {acc_before:.1%}   <- what GRPO optimizes")
        print(f"  Held-out accuracy, greedy (argmax):     {greedy_before:.1%}")
        _show_sample(policy, setup)
        print()

        # TRAINING: sample, score, advantages, loss, step, MAX_STEPS times
        curve_steps, curve_rewards = _train(setup)
        print()

        # AFTER: the same measurements, and the reward-curve image
        _progress("  GRPO training: measuring the trained policy")
        acc_after = _eval_sampled_accuracy(policy, setup)
        greedy_after = _eval_greedy_accuracy(policy, setup)
        _progress("")  # clear the progress line
        print("After GRPO:")
        print(f"  Held-out accuracy, sampled (temp {TEMPERATURE}): {acc_after:.1%}   (was {acc_before:.1%})")
        print(f"  Held-out accuracy, greedy (argmax):     {greedy_after:.1%}   (was {greedy_before:.1%})")
        _show_sample(policy, setup)
        save_reward_curve(curve_steps, curve_rewards, OUTPUT_DIR / "reward_curve.png", acc_before, acc_after)
        print("  Reward curve saved to output/reward_curve.png")

        results.update(curve_rewards=curve_rewards, acc_before=acc_before, acc_after=acc_after,
                       greedy_before=greedy_before, greedy_after=greedy_after)

    return run_step("GRPO training (Steps 1-10 together)", show, lambda: check_training(results))


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
    steps = list(STEPS) if args.step == "all" else [args.step]

    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup = load_setup()  # the policy, the frozen reference, tokenizer maps, prompts
    results: dict = {}    # the example group and the training outcomes, shared between steps

    for step in steps:
        STEPS[step](setup, results)


if __name__ == "__main__":
    main()
