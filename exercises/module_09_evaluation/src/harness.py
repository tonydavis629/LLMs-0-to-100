"""Running the models on the suite, provided for you.

You do NOT need to edit this file. None of this is a metric. It is the
evaluation harness: the plumbing that turns a checkpoint plus a case into the
raw material a metric consumes (a loss, a generated answer, or the
log-probability of each multiple-choice option).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model import generate
from src.tokenizer import decode

END_TOKEN = "<|end|>"


@torch.no_grad()
def mean_token_loss(model, text: str, stoi: dict[str, int], block_size: int) -> float:
    """Average next-token cross-entropy over held-out text, in nats per token.

    The text is cut into non-overlapping block_size windows and scored the same way
    Module 5 measured validation loss. This is the input to perplexity, and the one
    measurement in the suite that needs no labels and no generation.
    """
    ids = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(ids) - block_size - 1, block_size):
        window = ids[start:start + block_size + 1]
        logits = model(window[:-1].unsqueeze(0))[0]
        loss = F.cross_entropy(logits, window[1:], reduction="sum")
        total_loss += loss.item()
        total_tokens += window.shape[0] - 1
    return total_loss / total_tokens


def _prefix_ids(prompt: str, special: dict[str, int], enc) -> torch.Tensor:
    """The chat-template generation prefix: [user] prompt [end] [assistant]."""
    ids = ([special["<|user|>"]] + enc(prompt) + [special["<|end|>"]]
           + [special["<|assistant|>"]])
    return torch.tensor([ids], dtype=torch.long)


def _response_text(seq: torch.Tensor, prompt_len: int, itos) -> str:
    """Decode the generated portion and cut it at the first <|end|>."""
    return decode(seq[prompt_len:], itos).split(END_TOKEN, 1)[0]


def run_cases(model, cases, special, enc, itos, *, max_new_tokens: int, block_size: int,
              n_samples: int, temperature: float, seed: int) -> list[dict]:
    """Generate one greedy answer and n_samples sampled answers per case.

    Two decoding protocols on the same model in one pass: the greedy answer is what
    exact match and F1 score, and the sampled answers are what pass@k needs. The
    sampler is seeded per case so the run is reproducible.
    """
    out = []
    for case in cases:
        prefix = _prefix_ids(case["prompt"], special, enc)
        prompt_len = prefix.shape[1]
        greedy = generate(model, prefix, max_new_tokens, block_size, greedy=True)
        gen = torch.Generator().manual_seed(seed)
        samples = [
            _response_text(
                generate(model, prefix, max_new_tokens, block_size,
                         temperature=temperature, generator=gen)[0],
                prompt_len, itos)
            for _ in range(n_samples)
        ]
        out.append({
            "id": case["id"],
            "task": case["task"],
            "prompt": case["prompt"],
            "answers": case["answers"],
            "greedy": _response_text(greedy[0], prompt_len, itos),
            "samples": samples,
        })
    return out


@torch.no_grad()
def option_log_probs(model, question: str, options: list[str], special, enc):
    """Total log-probability and token count for each multiple-choice option.

    No generation happens here. Each option is appended to the prompt and scored
    under the model, exactly the way MMLU and HellaSwag are run on base models. The
    two lists returned are what `score_multiple_choice` compares.
    """
    totals, lengths = [], []
    for option in options:
        prefix = _prefix_ids(question, special, enc)[0].tolist()
        option_ids = enc(option)
        seq = torch.tensor([prefix + option_ids], dtype=torch.long)
        logits = model(seq[:, :-1])[0]
        log_probs = F.log_softmax(logits, dim=-1)
        # Score only the option's own tokens: positions len(prefix)-1 .. end predict them.
        scored = log_probs[len(prefix) - 1:].gather(
            -1, torch.tensor(option_ids).unsqueeze(-1)).squeeze(-1)
        totals.append(scored.sum().item())
        lengths.append(len(option_ids))
    return totals, lengths
