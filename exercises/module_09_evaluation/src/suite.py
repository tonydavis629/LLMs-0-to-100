"""Loading the benchmark suite, provided for you.

You do NOT need to edit this file. `load_suite()` reads both checkpoints and
every evaluation file from data/ and bundles them into one `Suite` that the
steps in src/main.py share.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data import load_jsonl
from src.model import load_instruct_model
from src.tokenizer import SPECIAL_TOKENS, encode

_THIS_DIR = Path(__file__).resolve().parent


def find_data_file(name: str) -> Path:
    """Walk up from this file until we find data/<name>."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


@dataclass
class Suite:
    """Everything the steps share: both models, the tokenizer maps, and the data."""

    models: dict           # "instruct" / "rl" -> a loaded TinyGPT
    stoi: dict             # character -> token id (same for both models)
    itos: dict             # token id -> character
    special: dict          # special token string -> its id
    cases: list            # the 50 generated-answer cases from tasks.jsonl
    mc_cases: list         # the 16 multiple-choice questions
    held_out: str          # held-out text for perplexity
    counts: dict           # task -> number of cases in that task, in report order

    def enc(self, text: str) -> list[int]:
        """Encode a plain string into token ids."""
        return encode(text, self.stoi)


def load_suite(model_files: list[tuple[str, str]], task_order: list[str]) -> Suite:
    """Load both checkpoints and every evaluation file.

    `model_files` pairs each model's short name with its checkpoint in data/.
    """
    models = {}
    for name, filename in model_files:
        model, stoi, itos = load_instruct_model(find_data_file(filename))
        models[name] = model
    cases = load_jsonl(find_data_file("tasks.jsonl"))
    return Suite(
        models=models,
        stoi=stoi,
        itos=itos,
        special={tok: stoi[tok] for tok in SPECIAL_TOKENS},
        cases=cases,
        mc_cases=load_jsonl(find_data_file("multiple_choice.jsonl")),
        held_out=find_data_file("held_out.txt").read_text(encoding="utf-8"),
        counts={task: sum(1 for c in cases if c["task"] == task) for task in task_order},
    )
