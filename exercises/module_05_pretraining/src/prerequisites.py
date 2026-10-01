"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. The training loop calls the code from
several steps at once. If Step 4 were unfinished, running Step 7 would
report Step 4's TODO under Step 7's header. Two helpers avoid that:

- `try_own_blank(step)` calls the step's own function once on tiny inputs,
  so its own TODO message is the first one reported.
- `require(step, purpose)` raises a "needs Step N" message if an earlier
  step is still unfinished.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn

from exercise import compute_loss, estimate_loss, get_batch, train_step, train_val_split

# Short names for the steps, used in "needs Step N (...)" messages
STEP_NAMES = {
    "1": "encode",
    "2": "train_val_split",
    "3": "get_batch",
    "4": "compute_loss",
    "5": "train_step",
    "7": "estimate_loss",
}


def _tiny_model() -> nn.Embedding:
    """A 3-token bigram model of zeros: just enough to call a function once."""
    return nn.Embedding.from_pretrained(torch.zeros(3, 3), freeze=False)


def _tiny_ids() -> torch.Tensor:
    """A short stream of token IDs (0, 1, 2, 0, 1, 2, ...) for the tiny model."""
    return torch.arange(12) % 3


def _placeholder_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stand-in for compute_loss(): a zero that autograd can backpropagate (not the answer)."""
    return logits.sum() * 0.0


def _placeholder_batch(data, block_size, batch_size, generator=None):
    """Stand-in for get_batch(): the first block, repeated (not the answer)."""
    x = data[:block_size].repeat(batch_size, 1)
    return x, x


def _with_placeholders(fn, *args, **placeholders) -> None:
    """Call fn once with some of the exercise functions it uses swapped out.

    `placeholders` replace the earlier-step functions it calls (such as
    compute_loss) for this one call, so an unfinished earlier step cannot hide
    this step's TODO. A crash is ignored here; the real demo reports it.
    """
    namespace = inspect.unwrap(fn).__globals__  # the exercise module's variables
    saved = {name: namespace[name] for name in placeholders}
    namespace.update(placeholders)
    try:
        fn(*args)
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001 - reported by the real demo instead
        pass
    finally:
        namespace.update(saved)  # put the real functions back


def _probe_train_step() -> None:
    """Run train_step once on the tiny model, so the real model is untouched."""
    model = _tiny_model()
    ids = _tiny_ids()[:4].view(1, 4)
    train_step(model, torch.optim.SGD(model.parameters(), lr=0.1), ids, ids)


# One tiny call per step. Each raises NotImplementedError while that step's
# blank is unfinished, and runs in a fraction of a second.
_PROBES = {
    "3": lambda: get_batch(_tiny_ids(), 4, 1, torch.Generator().manual_seed(0)),
    "4": lambda: compute_loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long)),
    "5": _probe_train_step,
}


def try_own_blank(step: str) -> None:
    """Call this step's own function on tiny inputs, so its own TODO shows first."""
    if step == "2":
        train_val_split(torch.arange(10))
    elif step in ("3", "4"):
        _PROBES[step]()
    elif step == "5":
        tiny, ids = _tiny_model(), _tiny_ids()[:4].view(1, 4)
        _with_placeholders(train_step, tiny, torch.optim.SGD(tiny.parameters(), lr=0.1), ids, ids,
                           compute_loss=_placeholder_loss)
    elif step == "7":
        _with_placeholders(estimate_loss, _tiny_model(), _tiny_ids(), 4, 1, 1,
                           torch.Generator().manual_seed(0),
                           get_batch=_placeholder_batch, compute_loss=_placeholder_loss)


def _finished(step: str) -> bool:
    """True unless the step's probe raises NotImplementedError."""
    try:
        _PROBES[step]()
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - the student wrote *something*; its own step reports the crash
        return True
    return True


def require(step: str, purpose: str, done: bool | None = None) -> None:
    """Raise NotImplementedError naming `step` if it is unfinished.

    By default the step is probed on tiny inputs. Pass `done` instead when the
    question is whether the step already ran (for example, whether Step 2's
    token streams exist yet).
    """
    if done is None:
        done = _finished(step)
    if not done:
        raise NotImplementedError(f"needs Step {step} ({STEP_NAMES[step]}) {purpose}")
