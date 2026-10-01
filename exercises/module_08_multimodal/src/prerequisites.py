"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. Most steps call the code from several
earlier steps at once. If Step 2 were unfinished, running Step 5 would
report Step 2's TODO under Step 5's header. `require()` avoids that: it
first tries the current step's own blank on a tiny input, then each earlier
step it depends on, and names the first one that is still missing.
"""

from __future__ import annotations

import torch

from exercise import (
    captioning_loss,
    clip_loss,
    greedy_next_token,
    image_to_prefix,
    l2_normalize,
    patchify,
    pool_patches,
    similarity_matrix,
)

# Each student function, with a tiny input that runs it. Each call raises
# NotImplementedError while that step's blank is unfinished, and runs instantly.
_PROBES = {
    "1": ("patchify", lambda: patchify(torch.zeros(1, 3, 8, 8), 4)),
    "2": ("pool_patches", lambda: pool_patches(torch.zeros(1, 2, 3))),
    "3": ("l2_normalize", lambda: l2_normalize(torch.ones(1, 2))),
    "4": ("similarity_matrix", lambda: similarity_matrix(torch.ones(2, 2), torch.ones(2, 2), 1.0)),
    "5": ("clip_loss", lambda: clip_loss(torch.eye(2))),
    "6": ("image_to_prefix", lambda: image_to_prefix(torch.zeros(1, 2), torch.nn.Linear(2, 4), 2)),
    "7": ("captioning_loss", lambda: captioning_loss(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long),
                                                     torch.ones(2, dtype=torch.bool))),
    "8": ("greedy_next_token", lambda: greedy_next_token(torch.zeros(1, 2, 3))),
}


def require(step: str, needs: str, purpose: str) -> None:
    """Raise NotImplementedError if `step` or any step in `needs` is unfinished.

    The current step is tried first so that its own TODO message is the one
    reported. A missing earlier step gets a "needs Step N" message instead.
    """
    _PROBES[step][1]()  # lets this step's own TODO message through unchanged
    for earlier in needs:
        name, probe = _PROBES[earlier]
        try:
            probe()
        except NotImplementedError:
            raise NotImplementedError(f"needs Step {earlier} ({name}) to {purpose}") from None
        except Exception:  # noqa: BLE001 - a crash shows up in that step's own report
            pass
