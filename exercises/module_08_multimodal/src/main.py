"""
Module 8 Exercise runner: Align image embeddings with NanoGPT

Run with:
    uv run python module_08_multimodal/src/main.py

Every step is tagged on its header line, then its output follows: what your
code produced (shapes, training progress, generated captions) and the result
of each test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-8).
Add --solution to run the finished answers from solution/exercise.py.

The steps build a tiny vision-language model on the bundled synthetic shapes
dataset in three stages:

  Steps 1-2  Vision tower: turn a 32x32 image into one embedding.
  Steps 3-5  CLIP alignment: a contrastive loss pulls each image toward its
             caption; held-out retrieval accuracy climbs from chance.
  Steps 6-8  The bridge: project the image embedding into NanoGPT's width as
             visual prefix tokens, finetune so the language model captions the
             image, and check that the answer changes when the image changes.
"""

from __future__ import annotations

import argparse
import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided model / tokenizer / data / vision / plotting.
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

from exercise import (  # noqa: E402  (import after sys.path edits) - the eight student steps
    patchify,
    pool_patches,
    l2_normalize,
    similarity_matrix,
    clip_loss,
    image_to_prefix,
    captioning_loss,
    greedy_next_token,
)
from ops import (  # noqa: E402  (provided plumbing, not part of the exercise)
    flatten_patches,
    project_patches,
    add_position_embeddings,
    encode_text,
    retrieval_accuracy,
    concat_visual_prefix,
)
from model import load_instruct_model  # noqa: E402
from tokenizer import encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import load_dataset, build_dataset, save_dataset, questions_for  # noqa: E402
from vision import VisionEncoder, TextEncoder, Projector, PREFIX_LEN, PATCH_SIZE  # noqa: E402
from visualization import save_image_grid, save_retrieval_heatmap  # noqa: E402

# One test file per step lives in tests/
from tests.test_step1_patchify import check_patchify  # noqa: E402
from tests.test_step2_pool_patches import check_pool_patches  # noqa: E402
from tests.test_step3_l2_normalize import check_l2_normalize  # noqa: E402
from tests.test_step4_similarity_matrix import check_similarity_matrix  # noqa: E402
from tests.test_step5_clip_loss import check_clip_loss, check_clip_training  # noqa: E402
from tests.test_step6_image_to_prefix import check_image_to_prefix  # noqa: E402
from tests.test_step7_captioning_loss import check_bridge_training, check_captioning_loss  # noqa: E402
from tests.test_step8_greedy_next_token import check_greedy_next_token, check_grounded_captions  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (small enough to run on a laptop CPU in a couple of minutes)
# ---------------------------------------------------------------------------
TEMPERATURE = 0.07       # CLIP softmax temperature
CLIP_BATCH = 32          # image-caption pairs per contrastive step
CLIP_STEPS = 400         # contrastive training steps
CLIP_LR = 1e-3
CLIP_EVAL_INTERVAL = 50

BRIDGE_BATCH = 16        # image-conditioned examples per bridge step
BRIDGE_STEPS = 400       # bridge (projector + LM) finetuning steps
BRIDGE_LR = 3e-4
BRIDGE_EVAL_INTERVAL = 50
MAX_ANSWER_TOKENS = 40   # generation cap for captions/answers (longest caption ~35 chars)

TEXT_MAXLEN = 40         # padded caption length for the text encoder
SEED = 1337

_THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _THIS_DIR.parent / "output"
DESCRIBE = "describe the image"  # the one prompt used for every caption


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _find_data_file(name: str) -> Path:
    """Walk up from src/ until we find data/<name> (the bundled files)."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def _pad_caption_ids(caption: str, stoi: dict, pad_id: int) -> list[int]:
    """Encode one caption and pad it with <|pad|> up to TEXT_MAXLEN tokens."""
    ids = encode(caption, stoi)[:TEXT_MAXLEN]
    return ids + [pad_id] * (TEXT_MAXLEN - len(ids))


def _caption_batch_ids(captions: list[str], stoi: dict, pad_id: int) -> torch.Tensor:
    """Encode and pad a list of captions into one (B, TEXT_MAXLEN) tensor of ids."""
    return torch.tensor([_pad_caption_ids(c, stoi, pad_id) for c in captions], dtype=torch.long)


# ---------------------------------------------------------------------------
# Vision forward: chains patchify (yours) and pool_patches (yours) around the
# provided flatten/project/position ops and the provided patch mixer.
# ---------------------------------------------------------------------------

def vision_forward(venc: VisionEncoder, images: torch.Tensor) -> torch.Tensor:
    """(B, 3, 32, 32) images -> (B, D_EMBED) pooled image embeddings."""
    patches = patchify(images, venc.patch_size)               # Step 1: (B, 16, 3, 8, 8)
    flat = flatten_patches(patches)                           # (B, 16, 192)
    x = project_patches(flat, venc.patch_proj)                # (B, 16, 64)
    x = add_position_embeddings(x, venc.pos_embed)            # where each patch sat
    x = venc.mix(x)                                           # patches share information
    return pool_patches(x)                                    # Step 2: (B, 64)


# ---------------------------------------------------------------------------
# Bridge helpers: build the chat sequence, its targets, and the response mask.
# ---------------------------------------------------------------------------

def _chat_ids(prompt: str, response: str, special: dict, stoi: dict):
    """Build [user] prompt [end] [assistant] response [end] and the response start."""
    prefix = [special["<|user|>"]] + encode(prompt, stoi) + [special["<|end|>"]] + [special["<|assistant|>"]]
    seq = prefix + encode(response, stoi) + [special["<|end|>"]]
    return torch.tensor(seq, dtype=torch.long), len(prefix)


def _targets_and_mask(seq: torch.Tensor, resp_start: int, k: int):
    """Align next-token targets to a sequence that has k visual prefix slots in front.

    Logits position t predicts input position t+1; input position t+1 is text token
    seq[t+1-k]. We train only where that predicted token is part of the response.
    """
    L = seq.shape[0]
    T = k + L
    pos = torch.arange(T)
    seq_idx = pos + 1 - k
    valid = (seq_idx >= 0) & (seq_idx <= L - 1)
    targets = torch.zeros(T, dtype=torch.long)
    targets[valid] = seq[seq_idx[valid].clamp(min=0)]
    mask = valid & (seq_idx >= resp_start)
    return targets, mask


def _bridge_example_loss(lm, venc, projector, image, prompt, response, special, stoi):
    """Per-example image-conditioned captioning loss (one forward through the LM)."""
    img_embed = vision_forward(venc, image.unsqueeze(0))                 # (1, D)
    prefix = image_to_prefix(img_embed, projector.to_prefix, PREFIX_LEN)  # Step 6: (1, K, d_llm)
    seq, resp_start = _chat_ids(prompt, response, special, stoi)
    tok_embed = lm.embed_tokens(seq.unsqueeze(0))                        # (1, L, d_llm)
    inp = concat_visual_prefix(prefix, tok_embed)                       # (1, K+L, d_llm)
    logits = lm.forward_embeds(inp)[0]                                  # (K+L, V)
    targets, mask = _targets_and_mask(seq, resp_start, PREFIX_LEN)
    return captioning_loss(logits, targets, mask)                       # Step 7


@torch.no_grad()
def generate_answer(lm, venc, projector, image, prompt, special, stoi, itos) -> str:
    """Greedily decode an image-conditioned answer to `prompt` for one image."""
    img_embed = vision_forward(venc, image.unsqueeze(0))
    prefix = image_to_prefix(img_embed, projector.to_prefix, PREFIX_LEN)
    seq = [special["<|user|>"]] + encode(prompt, stoi) + [special["<|end|>"]] + [special["<|assistant|>"]]
    seq_t = torch.tensor(seq, dtype=torch.long)
    start = len(seq)
    end_id = special["<|end|>"]
    for _ in range(MAX_ANSWER_TOKENS):
        tok_embed = lm.embed_tokens(seq_t.unsqueeze(0))
        inp = concat_visual_prefix(prefix, tok_embed)
        logits = lm.forward_embeds(inp)
        nxt = int(greedy_next_token(logits).item())                    # Step 8
        if nxt == end_id:
            break
        seq_t = torch.cat([seq_t, torch.tensor([nxt], dtype=torch.long)])
    return decode(seq_t[start:], itos)


# ---------------------------------------------------------------------------
# CLIP contrastive alignment (Steps 1-5 together)
# ---------------------------------------------------------------------------

def train_clip(venc, tenc, train, stoi, pad_id, rng) -> None:
    """Train both towers so each image lands next to its own caption."""
    optimizer = torch.optim.AdamW(list(venc.parameters()) + list(tenc.parameters()), lr=CLIP_LR)
    images = train["images"]
    captions = train["captions"]
    n = images.shape[0]
    venc.train(); tenc.train()
    for step in range(1, CLIP_STEPS + 1):
        # A random batch of matched image-caption pairs
        idx = torch.randperm(n, generator=rng)[:CLIP_BATCH]
        img_batch = images[idx]
        cap_batch = [captions[i] for i in idx.tolist()]
        cap_ids = _caption_batch_ids(cap_batch, stoi, pad_id)

        img_embed = l2_normalize(vision_forward(venc, img_batch))      # Steps 1-3
        txt_embed = l2_normalize(encode_text(cap_ids, tenc))           # Step 3
        logits = similarity_matrix(img_embed, txt_embed, TEMPERATURE)  # Step 4
        loss = clip_loss(logits)                                       # Step 5

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(venc.parameters()) + list(tenc.parameters()), 1.0)
        optimizer.step()

        if step % CLIP_EVAL_INTERVAL == 0 or step == 1:
            acc = retrieval_accuracy(logits)
            print(f"  step {step:>4}   contrastive loss {loss.item():>6.3f}   batch retrieval acc {acc:>5.1%}")
    venc.eval(); tenc.eval()


@torch.no_grad()
def eval_retrieval(venc, tenc, split, stoi, pad_id):
    """Held-out image-to-caption retrieval accuracy and the similarity matrix."""
    images = split["images"]
    captions = split["captions"]
    cap_ids = _caption_batch_ids(captions, stoi, pad_id)
    img_embed = l2_normalize(vision_forward(venc, images))
    txt_embed = l2_normalize(encode_text(cap_ids, tenc))
    logits = similarity_matrix(img_embed, txt_embed, TEMPERATURE)
    return retrieval_accuracy(logits), logits


# ---------------------------------------------------------------------------
# The bridge: finetune the projector and the language model (Steps 6-7)
# ---------------------------------------------------------------------------

def _bridge_examples(split):
    """Flatten each scene into (image, prompt, response) tasks: describe + 4 VQAs."""
    out = []
    images = split["images"]
    for i in range(images.shape[0]):
        img = images[i]
        out.append((img, DESCRIBE, split["captions"][i]))
        for q, a in questions_for(tuple(split["top"][i]), tuple(split["bottom"][i])):
            out.append((img, q, a))
    return out


def train_bridge(lm, venc, projector, train, special, stoi, rng) -> list[float]:
    """Finetune the projector and the LM on image-conditioned examples; return the losses."""
    # The vision tower is frozen: CLIP already taught it what the shapes are
    for p in venc.parameters():
        p.requires_grad = False
    venc.eval()
    optimizer = torch.optim.AdamW(list(lm.parameters()) + list(projector.parameters()), lr=BRIDGE_LR)
    examples = _bridge_examples(train)
    n = len(examples)
    lm.train(); projector.train()
    losses: list[float] = []
    for step in range(1, BRIDGE_STEPS + 1):
        idx = torch.randperm(n, generator=rng)[:BRIDGE_BATCH].tolist()
        batch_losses = []
        for j in idx:
            img, prompt, response = examples[j]
            batch_losses.append(_bridge_example_loss(lm, venc, projector, img, prompt, response, special, stoi))
        loss = torch.stack(batch_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lm.parameters()) + list(projector.parameters()), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % BRIDGE_EVAL_INTERVAL == 0 or step == 1:
            print(f"  step {step:>4}   captioning loss {loss.item():>6.3f}")
    lm.eval(); projector.eval()
    return losses


def _answer_matches(pred: str, truth: str) -> bool:
    """Exact match, ignoring leading and trailing spaces."""
    return pred.strip() == truth.strip()


def _verdict(pred: str, truth: str) -> str:
    """'(correct)', or '(wrong, want ...)' with the expected answer."""
    return "(correct)" if _answer_matches(pred, truth) else f"(wrong, want {truth!r})"


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


# ---------------------------------------------------------------------------
# Which earlier steps a demo depends on
# ---------------------------------------------------------------------------

# Each student function, with a tiny input that runs it. A step's demo uses these
# to find an unfinished earlier step and name it, instead of failing on that
# step's TODO message.
_STEP_PROBES = {
    1: ("patchify", lambda: patchify(torch.zeros(1, 3, 8, 8), 4)),
    2: ("pool_patches", lambda: pool_patches(torch.zeros(1, 2, 3))),
    3: ("l2_normalize", lambda: l2_normalize(torch.ones(1, 2))),
    4: ("similarity_matrix", lambda: similarity_matrix(torch.ones(2, 2), torch.ones(2, 2), 1.0)),
    5: ("clip_loss", lambda: clip_loss(torch.eye(2))),
    6: ("image_to_prefix", lambda: image_to_prefix(torch.zeros(1, 2), torch.nn.Linear(2, 4), 2)),
    7: ("captioning_loss", lambda: captioning_loss(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long),
                                                   torch.ones(2, dtype=torch.bool))),
    8: ("greedy_next_token", lambda: greedy_next_token(torch.zeros(1, 2, 3))),
}


def _probe(step: int) -> None:
    """Run one step's function on a tiny input; if unfinished, it raises its own TODO."""
    _STEP_PROBES[step][1]()


def _needs(steps: list[int], purpose: str) -> None:
    """Raise a pointed NotImplementedError naming the first unfinished step in `steps`."""
    for step in steps:
        name, probe = _STEP_PROBES[step]
        try:
            probe()
        except NotImplementedError:
            raise NotImplementedError(f"needs Step {step} ({name}) to {purpose}") from None
        except Exception:  # noqa: BLE001 - a crash shows up in that step's own report
            pass


# ---------------------------------------------------------------------------
# Shared training runs: each happens once per run, and later steps reuse it
# ---------------------------------------------------------------------------

def _train_clip_once(ctx: dict, quiet: bool = False) -> None:
    """CLIP-train the vision and text towers (Steps 1-5), unless already done.

    Step 5 shows the training progress. A later step run on its own (--step 7)
    trains quietly and prints one line saying so.
    """
    if "clip_acc_after" in ctx:
        return
    args = (ctx["venc"], ctx["tenc"], ctx["eval"], ctx["stoi"], ctx["pad_id"])
    ctx["clip_acc_before"], _ = eval_retrieval(*args)
    with redirect_stdout(io.StringIO() if quiet else sys.stdout):
        train_clip(ctx["venc"], ctx["tenc"], ctx["train"], ctx["stoi"], ctx["pad_id"], ctx["rng"])
    ctx["clip_acc_after"], ctx["clip_logits"] = eval_retrieval(*args)
    if quiet:
        print(f"(ran the CLIP training from Step 5 first: held-out retrieval {ctx['clip_acc_after']:.1%})")


def _train_bridge_once(ctx: dict, quiet: bool = False) -> None:
    """Finetune the projector and the LM (Steps 6-7), unless already done."""
    if "bridge_losses" in ctx:
        return
    # Keep untouched copies so Step 8 can show what the untrained bridge says
    ctx["untrained_bridge"] = (copy.deepcopy(ctx["lm"]), copy.deepcopy(ctx["projector"]))
    with redirect_stdout(io.StringIO() if quiet else sys.stdout):
        ctx["bridge_losses"] = train_bridge(ctx["lm"], ctx["venc"], ctx["projector"], ctx["train"],
                                            ctx["special"], ctx["stoi"], ctx["rng"])
    if quiet:
        losses = ctx["bridge_losses"]
        print(f"(ran the bridge training from Step 7 first: captioning loss {losses[0]:.3f} -> {losses[-1]:.3f})")


# ---------------------------------------------------------------------------
# The steps. `ctx` holds the data, the models, and results shared between steps.
# ---------------------------------------------------------------------------


def step_1(ctx: dict) -> str:
    """Cut every training image into a sequence of patches."""

    def show():
        images = ctx["train"]["images"]
        patches = patchify(images, PATCH_SIZE)
        n_eval = ctx["eval"]["images"].shape[0]
        print(f"Dataset: {images.shape[0]} training + {n_eval} held-out scenes; "
              "sample grid in output/sample_scenes.png")
        print(f"Images {tuple(images.shape)} -> patches {tuple(patches.shape)}")
        print(f"  each 32 x 32 image is now a sequence of {patches.shape[1]} patches of "
              f"{PATCH_SIZE} x {PATCH_SIZE} pixels")

    return run_step("Step 1: patchify()", show, lambda: check_patchify(patchify))


def step_2(ctx: dict) -> str:
    """Run the whole (untrained) vision tower and print the shape after each stage."""

    def show():
        _probe(2)  # an unfinished pool_patches reports its own TODO
        _needs([1], "cut the images into patches")
        venc = ctx["venc"]
        images = ctx["train"]["images"]
        with torch.no_grad():
            patches = patchify(images, venc.patch_size)
            flat = flatten_patches(patches)
            projected = add_position_embeddings(project_patches(flat, venc.patch_proj), venc.pos_embed)
            mixed = venc.mix(projected)
            pooled = pool_patches(mixed)
        print(f"Vision tower ({venc.num_params():,} parameters, not trained yet) on {images.shape[0]} images:")
        print(f"  patchify (Step 1)             {tuple(patches.shape)}")
        print(f"  flatten each patch            {tuple(flat.shape)}")
        print(f"  project, add positions, mix   {tuple(mixed.shape)}")
        print(f"  pool (Step 2)                 {tuple(pooled.shape)}: one embedding per image")

    return run_step("Step 2: pool_patches()", show, lambda: check_pool_patches(pool_patches))


def step_3(ctx: dict) -> str:
    """Normalize the image embeddings and show that every length becomes 1."""

    def show():
        _probe(3)
        _needs([1, 2], "embed the images it normalizes")
        with torch.no_grad():
            embeds = vision_forward(ctx["venc"], ctx["train"]["images"])
            unit = l2_normalize(embeds)
        before, after = embeds.norm(dim=-1), unit.norm(dim=-1)
        print(f"Lengths of the {embeds.shape[0]} image embeddings:")
        print(f"  before normalizing: {float(before.min()):.3f} to {float(before.max()):.3f}")
        print(f"  after normalizing:  {float(after.min()):.3f} to {float(after.max()):.3f}")

    return run_step("Step 3: l2_normalize()", show, lambda: check_l2_normalize(l2_normalize))


def step_4(ctx: dict) -> str:
    """Score every held-out image against every held-out caption, before any training."""

    def show():
        _probe(4)
        _needs([1, 2, 3], "embed and normalize the held-out images")
        acc, logits = eval_retrieval(ctx["venc"], ctx["tenc"], ctx["eval"], ctx["stoi"], ctx["pad_id"])
        n = logits.shape[0]
        print(f"Similarity matrix of the {n} held-out images x {n} captions: shape {tuple(logits.shape)}")
        print(f"  Retrieval accuracy before training: {acc:.1%}  (chance is 1/{n})")

    return run_step("Step 4: similarity_matrix()", show, lambda: check_similarity_matrix(similarity_matrix))


def step_5(ctx: dict) -> str:
    """CLIP-train both towers with Steps 1-5 together."""

    def show():
        _probe(5)
        _needs([1, 2, 3, 4], "run the contrastive training loop")
        print(f"CLIP training: {CLIP_STEPS} steps of {CLIP_BATCH} image-caption pairs")
        _train_clip_once(ctx)
        before, after = ctx["clip_acc_before"], ctx["clip_acc_after"]
        print(f"  Held-out retrieval accuracy: {before:.1%} before training, {after:.1%} after")
        save_retrieval_heatmap(ctx["clip_logits"], ctx["eval"]["captions"], OUTPUT_DIR / "retrieval_heatmap.png")
        print("  Saved retrieval heatmap to output/retrieval_heatmap.png")

    return run_step("Step 5: clip_loss()", show,
                    lambda: check_clip_loss(clip_loss) + check_clip_training(ctx))


def step_6(ctx: dict) -> str:
    """Project one image into visual prefix tokens and line them up with the prompt."""

    def show():
        _probe(6)
        _needs([1, 2], "embed the image it projects")
        lm, projector, special, stoi = ctx["lm"], ctx["projector"], ctx["special"], ctx["stoi"]
        # The prompt as the language model sees it: [user] prompt [end] [assistant]
        prompt_ids = [special["<|user|>"]] + encode(DESCRIBE, stoi) + [special["<|end|>"], special["<|assistant|>"]]
        with torch.no_grad():
            embed = vision_forward(ctx["venc"], ctx["eval"]["images"][:1])
            prefix = image_to_prefix(embed, projector.to_prefix, PREFIX_LEN)
            tokens = lm.embed_tokens(torch.tensor([prompt_ids]))
            combined = concat_visual_prefix(prefix, tokens)
        print(f"Projector: Linear({embed.shape[-1]} -> {PREFIX_LEN} x {prefix.shape[-1]}), "
              f"{projector.num_params():,} parameters")
        print(f"  image embedding {tuple(embed.shape)} -> visual prefix {tuple(prefix.shape)}")
        print(f"  prompt {DESCRIBE!r} as token embeddings {tuple(tokens.shape)}")
        print(f"  prefix + prompt, the language model's input {tuple(combined.shape)}")

    return run_step("Step 6: image_to_prefix()", show, lambda: check_image_to_prefix(image_to_prefix))


def step_7(ctx: dict) -> str:
    """Finetune the projector and the language model to caption and answer."""

    def show():
        _probe(7)
        _needs([1, 2, 3, 4, 5, 6], "train the bridge")
        _train_clip_once(ctx, quiet=True)  # no-op when Step 5 already trained it
        print(f"Bridge training: {BRIDGE_STEPS} steps of {BRIDGE_BATCH} examples "
              "(a caption and 4 questions per scene)")
        _train_bridge_once(ctx)

    return run_step("Step 7: captioning_loss()", show,
                    lambda: check_captioning_loss(captioning_loss) + check_bridge_training(ctx))


def step_8(ctx: dict) -> str:
    """Caption held-out images and answer questions about one of them."""

    def show():
        _probe(8)
        _needs([1, 2, 3, 4, 5, 6, 7], "train the bridge it decodes from")
        _train_clip_once(ctx, quiet=True)    # no-ops when Steps 5 and 7 already ran
        _train_bridge_once(ctx, quiet=True)
        venc, special, stoi, itos = ctx["venc"], ctx["special"], ctx["stoi"], ctx["itos"]
        split = ctx["eval"]
        demo = range(3)  # the first three held-out scenes

        def answer(lm, projector, i, prompt):
            return generate_answer(lm, venc, projector, split["images"][i], prompt, special, stoi, itos)

        untrained_lm, untrained_projector = ctx["untrained_bridge"]
        print(f"Before bridge training (projector is random), {DESCRIBE!r} gives:")
        for i in demo:
            print(f"    image[{split['captions'][i]!r}] -> {answer(untrained_lm, untrained_projector, i, DESCRIBE)!r}")
        print()

        # Caption every held-out scene with the trained bridge
        lm, projector = ctx["lm"], ctx["projector"]
        total = split["images"].shape[0]
        preds = [answer(lm, projector, i, DESCRIBE) for i in range(total)]
        correct = sum(_answer_matches(p, c) for p, c in zip(preds, split["captions"]))
        ctx["caption_correct"], ctx["caption_total"] = correct, total
        print(f"After bridge training, held-out caption exact-match: {correct}/{total} = {correct / total:.1%}")
        print()

        print("Same prompt, different images (the grounding test):")
        ctx["demo_captions"] = [preds[i] for i in demo]
        for i in demo:
            print(f"    describe -> {preds[i]!r:<37} {_verdict(preds[i], split['captions'][i])}")
        print()

        print("Grounded visual questions on one held-out image:")
        for q, a in questions_for(tuple(split["top"][0]), tuple(split["bottom"][0])):
            pred = answer(lm, projector, 0, q)
            print(f"    {q:<26} -> {pred!r:<10} {_verdict(pred, a)}")

    return run_step("Step 8: greedy_next_token()", show,
                    lambda: check_greedy_next_token(greedy_next_token) + check_grounded_captions(ctx))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4,
         "5": step_5, "6": step_6, "7": step_7, "8": step_8}


def main() -> None:
    parser = argparse.ArgumentParser(description="Align image embeddings with NanoGPT")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = list(STEPS) if args.step == "all" else [args.step]

    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)  # batch order for both training runs

    # Both bundled artifacts ship with the repo. data/instruct_model.pt is the same
    # Module 6 instruct checkpoint Module 7 starts from (built by
    # module_07_rl/solution/src/make_instruct_checkpoint.py); the vision tower gets bolted
    # onto it here. data/shapes_dataset.pt is the synthetic 340 train / 60 held-out
    # image-caption split generated by src/data.py.
    lm, stoi, itos = load_instruct_model(_find_data_file("instruct_model.pt"))
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    pad_id = special["<|pad|>"]
    try:
        data = load_dataset(_find_data_file("shapes_dataset.pt"))
    except FileNotFoundError:
        data = build_dataset()
        save_dataset(data, _THIS_DIR.parent / "data" / "shapes_dataset.pt")

    # The towers and the projector start from random weights (seeded above)
    venc = VisionEncoder()
    tenc = TextEncoder(vocab_size=len(stoi), pad_id=pad_id, max_len=TEXT_MAXLEN)
    projector = Projector(d_llm=lm.cfg.n_embd)

    # A picture of the dataset needs no student code
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_image_grid(data["train"]["images"], data["train"]["captions"], OUTPUT_DIR / "sample_scenes.png")

    # Everything the steps share: data, models, and results passed from step to step
    ctx = {"lm": lm, "stoi": stoi, "itos": itos, "special": special, "pad_id": pad_id,
           "train": data["train"], "eval": data["eval"], "rng": rng,
           "venc": venc, "tenc": tenc, "projector": projector}
    for step in steps:
        STEPS[step](ctx)


if __name__ == "__main__":
    main()
