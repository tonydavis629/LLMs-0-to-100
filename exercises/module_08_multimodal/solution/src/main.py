"""
Module 8 Exercise runner: Align image embeddings with NanoGPT

Run with:
    uv run python module_08_multimodal/src/main.py

Builds a tiny vision-language model on the bundled synthetic shapes dataset in three
visible stages:

  1. Vision tower  - your steps turn a 32x32 image into one embedding.
  2. CLIP alignment - a contrastive loss pulls each image toward its caption and pushes
     it away from the others; retrieval accuracy climbs from chance toward 1.0.
  3. The bridge     - a projector maps the image embedding into NanoGPT's hidden width
     as visual prefix tokens, and we finetune so the language model captions the image
     and answers questions about it. The final check: the answer changes when the
     image changes.

Any step in exercise.py that still raises NotImplementedError is detected and skipped,
so you can implement one step at a time and re-run immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided model / tokenizer / data / vision / plotting.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from vision import VisionEncoder, TextEncoder, Projector, PREFIX_LEN, PATCH_SIZE, N_PATCHES  # noqa: E402
from visualization import save_image_grid, save_retrieval_heatmap  # noqa: E402


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
END_TOKEN = "<|end|>"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

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


def _heading(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _pad_caption_ids(caption: str, stoi: dict, pad_id: int) -> list[int]:
    ids = encode(caption, stoi)[:TEXT_MAXLEN]
    return ids + [pad_id] * (TEXT_MAXLEN - len(ids))


def _caption_batch_ids(captions: list[str], stoi: dict, pad_id: int) -> torch.Tensor:
    return torch.tensor([_pad_caption_ids(c, stoi, pad_id) for c in captions], dtype=torch.long)


# ---------------------------------------------------------------------------
# Vision forward: chains patchify (yours) and pool_patches (yours) around the
# provided flatten/project/position ops and the provided patch mixer.
# ---------------------------------------------------------------------------

def vision_forward(venc: VisionEncoder, images: torch.Tensor) -> torch.Tensor:
    """(B, 3, 32, 32) images -> (B, D_EMBED) pooled image embeddings."""
    patches = patchify(images, venc.patch_size)
    flat = flatten_patches(patches)
    x = project_patches(flat, venc.patch_proj)
    x = add_position_embeddings(x, venc.pos_embed)
    x = venc.mix(x)
    return pool_patches(x)


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
    prefix = image_to_prefix(img_embed, projector.to_prefix, PREFIX_LEN)  # (1, K, d_llm)
    seq, resp_start = _chat_ids(prompt, response, special, stoi)
    tok_embed = lm.embed_tokens(seq.unsqueeze(0))                        # (1, L, d_llm)
    inp = concat_visual_prefix(prefix, tok_embed)                       # (1, K+L, d_llm)
    logits = lm.forward_embeds(inp)[0]                                  # (K+L, V)
    targets, mask = _targets_and_mask(seq, resp_start, PREFIX_LEN)
    return captioning_loss(logits, targets, mask)


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
        nxt = int(greedy_next_token(logits).item())
        if nxt == end_id:
            break
        seq_t = torch.cat([seq_t, torch.tensor([nxt], dtype=torch.long)])
    return decode(seq_t[start:], itos)


# ---------------------------------------------------------------------------
# Step detection
# ---------------------------------------------------------------------------

def _probe_steps(venc, tenc, projector, images, cap_ids) -> dict:
    """Detect which of the eight student steps in exercise.py are implemented."""
    dummy_img = images[:2]
    emb2 = torch.randn(2, 64)
    logits_bb = torch.randn(4, 4)
    small_logits = torch.randn(5, 69)
    small_targets = torch.randint(0, 69, (5,))
    small_mask = torch.tensor([False, True, True, True, True])
    return {
        "patchify": _is_implemented(patchify, dummy_img, PATCH_SIZE),
        "pool_patches": _is_implemented(pool_patches, torch.randn(2, N_PATCHES, 64)),
        "l2_normalize": _is_implemented(l2_normalize, emb2),
        "similarity_matrix": _is_implemented(similarity_matrix, emb2, emb2, TEMPERATURE),
        "clip_loss": _is_implemented(clip_loss, logits_bb),
        "image_to_prefix": _is_implemented(image_to_prefix, emb2, projector.to_prefix, PREFIX_LEN),
        "captioning_loss": _is_implemented(captioning_loss, small_logits, small_targets, small_mask),
        "greedy_next_token": _is_implemented(greedy_next_token, small_logits.unsqueeze(0)),
    }


# ---------------------------------------------------------------------------
# Phase 1: CLIP contrastive alignment
# ---------------------------------------------------------------------------

def train_clip(venc, tenc, train, stoi, pad_id, rng):
    optimizer = torch.optim.AdamW(list(venc.parameters()) + list(tenc.parameters()), lr=CLIP_LR)
    images = train["images"]
    captions = train["captions"]
    n = images.shape[0]
    venc.train(); tenc.train()
    for step in range(1, CLIP_STEPS + 1):
        idx = torch.randperm(n, generator=rng)[:CLIP_BATCH]
        img_batch = images[idx]
        cap_batch = [captions[i] for i in idx.tolist()]
        cap_ids = _caption_batch_ids(cap_batch, stoi, pad_id)

        img_embed = l2_normalize(vision_forward(venc, img_batch))
        txt_embed = l2_normalize(encode_text(cap_ids, tenc))
        logits = similarity_matrix(img_embed, txt_embed, TEMPERATURE)
        loss = clip_loss(logits)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(venc.parameters()) + list(tenc.parameters()), 1.0)
        optimizer.step()

        if step % CLIP_EVAL_INTERVAL == 0 or step == 1:
            acc = retrieval_accuracy(logits)
            print(f"  step {step:>4}   contrastive loss {loss.item():>6.3f}   batch retrieval acc {acc:>5.1%}")
    venc.eval(); tenc.eval()


@torch.no_grad()
def eval_retrieval(venc, tenc, split, stoi, pad_id, n=64):
    images = split["images"][:n]
    captions = split["captions"][:n]
    cap_ids = _caption_batch_ids(captions, stoi, pad_id)
    img_embed = l2_normalize(vision_forward(venc, images))
    txt_embed = l2_normalize(encode_text(cap_ids, tenc))
    logits = similarity_matrix(img_embed, txt_embed, TEMPERATURE)
    return retrieval_accuracy(logits), logits


# ---------------------------------------------------------------------------
# Phase 2: the bridge (projector + LM finetune)
# ---------------------------------------------------------------------------

def _bridge_examples(split, special, stoi):
    """Flatten each scene into (image, prompt, response) tasks: describe + 4 VQAs."""
    out = []
    images = split["images"]
    for i in range(images.shape[0]):
        img = images[i]
        out.append((img, "describe the image", split["captions"][i]))
        for q, a in questions_for(tuple(split["top"][i]), tuple(split["bottom"][i])):
            out.append((img, q, a))
    return out


def train_bridge(lm, venc, projector, train, special, stoi, rng):
    for p in venc.parameters():
        p.requires_grad = False
    venc.eval()
    optimizer = torch.optim.AdamW(list(lm.parameters()) + list(projector.parameters()), lr=BRIDGE_LR)
    examples = _bridge_examples(train, special, stoi)
    n = len(examples)
    lm.train(); projector.train()
    for step in range(1, BRIDGE_STEPS + 1):
        idx = torch.randperm(n, generator=rng)[:BRIDGE_BATCH].tolist()
        losses = []
        for j in idx:
            img, prompt, response = examples[j]
            losses.append(_bridge_example_loss(lm, venc, projector, img, prompt, response, special, stoi))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lm.parameters()) + list(projector.parameters()), 1.0)
        optimizer.step()
        if step % BRIDGE_EVAL_INTERVAL == 0 or step == 1:
            print(f"  step {step:>4}   captioning loss {loss.item():>6.3f}")
    lm.eval(); projector.eval()


def _answer_matches(pred: str, truth: str) -> bool:
    return pred.strip() == truth.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    # Both bundled artifacts ship with the repo. data/instruct_model.pt is the same
    # Module 6 instruct checkpoint Module 7 starts from (built by
    # module_07_rl/solution/src/make_instruct_checkpoint.py); the vision tower gets bolted
    # onto it here. data/shapes_dataset.pt is the synthetic 340 train / 60 held-out
    # image-caption split generated by solution/src/data.py.
    ckpt = _find_data_file("instruct_model.pt")
    lm, stoi, itos = load_instruct_model(ckpt)
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    pad_id = special["<|pad|>"]

    data_path = _find_data_file("shapes_dataset.pt") if any(
        (p / "data" / "shapes_dataset.pt").exists() for p in _THIS_DIR.parents) else None
    if data_path is None:
        data = build_dataset()
        save_dataset(data, _THIS_DIR.parent / "data" / "shapes_dataset.pt")
    else:
        data = load_dataset(data_path)
    train, eval_split = data["train"], data["eval"]

    venc = VisionEncoder()
    tenc = TextEncoder(vocab_size=len(stoi), pad_id=pad_id, max_len=TEXT_MAXLEN)
    projector = Projector(d_llm=lm.cfg.n_embd)

    cap_ids = _caption_batch_ids(train["captions"][:4], stoi, pad_id)
    steps = _probe_steps(venc, tenc, projector, train["images"], cap_ids)

    _heading("MODULE 8: Align image embeddings with NanoGPT")
    print(f"Dataset: {train['images'].shape[0]} train + {eval_split['images'].shape[0]} held-out scenes")
    print(f"Image: 3 x 32 x 32   Patch: {PATCH_SIZE} x {PATCH_SIZE}   Visual tokens per image: {N_PATCHES}")
    print(f"Vision encoder params: {venc.num_params():,}   Text encoder params: {tenc.num_params():,}")
    print(f"Language model params: {lm.num_params():,}   Projector params: {projector.num_params():,}")
    print(f"Visual prefix length K = {PREFIX_LEN}   Shared embedding width = 64   LLM width = {lm.cfg.n_embd}")
    print()

    out_dir = _THIS_DIR.parent / "output"

    # --- Sample grid (needs no student code) ---
    save_image_grid(train["images"], train["captions"], out_dir / "sample_scenes.png")
    print(f"Saved sample scene grid to {out_dir / 'sample_scenes.png'}")
    print()

    # --- Phase 1: CLIP alignment ---
    clip_ready = all(steps[s] for s in (
        "patchify", "pool_patches", "l2_normalize", "similarity_matrix", "clip_loss"))
    _heading("PHASE 1: CLIP-style image-text alignment")
    if clip_ready:
        acc0, _ = eval_retrieval(venc, tenc, eval_split, stoi, pad_id)
        print(f"  Held-out retrieval accuracy before training: {acc0:.1%}  (chance is ~1/64)")
        train_clip(venc, tenc, train, stoi, pad_id, rng)
        acc1, logits1 = eval_retrieval(venc, tenc, eval_split, stoi, pad_id)
        print(f"  Held-out retrieval accuracy after training:  {acc1:.1%}")
        save_retrieval_heatmap(logits1, eval_split["captions"], out_dir / "retrieval_heatmap.png")
        print(f"  Saved retrieval heatmap to {out_dir / 'retrieval_heatmap.png'}")
    else:
        need = [s for s in steps if not steps[s]]
        print(f"  [skipped: implement the vision + CLIP steps to align images and captions]")
        print(f"  [still missing: {', '.join(need)}]")
    print()

    # --- Phase 2: the bridge ---
    bridge_ready = clip_ready and all(steps[s] for s in (
        "image_to_prefix", "captioning_loss", "greedy_next_token"))
    _heading("PHASE 2: bridge the image into the language model")
    demo = [eval_split["images"][i] for i in range(3)]
    demo_caps = [eval_split["captions"][i] for i in range(3)]
    if bridge_ready:
        print("  Before bridge training (projector is random), 'describe the image' gives:")
        for img, cap in zip(demo, demo_caps):
            pred = generate_answer(lm, venc, projector, img, "describe the image", special, stoi, itos)
            print(f"    image[{cap!r}] -> {pred!r}")
        print()
        train_bridge(lm, venc, projector, train, special, stoi, rng)
        print()
        _heading("AFTER BRIDGE TRAINING: does the answer follow the image?")
        correct = 0
        for i in range(eval_split["images"].shape[0]):
            pred = generate_answer(lm, venc, projector, eval_split["images"][i],
                                   "describe the image", special, stoi, itos)
            correct += _answer_matches(pred, eval_split["captions"][i])
        print(f"  Held-out caption exact-match accuracy: {correct}/{eval_split['images'].shape[0]}"
              f" = {correct / eval_split['images'].shape[0]:.1%}")
        print()
        print("  Same prompt, different images (the grounding test):")
        for img, cap in zip(demo, demo_caps):
            pred = generate_answer(lm, venc, projector, img, "describe the image", special, stoi, itos)
            ok = "correct" if _answer_matches(pred, cap) else "wrong"
            print(f"    describe -> {pred!r}   (want {cap!r}: {ok})")
        print()
        print("  Grounded visual questions on one held-out image:")
        img0 = eval_split["images"][0]
        for q, a in questions_for(tuple(eval_split["top"][0]), tuple(eval_split["bottom"][0])):
            pred = generate_answer(lm, venc, projector, img0, q, special, stoi, itos)
            ok = "correct" if _answer_matches(pred, a) else "wrong"
            print(f"    {q:<26} -> {pred!r}   (want {a!r}: {ok})")
    else:
        print("  [skipped: finish Phase 1, then implement the three bridge steps]")
    print()

    _heading("Done")
    print("Run after each step; unfinished steps are skipped automatically.")


if __name__ == "__main__":
    main()
