"""
Module 8 Exercise runner: Align image embeddings with NanoGPT

Run with:
    uv run python module_08_multimodal/src/main.py

Each step below runs one of the functions you write in exercise.py, then
tests it. Read top to bottom, the steps build a tiny vision-language model on
the bundled synthetic shapes dataset in three stages:

  Steps 1-2  Vision tower: turn a 32x32 image into one embedding.
  Steps 3-5  CLIP alignment: a contrastive loss pulls each image toward its
             caption; held-out retrieval accuracy climbs from chance.
  Steps 6-8  The bridge: project the image embedding into NanoGPT's width as
             visual prefix tokens, finetune so the language model captions the
             image, and check that the answer changes when the image changes.

Add --step N to run one step (1-8).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import copy
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (  # the eight steps you write
    captioning_loss,
    clip_loss,
    greedy_next_token,
    image_to_prefix,
    l2_normalize,
    patchify,
    pool_patches,
    similarity_matrix,
)
from src.chat import bridge_examples, chat_ids, prompt_ids, targets_and_mask
from src.data import build_dataset, load_dataset, questions_for, save_dataset
from src.grading import answer_matches, verdict
from src.model import load_instruct_model
from src.ops import (  # the small tensor operations between your steps
    add_position_embeddings,
    concat_visual_prefix,
    encode_text,
    flatten_patches,
    project_patches,
    retrieval_accuracy,
)
from src.prerequisites import require
from src.reporting import catch_up, run_step
from src.tokenizer import SPECIAL_TOKENS, decode, pad_captions
from src.vision import PATCH_SIZE, PREFIX_LEN, Projector, TextEncoder, VisionEncoder
from src.visualization import save_image_grid, save_retrieval_heatmap

# One test file per step lives in tests/
from tests.test_step1_patchify import check_patchify
from tests.test_step2_pool_patches import check_pool_patches
from tests.test_step3_l2_normalize import check_l2_normalize
from tests.test_step4_similarity_matrix import check_similarity_matrix
from tests.test_step5_clip_loss import check_clip_loss, check_clip_training
from tests.test_step6_image_to_prefix import check_image_to_prefix
from tests.test_step7_captioning_loss import check_bridge_training, check_captioning_loss
from tests.test_step8_greedy_next_token import check_greedy_next_token, check_grounded_captions

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

MODULE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = MODULE_DIR / "output"
DESCRIBE = "describe the image"  # the one prompt used for every caption


# ---------------------------------------------------------------------------
# Setup: the data, the language model, and the untrained towers
# ---------------------------------------------------------------------------


@cache  # build everything once, then every step shares the same objects
def setup() -> SimpleNamespace:
    """Load the bundled data and language model, and build the untrained towers."""
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)  # batch order for both training runs

    # data/instruct_model.pt is the Module 6 instruct checkpoint that Module 7
    # starts from; the vision tower gets bolted onto it here.
    lm, stoi, itos = load_instruct_model(DATA_DIR / "instruct_model.pt")
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}  # ids of <|user|>, <|end|>, ...

    # data/shapes_dataset.pt is the synthetic 340 train / 60 held-out split from src/data.py
    try:
        data = load_dataset(DATA_DIR / "shapes_dataset.pt")
    except FileNotFoundError:
        data = build_dataset()
        save_dataset(data, DATA_DIR / "shapes_dataset.pt")

    # The two towers and the projector start from random weights (seeded above)
    venc = VisionEncoder()
    tenc = TextEncoder(vocab_size=len(stoi), pad_id=special["<|pad|>"], max_len=TEXT_MAXLEN)
    projector = Projector(d_llm=lm.cfg.n_embd)

    # A picture of the dataset needs no student code
    save_image_grid(data["train"]["images"], data["train"]["captions"], OUTPUT_DIR / "sample_scenes.png")

    return SimpleNamespace(lm=lm, stoi=stoi, itos=itos, special=special, pad_id=special["<|pad|>"],
                           train=data["train"], eval=data["eval"], rng=rng,
                           venc=venc, tenc=tenc, projector=projector)


# ---------------------------------------------------------------------------
# The vision tower and CLIP training (Steps 1-5 working together)
# ---------------------------------------------------------------------------


def vision_forward(venc: VisionEncoder, images: torch.Tensor) -> torch.Tensor:
    """(B, 3, 32, 32) images -> (B, D_EMBED) pooled image embeddings."""
    patches = patchify(images, venc.patch_size)               # Step 1: (B, 16, 3, 8, 8)
    flat = flatten_patches(patches)                           # (B, 16, 192)
    x = project_patches(flat, venc.patch_proj)                # (B, 16, 64)
    x = add_position_embeddings(x, venc.pos_embed)            # where each patch sat
    x = venc.mix(x)                                           # patches share information
    return pool_patches(x)                                    # Step 2: (B, 64)


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
        cap_ids = pad_captions([captions[i] for i in idx.tolist()], stoi, pad_id, TEXT_MAXLEN)

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
    cap_ids = pad_captions(split["captions"], stoi, pad_id, TEXT_MAXLEN)
    img_embed = l2_normalize(vision_forward(venc, split["images"]))
    txt_embed = l2_normalize(encode_text(cap_ids, tenc))
    logits = similarity_matrix(img_embed, txt_embed, TEMPERATURE)
    return retrieval_accuracy(logits), logits


@cache  # train once; later steps reuse the trained towers
def clip_training():
    """CLIP-train both towers; return held-out accuracy before and after, and the final logits."""
    lab = setup()
    before, _ = eval_retrieval(lab.venc, lab.tenc, lab.eval, lab.stoi, lab.pad_id)
    train_clip(lab.venc, lab.tenc, lab.train, lab.stoi, lab.pad_id, lab.rng)
    after, logits = eval_retrieval(lab.venc, lab.tenc, lab.eval, lab.stoi, lab.pad_id)
    return before, after, logits


# ---------------------------------------------------------------------------
# The bridge into the language model (Steps 6-8 working together)
# ---------------------------------------------------------------------------


def bridge_example_loss(lm, venc, projector, image, prompt, response, special, stoi):
    """Image-conditioned captioning loss for one example (one forward through the LM)."""
    img_embed = vision_forward(venc, image.unsqueeze(0))                 # (1, D)
    prefix = image_to_prefix(img_embed, projector.to_prefix, PREFIX_LEN)  # Step 6: (1, K, d_llm)
    seq, resp_start = chat_ids(prompt, response, special, stoi)
    tok_embed = lm.embed_tokens(seq.unsqueeze(0))                        # (1, L, d_llm)
    inp = concat_visual_prefix(prefix, tok_embed)                       # (1, K+L, d_llm)
    logits = lm.forward_embeds(inp)[0]                                  # (K+L, V)
    targets, mask = targets_and_mask(seq, resp_start, PREFIX_LEN)
    return captioning_loss(logits, targets, mask)                       # Step 7


def train_bridge(lm, venc, projector, train, special, stoi, rng) -> list[float]:
    """Finetune the projector and the LM on image-conditioned examples; return the losses."""
    # The vision tower is frozen: CLIP already taught it what the shapes are
    for p in venc.parameters():
        p.requires_grad = False
    venc.eval()
    optimizer = torch.optim.AdamW(list(lm.parameters()) + list(projector.parameters()), lr=BRIDGE_LR)
    examples = bridge_examples(train, DESCRIBE)  # a caption and 4 questions per scene
    n = len(examples)
    lm.train(); projector.train()
    losses: list[float] = []
    for step in range(1, BRIDGE_STEPS + 1):
        idx = torch.randperm(n, generator=rng)[:BRIDGE_BATCH].tolist()
        batch_losses = []
        for j in idx:
            img, prompt, response = examples[j]
            batch_losses.append(bridge_example_loss(lm, venc, projector, img, prompt, response, special, stoi))
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


@cache  # train once; Step 8 reuses the trained bridge
def bridge_training():
    """Finetune the bridge; return the losses and untouched copies of the untrained LM and projector."""
    lab = setup()
    untrained = (copy.deepcopy(lab.lm), copy.deepcopy(lab.projector))  # for Step 8's before/after
    losses = train_bridge(lab.lm, lab.venc, lab.projector, lab.train, lab.special, lab.stoi, lab.rng)
    return losses, untrained


@torch.no_grad()
def generate_answer(lm, venc, projector, image, prompt, special, stoi, itos) -> str:
    """Greedily decode an image-conditioned answer to `prompt` for one image."""
    img_embed = vision_forward(venc, image.unsqueeze(0))
    prefix = image_to_prefix(img_embed, projector.to_prefix, PREFIX_LEN)
    seq = prompt_ids(prompt, special, stoi)
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
# Step 1: cut each image into patches
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        lab = setup()
        images = lab.train["images"]
        patches = patchify(images, PATCH_SIZE)
        print(f"Dataset: {images.shape[0]} training + {lab.eval['images'].shape[0]} held-out scenes; "
              "sample grid in output/sample_scenes.png")
        print(f"Images {tuple(images.shape)} -> patches {tuple(patches.shape)}")
        print(f"  each 32 x 32 image is now a sequence of {patches.shape[1]} patches of "
              f"{PATCH_SIZE} x {PATCH_SIZE} pixels")

    return run_step("Step 1: patchify()", show, lambda: check_patchify(patchify))


# ---------------------------------------------------------------------------
# Step 2: pool the patches into one embedding per image
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        require("2", needs="1", purpose="cut the images into patches")
        lab = setup()
        venc = lab.venc
        images = lab.train["images"]
        # The same stages as vision_forward(), printing the shape after each one
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


# ---------------------------------------------------------------------------
# Step 3: normalize every embedding to length 1
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        require("3", needs="12", purpose="embed the images it normalizes")
        lab = setup()
        with torch.no_grad():
            embeds = vision_forward(lab.venc, lab.train["images"])
            unit = l2_normalize(embeds)
        before, after = embeds.norm(dim=-1), unit.norm(dim=-1)
        print(f"Lengths of the {embeds.shape[0]} image embeddings:")
        print(f"  before normalizing: {float(before.min()):.3f} to {float(before.max()):.3f}")
        print(f"  after normalizing:  {float(after.min()):.3f} to {float(after.max()):.3f}")

    return run_step("Step 3: l2_normalize()", show, lambda: check_l2_normalize(l2_normalize))


# ---------------------------------------------------------------------------
# Step 4: score every image against every caption
# ---------------------------------------------------------------------------


def step_4() -> str:
    def show():
        require("4", needs="123", purpose="embed and normalize the held-out images")
        lab = setup()
        acc, logits = eval_retrieval(lab.venc, lab.tenc, lab.eval, lab.stoi, lab.pad_id)
        n = logits.shape[0]
        print(f"Similarity matrix of the {n} held-out images x {n} captions: shape {tuple(logits.shape)}")
        print(f"  Retrieval accuracy before training: {acc:.1%}  (chance is 1/{n})")

    return run_step("Step 4: similarity_matrix()", show, lambda: check_similarity_matrix(similarity_matrix))


# ---------------------------------------------------------------------------
# Step 5: CLIP training with the contrastive loss
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        require("5", needs="1234", purpose="run the contrastive training loop")
        print(f"CLIP training: {CLIP_STEPS} steps of {CLIP_BATCH} image-caption pairs")
        before, after, logits = clip_training()
        print(f"  Held-out retrieval accuracy: {before:.1%} before training, {after:.1%} after")
        save_retrieval_heatmap(logits, setup().eval["captions"], OUTPUT_DIR / "retrieval_heatmap.png")
        print("  Saved retrieval heatmap to output/retrieval_heatmap.png")

    def check():
        before, after, _ = clip_training()
        return check_clip_loss(clip_loss) + check_clip_training(before, after)

    return run_step("Step 5: clip_loss()", show, check)


def catch_up_clip() -> None:
    """Run Step 5's CLIP training quietly if a later step is run on its own."""
    catch_up(clip_training,
             lambda r: f"(ran the CLIP training from Step 5 first: held-out retrieval {r[1]:.1%})")


# ---------------------------------------------------------------------------
# Step 6: turn one image into visual prefix tokens for the language model
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        require("6", needs="12", purpose="embed the image it projects")
        lab = setup()
        with torch.no_grad():
            embed = vision_forward(lab.venc, lab.eval["images"][:1])
            prefix = image_to_prefix(embed, lab.projector.to_prefix, PREFIX_LEN)
            # The prompt as the language model sees it: <|user|> prompt <|end|> <|assistant|>
            tokens = lab.lm.embed_tokens(torch.tensor([prompt_ids(DESCRIBE, lab.special, lab.stoi)]))
            combined = concat_visual_prefix(prefix, tokens)
        print(f"Projector: Linear({embed.shape[-1]} -> {PREFIX_LEN} x {prefix.shape[-1]}), "
              f"{lab.projector.num_params():,} parameters")
        print(f"  image embedding {tuple(embed.shape)} -> visual prefix {tuple(prefix.shape)}")
        print(f"  prompt {DESCRIBE!r} as token embeddings {tuple(tokens.shape)}")
        print(f"  prefix + prompt, the language model's input {tuple(combined.shape)}")

    return run_step("Step 6: image_to_prefix()", show, lambda: check_image_to_prefix(image_to_prefix))


# ---------------------------------------------------------------------------
# Step 7: finetune the bridge with the captioning loss
# ---------------------------------------------------------------------------


def step_7() -> str:
    def show():
        require("7", needs="123456", purpose="train the bridge")
        catch_up_clip()  # does nothing when Step 5 already trained the towers
        print(f"Bridge training: {BRIDGE_STEPS} steps of {BRIDGE_BATCH} examples "
              "(a caption and 4 questions per scene)")
        bridge_training()

    def check():
        losses, _ = bridge_training()
        return check_captioning_loss(captioning_loss) + check_bridge_training(losses)

    return run_step("Step 7: captioning_loss()", show, check)


def catch_up_bridge() -> None:
    """Run Step 7's bridge training quietly if a later step is run on its own."""
    catch_up(bridge_training,
             lambda r: f"(ran the bridge training from Step 7 first: "
                       f"captioning loss {r[0][0]:.3f} -> {r[0][-1]:.3f})")


# ---------------------------------------------------------------------------
# Step 8: caption images and answer questions, one greedy token at a time
# ---------------------------------------------------------------------------


@cache  # caption once; the tests reuse the same captions
def held_out_captions() -> list[str]:
    """Caption every held-out scene with the trained bridge."""
    lab = setup()
    return [generate_answer(lab.lm, lab.venc, lab.projector, image, DESCRIBE, lab.special, lab.stoi, lab.itos)
            for image in lab.eval["images"]]


def step_8() -> str:
    def show():
        require("8", needs="1234567", purpose="train the bridge it decodes from")
        catch_up_clip()    # these do nothing when Steps 5 and 7 already ran
        catch_up_bridge()
        lab = setup()
        split = lab.eval
        demo = range(3)  # the first three held-out scenes

        def answer(lm, projector, i, prompt):
            return generate_answer(lm, lab.venc, projector, split["images"][i], prompt,
                                   lab.special, lab.stoi, lab.itos)

        _, (untrained_lm, untrained_projector) = bridge_training()
        print(f"Before bridge training (projector is random), {DESCRIBE!r} gives:")
        for i in demo:
            print(f"    image[{split['captions'][i]!r}] -> {answer(untrained_lm, untrained_projector, i, DESCRIBE)!r}")
        print()

        preds = held_out_captions()
        correct = sum(answer_matches(p, c) for p, c in zip(preds, split["captions"]))
        total = len(preds)
        print(f"After bridge training, held-out caption exact-match: {correct}/{total} = {correct / total:.1%}")
        print()

        print("Same prompt, different images (the grounding test):")
        for i in demo:
            print(f"    describe -> {preds[i]!r:<37} {verdict(preds[i], split['captions'][i])}")
        print()

        print("Grounded visual questions on one held-out image:")
        for q, a in questions_for(tuple(split["top"][0]), tuple(split["bottom"][0])):
            pred = answer(lab.lm, lab.projector, 0, q)
            print(f"    {q:<26} -> {pred!r:<10} {verdict(pred, a)}")

    def check():
        return (check_greedy_next_token(greedy_next_token)
                + check_grounded_captions(held_out_captions(), setup().eval["captions"]))

    return run_step("Step 8: greedy_next_token()", show, check)


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

    OUTPUT_DIR.mkdir(exist_ok=True)
    setup()  # load the data and models before the first step
    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
