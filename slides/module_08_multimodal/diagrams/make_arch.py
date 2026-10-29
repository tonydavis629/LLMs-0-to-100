"""Module 8 architecture diagrams: one layout, many variants.

Every architecture slide in the deck uses the same picture: a modality lane
(input -> encoder -> connector) and a text lane (prompt -> token embedding)
feeding one language model. Each variant only changes what sits in those
slots and which pieces are trained or frozen, so students can see exactly
what differs between LLaVA, BLIP-2, Flamingo, early fusion, audio, and video.

Run from the repo root:  uv run python slides/module_08_multimodal/diagrams/make_arch.py
Writes SVG files into slides/module_08_multimodal/images/.
"""

from pathlib import Path

IMAGES = Path(__file__).resolve().parent.parent / "images"

# Palette shared with the deck (styles.css) and the Manim scenes.
BG = "#0a0e1a"
PANEL = "#141a28"
TEXT = "#e8eaf0"
MUTED = "#8fa0bd"
DIM = "#3a4560"
BLUE = "#4a9eff"
ORANGE = "#f5a623"
GREEN = "#50c878"
PURPLE = "#a06bd4"
RED = "#e05252"
LIGHT = "#c7d0e0"
FONT = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"

# Box status -> (stroke color, dash pattern, tag text)
STATUS = {
    "trained": (ORANGE, None, "trained"),
    "frozen": (MUTED, "6 4", "frozen"),
    "empty": (DIM, "4 4", None),
    "dim": (DIM, None, None),
    "plain": (BLUE, None, None),
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    """Tiny SVG builder: collects elements, then wraps them in a sized root."""

    def __init__(self, w: int, h: int, label: str):
        self.w, self.h, self.label = w, h, label
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=TEXT, anchor="middle", weight="normal", opacity=1.0):
        op = f' opacity="{opacity}"' if opacity < 1 else ""
        self.add(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{op}>{esc(s)}</text>')

    def rect(self, x, y, w, h, stroke, fill=PANEL, dash=None, sw=2, rx=8, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity < 1 else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{op}/>')

    def arrow(self, pts, color=MUTED, dash=None, sw=2, opacity=1.0, both=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity < 1 else ""
        start = ' marker-start="url(#ahs)"' if both else ""
        path = " ".join(f"{x},{y}" for x, y in pts)
        self.add(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{sw}"'
                 f'{d}{op}{start} marker-end="url(#ah)"/>')

    def save(self, name: str) -> None:
        defs = (
            '<defs>'
            f'<marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
            f'<path d="M0,0 L9,4.5 L0,9 Z" fill="{MUTED}"/></marker>'
            f'<marker id="ahs" markerWidth="9" markerHeight="9" refX="2" refY="4.5" orient="auto">'
            f'<path d="M9,0 L0,4.5 L9,9 Z" fill="{MUTED}"/></marker>'
            '</defs>'
        )
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" role="img" aria-label="{esc(self.label)}" '
               f'font-family="{FONT}">\n{defs}\n{body}\n</svg>\n')
        (IMAGES / name).write_text(svg)
        print("wrote", IMAGES / name)


# ---- shared pieces ---------------------------------------------------------


def box(svg: Svg, x, y, w, h, title, sub=(), status="plain", accent=None, focus=True):
    """A labeled component box. `status` picks stroke/dash/tag; `focus=False` dims it."""
    stroke, dash, tag = STATUS[status]
    if accent and status == "plain":
        stroke = accent
    op = 1.0 if focus else 0.35
    svg.rect(x, y, w, h, stroke, dash=dash, sw=3 if status == "trained" else 2, opacity=op)
    lines = [title, *sub]
    top = y + h / 2 - (len(lines) - 1) * 8 + 4
    tcolor = TEXT if status != "empty" else MUTED
    svg.text(x + w / 2, top, title, 14, tcolor, weight="600", opacity=op)
    for i, s in enumerate(sub):
        svg.text(x + w / 2, top + 17 * (i + 1), s, 11, MUTED, opacity=op)
    if tag:
        tw = 8 + 6.4 * len(tag)
        tx, ty = x + w - tw - 6, y - 9
        svg.rect(tx, ty, tw, 18, stroke, fill=BG, sw=1.5, rx=9, opacity=op)
        svg.text(tx + tw / 2, ty + 13, tag, 10.5, stroke, opacity=op)


def legend(svg: Svg, y, items=("trained", "frozen")):
    x = svg.w - 20
    for key in reversed(items):
        stroke, dash, tag = STATUS[key]
        label = {"trained": "trained in this stage", "frozen": "frozen (weights fixed)",
                 "empty": "slot not used"}[key]
        width = 30 + 6.6 * len(label)
        x -= width
        svg.rect(x, y - 11, 22, 14, stroke, dash=dash, sw=2, rx=3)
        svg.text(x + 28, y, label, 11.5, MUTED, anchor="start")
        x -= 24


# ---- end of helpers; diagram builders follow -------------------------------

# Slot geometry for the carried diagram (x, y, w, h).
MOD = (10, 40, 120, 84)
ENC = (200, 40, 170, 84)
CONN = (440, 40, 165, 84)
TXT = (10, 196, 120, 84)
EMB = (200, 196, 170, 84)
LLM = (690, 30, 175, 262)
OUT = (935, 126, 115, 70)


def mid_right(b):
    return b[0] + b[2], b[1] + b[3] / 2


def mid_left(b):
    return b[0], b[1] + b[3] / 2


def draw_llm(svg: Svg, title, sub, status, xattn=False, focus=True):
    """The language model: a box of stacked transformer blocks.

    With `xattn`, gated cross-attention layers (trained) sit between the
    frozen blocks, as in Flamingo. Returns the y of each cross-attn bar.
    """
    x, y, w, h = LLM
    stroke, dash, tag = STATUS[status]
    op = 1.0 if focus else 0.35
    svg.rect(x, y, w, h, stroke, dash=dash, sw=3 if status == "trained" else 2, opacity=op)
    svg.text(x + w / 2, y + 26, title, 14, TEXT, weight="600", opacity=op)
    for i, s in enumerate(sub):
        svg.text(x + w / 2, y + 44 + 15 * i, s, 11, MUTED, opacity=op)
    bars = ["block", "x-attn", "block", "x-attn", "block"] if xattn else ["block"] * 4
    top, bh, gap = y + 74, 26, 10 if not xattn else 7
    if xattn:
        bh = 24
    xbars = []
    for i, kind in enumerate(bars):
        by = top + i * (bh + gap)
        if kind == "x-attn":
            svg.rect(x + 16, by, w - 32, bh, ORANGE, fill=BG, sw=2, rx=5, opacity=op)
            svg.text(x + w / 2, by + bh / 2 + 4, "gated cross-attn", 10.5, ORANGE, opacity=op)
            xbars.append(by + bh / 2)
        else:
            svg.rect(x + 16, by, w - 32, bh, DIM, fill="#1b2436", sw=1.5, rx=5, opacity=op)
            svg.text(x + w / 2, by + bh / 2 + 4, "transformer block", 10.5, MUTED, opacity=op)
    if tag:
        tw = 8 + 6.4 * len(tag)
        tx, ty = x + w - tw - 6, y - 9
        svg.rect(tx, ty, tw, 18, stroke, fill=BG, sw=1.5, rx=9, opacity=op)
        svg.text(tx + tw / 2, ty + 13, tag, 10.5, stroke, opacity=op)
    return xbars


def arrow_label(svg: Svg, x, y, s, color=MUTED, opacity=1.0):
    """One- or two-line label above an arrow; '|' splits the lines."""
    lines = s.split("|")
    for i, line in enumerate(lines):
        svg.text(x, y - 12 * (len(lines) - 1 - i), line, 10.5, color, opacity=opacity)


def arch(name: str, label: str, mod, enc, conn, llm, out=("Text", ()),
         txt=("Text prompt", ("tokenizer",)), emb=("Token embedding", ("lookup table",)),
         labels=None, xattn=False, enc_to_emb=False, focus=None,
         legend_items=("trained", "frozen"), note=None):
    """Draw the carried diagram.

    mod/enc/conn: (title, subtitle-lines, status, accent) for the modality lane.
    llm: (title, subtitle-lines, status). `labels` maps arrow keys
    (raw, feat, vec, txt, emb) to the small text over each arrow.
    `enc_to_emb` routes the encoder's discrete ids into the shared embedding
    table (early fusion, codec audio) instead of through a connector.
    `focus` (a set of slot names) dims every other slot.
    """
    labels = labels or {}
    svg = Svg(1060, 335, label)
    f = (lambda k: True) if focus is None else (lambda k: k in focus)
    box(svg, *MOD, mod[0], mod[1], mod[2], mod[3], focus=f("mod"))
    box(svg, *ENC, enc[0], enc[1], enc[2], enc[3], focus=f("enc"))
    box(svg, *CONN, conn[0], conn[1], conn[2], conn[3], focus=f("conn"))
    box(svg, *TXT, txt[0], txt[1], "plain", TEXT, focus=f("txt"))
    box(svg, *EMB, emb[0], emb[1], "plain", TEXT, focus=f("emb"))
    xbars = draw_llm(svg, llm[0], llm[1], llm[2], xattn=xattn, focus=f("llm"))
    box(svg, *OUT, out[0], out[1], "plain", PURPLE, focus=f("out"))

    def op(*keys):
        return 1.0 if all(f(k) for k in keys) else 0.35

    # modality lane
    (x1, y1), (x2, _) = mid_right(MOD), mid_left(ENC)
    svg.arrow([(x1, y1), (x2 - 3, y1)], opacity=op("mod", "enc"))
    arrow_label(svg, (x1 + x2) / 2, y1 - 8, labels.get("raw", ""), opacity=op("mod", "enc"))
    if enc_to_emb:
        # discrete ids drop into the shared embedding table
        ex = ENC[0] + ENC[2] / 2
        svg.arrow([(ex, ENC[1] + ENC[3]), (ex, EMB[1] - 3)], color=ORANGE, opacity=op("enc", "emb"))
        svg.text(ex + 8, (ENC[1] + ENC[3] + EMB[1]) / 2 + 4, labels.get("ids", "ids"),
                 10.5, ORANGE, anchor="start", opacity=op("enc", "emb"))
    else:
        (x1, y1), (x2, _) = mid_right(ENC), mid_left(CONN)
        svg.arrow([(x1, y1), (x2 - 3, y1)], opacity=op("enc", "conn"))
        arrow_label(svg, (x1 + x2) / 2, y1 - 8, labels.get("feat", ""), opacity=op("enc", "conn"))
        (x1, y1) = mid_right(CONN)
        if xattn:
            # visual tokens feed the keys/values of every cross-attn layer
            bx = LLM[0] - 22
            svg.arrow([(x1, y1), (bx, y1), (bx, xbars[0]), (LLM[0] + 13, xbars[0])],
                      color=ORANGE, opacity=op("conn", "llm"))
            svg.arrow([(bx, xbars[0]), (bx, xbars[1]), (LLM[0] + 13, xbars[1])],
                      color=ORANGE, opacity=op("conn", "llm"))
            arrow_label(svg, (x1 + bx) / 2, y1 - 8, labels.get("vec", ""), opacity=op("conn", "llm"))
        else:
            svg.arrow([(x1, y1), (LLM[0] - 3, y1)], opacity=op("conn", "llm"))
            arrow_label(svg, (x1 + LLM[0]) / 2, y1 - 8, labels.get("vec", ""), opacity=op("conn", "llm"))

    # text lane
    (x1, y1), (x2, _) = mid_right(TXT), mid_left(EMB)
    svg.arrow([(x1, y1), (x2 - 3, y1)], opacity=op("txt", "emb"))
    arrow_label(svg, (x1 + x2) / 2, y1 - 8, labels.get("txt", "token ids"), opacity=op("txt", "emb"))
    (x1, y1) = mid_right(EMB)
    svg.arrow([(x1, y1), (LLM[0] - 3, y1)], opacity=op("emb", "llm"))
    arrow_label(svg, (x1 + LLM[0]) / 2, y1 - 8, labels.get("emb", "text vectors"), opacity=op("emb", "llm"))

    # output
    (x1, y1), (x2, _) = mid_right(LLM), mid_left(OUT)
    svg.arrow([(x1, y1), (x2 - 3, y1)], opacity=op("llm", "out"))
    arrow_label(svg, (x1 + x2) / 2, y1 - 8, labels.get("out", "next token"), opacity=op("llm", "out"))

    if note:
        svg.text(20, 322, note, 12, MUTED, anchor="start")
    if legend_items:
        legend(svg, 322, legend_items)
    svg.save(name)


# ---- end of arch(); variant definitions follow -----------------------------

IMG = ("Image", ("H x W x C",), "plain", BLUE)


def variants() -> None:
    arch("arch_text_only.svg", "The module template: a new modality lane (input, encoder, connector) beside the familiar text lane, both feeding the language model",
         mod=("Image / audio", ("new input",), "plain", BLUE),
         enc=("Encoder", ("new in this module",), "plain", LIGHT),
         conn=("Connector", ("new in this module",), "plain", GREEN),
         llm=("Language model", ("Modules 4-7",), "plain"),
         labels={"raw": "raw|signal", "feat": "features", "vec": "LLM-width|vectors"},
         legend_items=())

    arch("arch_generic.svg", "Image to vision encoder to connector to language model",
         mod=IMG,
         enc=("Vision encoder", ("pixels to features",), "plain", LIGHT),
         conn=("Connector", ("features to LLM width",), "plain", GREEN),
         llm=("Language model", ("attends over all vectors",), "plain"),
         labels={"raw": "raw|pixels", "feat": "patch|features", "vec": "LLM-width|vectors"},
         legend_items=())

    arch("arch_clip.svg", "CLIP pretrains the vision encoder slot",
         mod=IMG,
         enc=("Image encoder", ("CLIP / SigLIP", "contrastive pretraining"), "trained", None),
         conn=("Connector", ("next section",), "empty", None),
         llm=("Language model", ("next section",), "plain"),
         labels={"raw": "pixels", "feat": "features"},
         focus={"mod", "enc"}, legend_items=("trained",))

    arch("arch_llava.svg", "LLaVA: frozen CLIP encoder, trained MLP projector, trained LLM",
         mod=IMG,
         enc=("CLIP ViT-L/14", ("336 x 336 input",), "frozen", None),
         conn=("MLP projector", ("2 layers",), "trained", None),
         llm=("Vicuna LLM", ("visual tokens sit in", "the input sequence"), "trained"),
         labels={"raw": "pixels", "feat": "576 patch|features", "vec": "576 prefix|tokens"})

    arch("arch_blip2.svg", "BLIP-2: frozen encoder, trained Q-Former, frozen LLM",
         mod=IMG,
         enc=("EVA-CLIP ViT-g", ("257 patch features",), "frozen", None),
         conn=("Q-Former", ("32 learned queries", "+ linear layer"), "trained", None),
         llm=("OPT / Flan-T5", ("only 32 visual", "tokens enter"), "frozen"),
         labels={"raw": "pixels", "feat": "patch|features", "vec": "32 soft|tokens"})

    arch("arch_flamingo.svg", "Flamingo: frozen encoder, Perceiver Resampler, gated cross-attention inside a frozen LLM",
         mod=IMG,
         enc=("NFNet encoder", ("contrastively", "pretrained"), "frozen", None),
         conn=("Perceiver Resampler", ("64 learned latents",), "trained", None),
         llm=("Chinchilla LLM", ("frozen blocks, new", "cross-attn between"), "frozen"),
         xattn=True,
         labels={"raw": "pixels", "feat": "features", "vec": "64 visual|tokens"},
         note="Visual tokens never enter the text sequence; they are keys and values for the new layers.")

    arch("arch_fusion.svg", "Early fusion: image ids share the text vocabulary; one transformer trained from scratch",
         mod=IMG,
         enc=("VQ image tokenizer", ("512 x 512 to 1,024 ids", "codebook of 8,192"), "frozen", None),
         conn=("No connector", ("ids go straight into", "the embedding table"), "empty", None),
         emb=("Shared embedding", ("text + image ids",)),
         llm=("Chameleon", ("one model, trained", "on mixed tokens"), "trained"),
         out=("Text or image", ("tokens",)),
         enc_to_emb=True,
         labels={"raw": "pixels", "ids": "1,024 image ids", "emb": "one mixed|sequence"},
         legend_items=("trained", "frozen", "empty"))

    arch("arch_audio_enc.svg", "Audio as continuous vectors: spectrogram encoder plus projector",
         mod=("Audio", ("16 kHz waveform",), "plain", GREEN),
         enc=("Audio encoder", ("Whisper-style", "log-mel spectrogram in"), "frozen", None),
         conn=("Pool + projector", ("to LLM width",), "trained", None),
         llm=("Language model", ("audio vectors sit", "in the sequence"), "trained"),
         labels={"raw": "100 mel|frames/s", "feat": "50 vectors/s", "vec": "~25 vectors/s"})

    arch("arch_audio_codec.svg", "Audio as discrete tokens: a neural codec turns sound into codebook ids",
         mod=("Audio", ("24 kHz waveform",), "plain", GREEN),
         enc=("Neural codec", ("residual vector", "quantization"), "frozen", None),
         conn=("No connector", ("ids go straight into", "the embedding table"), "empty", None),
         emb=("Shared embedding", ("text + audio ids",)),
         llm=("Language model", ("reads and writes", "audio tokens"), "trained"),
         out=("Text or audio", ("codec decodes",)),
         enc_to_emb=True,
         labels={"raw": "samples", "ids": "codebook ids, e.g. 75 frames/s x 8", "emb": "one mixed|sequence"},
         legend_items=("trained", "frozen", "empty"))

    arch("arch_video.svg", "Video: sample frames, encode each, compress over time, project",
         mod=("Video", ("frames x H x W",), "plain", PURPLE),
         enc=("Frame sampler", ("+ image encoder", "per frame"), "frozen", None),
         conn=("Temporal pool", ("+ projector",), "trained", None),
         llm=("Language model", ("frame tokens with", "time positions"), "trained"),
         labels={"raw": "e.g. 1 frame/s", "feat": "tokens per|frame", "vec": "pooled|tokens"})


def callout(svg: Svg, x, y, n):
    """A numbered circle that ties a diagram part to the slide's list."""
    svg.add(f'<circle cx="{x}" cy="{y}" r="11" fill="{ORANGE}"/>')
    svg.text(x, y + 4.5, str(n), 12.5, BG, weight="700")


def full_stack() -> None:
    """The carried diagram with every slot filled, plus tools and the training loop."""
    svg = Svg(1060, 500, "Full multimodal stack: encoders, connectors, language model, tools, and the training loop")
    lanes = [("Image", BLUE, "Vision encoder"), ("Audio", GREEN, "Audio encoder / codec"),
             ("Video", PURPLE, "Frame encoder")]
    for i, (name, color, enc) in enumerate(lanes):
        y = 22 + i * 72
        box(svg, 10, y, 120, 56, name, (), "plain", color)
        box(svg, 200, y, 170, 56, enc, (), "plain", LIGHT)
        svg.arrow([(130, y + 28), (197, y + 28)])
        svg.arrow([(370, y + 28), (437, y + 28)])
    box(svg, 440, 22, 165, 200, "Connectors", ("projector, Q-Former,", "resampler, or a", "shared tokenizer"),
        "plain", GREEN)
    box(svg, 10, 252, 120, 64, "Text prompt", (), "plain", TEXT)
    box(svg, 200, 252, 170, 64, "Token embedding", (), "plain", TEXT)
    svg.arrow([(130, 284), (197, 284)])
    svg.arrow([(370, 284), (687, 284)])
    svg.arrow([(605, 122), (687, 122)])
    draw_llm(svg, "Language model", ("one sequence of", "text + other vectors"), "plain")
    box(svg, 935, 60, 115, 64, "Output", ("text, audio, or", "image tokens"), "plain", PURPLE)
    svg.arrow([(865, 92), (932, 92)])
    # tools: the model calls out and reads results back
    box(svg, 905, 188, 145, 104, "Tools", ("OCR, retrieval,", "code execution,", "image generation"), "plain", RED)
    svg.arrow([(865, 222), (902, 222)])
    svg.arrow([(905, 258), (868, 258)])
    svg.text(884, 214, "call", 10, MUTED)
    svg.text(884, 274, "result", 10, MUTED)

    # training loop band
    stages = ["Pretrain encoders + LLM", "Align modalities", "Multimodal SFT", "Preference / RL"]
    sx, sw, sy = 120, 185, 392
    svg.text(20, sy + 26, "Training", 13, MUTED, anchor="start", weight="600")
    for i, s in enumerate(stages):
        x = sx + i * (sw + 40)
        svg.rect(x, sy, sw, 44, BLUE, sw=2, rx=22)
        svg.text(x + sw / 2, sy + 27, s, 12.5, TEXT)
        if i:
            svg.arrow([(x - 40, sy + 22), (x - 3, sy + 22)])
    last = sx + 3 * (sw + 40) + sw
    svg.arrow([(last, sy + 22), (last + 22, sy + 22), (last + 22, sy + 78),
               (sx + 2 * (sw + 40) + sw / 2, sy + 78), (sx + 2 * (sw + 40) + sw / 2, sy + 47)],
              dash="5 4")
    svg.text((sx + 2 * (sw + 40) + last) / 2 + 40, sy + 94, "evaluate, collect new data and feedback, repeat",
             11, MUTED)
    svg.arrow([(777, sy - 4), (777, 296)], dash="5 4")
    svg.text(787, sy - 30, "updates the", 10.5, MUTED, anchor="start")
    svg.text(787, sy - 17, "weights", 10.5, MUTED, anchor="start")

    for n, (x, y) in enumerate([(200, 22), (440, 22), (690, 30), (120, 392), (905, 188)], 1):
        callout(svg, x, y, n)
    svg.save("arch_full_stack.svg")


def mini_strip(svg: Svg, x, y, states):
    """Three tiny boxes (encoder, projector, LLM) colored by training status."""
    names = ["Encoder", "Projector", "LLM"]
    w, gap = 64, 8
    for i, (name, st) in enumerate(zip(names, states)):
        bx = x + i * (w + gap)
        if st is None:
            svg.rect(bx, y, w, 38, DIM, dash="4 4", sw=1.5, rx=6)
            svg.text(bx + w / 2, y + 24, name, 10.5, DIM)
            continue
        stroke, dash, _ = STATUS[st]
        svg.rect(bx, y, w, 38, stroke, dash=dash, sw=2.5 if st == "trained" else 1.5, rx=6)
        svg.text(bx + w / 2, y + 24, name, 10.5, TEXT)


def train_stages() -> None:
    """The extra training stage a multimodal LLM adds after pretraining (LLaVA-1.5 recipe)."""
    svg = Svg(1060, 370, "Training stages of a multimodal LLM: pretrained parts, modality alignment, instruction tuning, preference tuning")
    cards = [
        ("0. Pretrained parts", ["trained", None, "trained"],
         ["Encoder: CLIP on image-caption pairs", "LLM: Modules 5-7 on text"], DIM),
        ("1. Modality alignment", ["frozen", "trained", "frozen"],
         ["558K image-caption pairs", "loss: next token of the caption", "only the projector learns"], ORANGE),
        ("2. Multimodal SFT", ["frozen", "trained", "trained"],
         ["665K image conversations,", "VQA, OCR, and region tasks", "masked loss on the answer"], BLUE),
        ("3. Preference / RL", ["frozen", "trained", "trained"],
         ["preferred vs rejected answers", "about the image (Module 7)"], BLUE),
    ]
    cw, gap, top = 240, 24, 70
    for i, (title, states, lines, accent) in enumerate(cards):
        x = 10 + i * (cw + gap)
        svg.rect(x, top, cw, 230, accent, sw=3 if accent == ORANGE else 1.5, rx=10)
        svg.text(x + cw / 2, top + 30, title, 15, TEXT, weight="600")
        mini_strip(svg, x + 12, top + 52, states)
        for j, line in enumerate(lines):
            svg.text(x + cw / 2, top + 130 + 20 * j, line, 12, MUTED)
        if i:
            svg.arrow([(x - gap + 2, top + 115), (x - 3, top + 115)])
    # brackets naming the two phases
    x1 = 10 + (cw + gap)
    svg.add(f'<path d="M{x1},{top - 14} L{x1},{top - 26} L{x1 + cw},{top - 26} L{x1 + cw},{top - 14}" '
            f'fill="none" stroke="{ORANGE}" stroke-width="2"/>')
    svg.text(x1 + cw / 2, top - 34, "new: teach the LLM to read the modality", 12.5, ORANGE, weight="600")
    x2 = 10 + 2 * (cw + gap)
    svg.add(f'<path d="M{x2},{top - 14} L{x2},{top - 26} L{x2 + 2 * cw + gap},{top - 26} '
            f'L{x2 + 2 * cw + gap},{top - 14}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    svg.text(x2 + cw + gap / 2, top - 34, "then: assistant behavior (next section)", 12.5, BLUE, weight="600")
    svg.text(10, 340, "Stage sizes from LLaVA-1.5 (Liu et al., 2023). Other models add or merge stages, "
             "but the alignment step is common.", 12, MUTED, anchor="start")
    legend(svg, 340)
    svg.save("train_stages.svg")


def patch_thumb(svg: Svg, x, y, size=56):
    """A 4 x 4 pixel patch drawn as flat cells (a stand-in for raw pixels)."""
    colors = ["#2f6db3", "#4a9eff", "#7fb8ff", "#2f6db3", "#4a9eff", "#c7d0e0", "#7fb8ff", "#4a9eff",
              "#7fb8ff", "#c7d0e0", "#f5a623", "#7fb8ff", "#2f6db3", "#7fb8ff", "#4a9eff", "#2f6db3"]
    c = size / 4
    for i, col in enumerate(colors):
        svg.add(f'<rect x="{x + (i % 4) * c}" y="{y + (i // 4) * c}" width="{c}" height="{c}" fill="{col}"/>')
    svg.add(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{LIGHT}" stroke-width="1.5"/>')


def table(svg: Svg, x, y, rows, hl, label, color):
    """A lookup table: stacked rows, one highlighted, with a caption below."""
    for r in range(rows):
        fill = color if r == hl else "#1b2436"
        svg.add(f'<rect x="{x}" y="{y + r * 11}" width="120" height="9" rx="2" fill="{fill}"/>')
    svg.text(x + 60, y + rows * 11 + 14, label, 10.5, MUTED)


def vector(svg: Svg, x, y, color):
    vals = [0.9, 0.3, 0.6, 0.15, 0.75, 0.45, 0.2, 0.65]
    for i, v in enumerate(vals):
        svg.add(f'<rect x="{x + i * 15}" y="{y}" width="13" height="30" fill="{color}" '
                f'fill-opacity="{0.25 + 0.7 * v:.2f}" stroke="{color}" stroke-width="1"/>')


def token_types() -> None:
    """Text token vs discrete image token vs continuous patch embedding."""
    svg = Svg(1060, 390, "Three things called tokens: text id lookup, VQ codebook lookup, and a projected patch with no lookup")
    rows = [(52, ("Text token",), TEXT), (172, ("Discrete image", "token (VQ)"), ORANGE),
            (292, ("Patch embedding", "(ViT)"), BLUE)]
    for y, names, color in rows:
        for i, name in enumerate(names):
            svg.text(14, y + 34 - 9 * (len(names) - 1) + 18 * i, name, 13.5, color, anchor="start", weight="600")
    svg.add('<g transform="translate(70,0)">')
    cols = [(150, "input"), (390, "discrete id"), (590, "table lookup"), (850, "d floats into the model")]
    for x, s in cols:
        svg.text(x, 22, s, 12, MUTED, weight="600")

    # row 1: word -> id -> embedding table -> vector
    y = 52
    svg.rect(110, y + 10, 80, 44, TEXT)
    svg.text(150, y + 38, '"cat"', 15, TEXT)
    svg.arrow([(190, y + 32), (347, y + 32)])
    svg.text(268, y + 24, "tokenizer", 10.5, MUTED)
    svg.rect(350, y + 12, 80, 40, TEXT, rx=20)
    svg.text(390, y + 37, "#4512", 14, TEXT, weight="600")
    svg.arrow([(430, y + 32), (527, y + 32)])
    table(svg, 530, y + 2, 5, 2, "embedding table: vocab x d", TEXT)
    svg.arrow([(650, y + 32), (787, y + 32)])
    vector(svg, 790, y + 17, TEXT)

    # row 2: patch -> nearest code -> codebook -> vector
    y = 172
    patch_thumb(svg, 122, y + 4)
    svg.arrow([(178, y + 32), (347, y + 32)])
    svg.text(262, y + 18, "encoder, then snap", 10.5, MUTED)
    svg.text(262, y + 26 + 4, "to the nearest code", 10.5, MUTED)
    svg.rect(350, y + 12, 80, 40, ORANGE, rx=20)
    svg.text(390, y + 37, "#317", 14, ORANGE, weight="600")
    svg.arrow([(430, y + 32), (527, y + 32)])
    table(svg, 530, y + 2, 5, 3, "learned codebook: K x d", ORANGE)
    svg.arrow([(650, y + 32), (787, y + 32)])
    vector(svg, 790, y + 17, ORANGE)

    # row 3: patch -> flatten -> linear projection -> vector (no id, no table)
    y = 292
    patch_thumb(svg, 122, y + 4)
    svg.arrow([(178, y + 32), (787, y + 32)])
    svg.rect(330, y + 6, 360, 52, BLUE, fill=BG, dash="6 4")
    svg.text(510, y + 27, "flatten 16 x 16 x 3 = 768 numbers, multiply by W", 12, TEXT)
    svg.text(510, y + 45, "no id, no table: any pixel values give a new vector", 11, MUTED)
    vector(svg, 790, y + 17, BLUE)

    svg.add("</g>")
    svg.save("token_types.svg")


def vqvae() -> None:
    """VQ-VAE: continuous encoder vectors snapped to the nearest codebook entry."""
    svg = Svg(1030, 300, "VQ-VAE: an encoder turns the image into a grid of continuous vectors, each is replaced by the id of its nearest codebook vector, and a decoder rebuilds the image from those codebook vectors")
    patch_thumb(svg, 20, 80, 96)
    svg.text(68, 200, "image", 12, MUTED)
    svg.arrow([(116, 128), (147, 128)])
    box(svg, 150, 88, 100, 80, "Encoder", ("neural net",), "plain", LIGHT)
    svg.arrow([(250, 128), (287, 128)])

    def grid(x0, y0, cell, color):
        for r in range(3):
            for c in range(3):
                svg.rect(x0 + c * (cell + 4), y0 + r * (cell + 4), cell, cell, color, fill=PANEL, sw=1.5, rx=4)

    # continuous vectors: each cell holds a small bar chart of d floats
    grid(290, 66, 40, BLUE)
    vals = [0.9, 0.3, 0.6, 0.2, 0.75, 0.45, 0.15, 0.8, 0.5]
    for r in range(3):
        for c in range(3):
            x, y = 290 + c * 44 + 7, 66 + r * 44 + 8
            for i in range(4):
                v = vals[(r * 3 + c + i) % 9]
                svg.add(f'<rect x="{x + i * 7}" y="{y}" width="5" height="24" fill="{BLUE}" '
                        f'fill-opacity="{0.2 + 0.75 * v:.2f}"/>')
    svg.text(354, 54, "continuous vectors", 12.5, BLUE, weight="600")
    svg.text(354, 218, "one d-dim vector per cell", 11, MUTED)

    # quantize: nearest codebook entry
    svg.arrow([(422, 128), (567, 128)])
    arrow_label(svg, 494, 120, "replace each vector|with its nearest code")
    table(svg, 434, 172, 6, 3, "codebook: K learned vectors", ORANGE)
    svg.add(f'<line x1="494" y1="132" x2="494" y2="168" stroke="{ORANGE}" stroke-width="1.5" stroke-dasharray="4 3"/>')

    # discrete ids
    grid(570, 66, 40, ORANGE)
    ids = ["317", "12", "4090", "7", "317", "88", "2051", "12", "930"]
    for i, k in enumerate(ids):
        r, c = divmod(i, 3)
        svg.text(570 + c * 44 + 20, 66 + r * 44 + 25, k, 12, TEXT, weight="600")
    svg.text(634, 54, "discrete ids", 12.5, ORANGE, weight="600")
    svg.text(634, 218, "what an LLM reads and writes", 11, ORANGE)

    svg.arrow([(702, 128), (773, 128)])
    arrow_label(svg, 737, 120, "look up")
    box(svg, 776, 88, 100, 80, "Decoder", ("neural net",), "plain", LIGHT)
    svg.arrow([(876, 128), (907, 128)])
    patch_thumb(svg, 910, 80, 96)
    svg.text(958, 200, "reconstruction", 12, MUTED)
    svg.text(20, 288, "Training: rebuild the image well, and keep encoder vectors close to their codes.",
             12, MUTED, anchor="start")
    svg.save("vqvae.svg")


def pope() -> None:
    """POPE-style probing: yes/no questions with random, popular, adversarial negatives."""
    svg = Svg(1000, 300, "POPE probing: the hardest absent objects are the ones that usually co-occur with what is present")
    # a schematic dining-table scene: table, plate, fork, cup
    svg.rect(20, 30, 300, 220, LIGHT, sw=1.5, rx=6)
    svg.add(f'<rect x="40" y="120" width="260" height="110" rx="6" fill="#5a4330"/>')
    svg.add(f'<ellipse cx="150" cy="170" rx="52" ry="30" fill="#e8eaf0"/>')
    svg.add(f'<ellipse cx="150" cy="170" rx="34" ry="18" fill="#c7d0e0"/>')
    svg.add(f'<rect x="222" y="138" width="5" height="62" rx="2" fill="#c7d0e0"/>')
    for dx in (-6, -2, 2, 6):
        svg.add(f'<rect x="{223 + dx}" y="128" width="2" height="14" fill="#c7d0e0"/>')
    svg.add(f'<rect x="250" y="146" width="30" height="36" rx="4" fill="{BLUE}"/>')
    svg.text(170, 62, "Image contains: dining table,", 12, MUTED)
    svg.text(170, 80, "plate, fork, cup", 12, MUTED)

    rows = [("Is there a fork in the image?", "present", "yes", GREEN),
            ("Is there a giraffe in the image?", "random absent object", "easy no", GREEN),
            ("Is there a person in the image?", "popular absent object", "harder", ORANGE),
            ("Is there a knife in the image?", "co-occurs with fork + plate", "hardest", RED)]
    for i, (q, kind, verdict, color) in enumerate(rows):
        y = 44 + i * 58
        svg.rect(360, y, 610, 44, color, sw=1.5, rx=8)
        svg.text(378, y + 27, q, 13.5, TEXT, anchor="start")
        svg.text(780, y + 27, kind, 11.5, MUTED, anchor="middle")
        svg.text(925, y + 27, verdict, 12.5, color, anchor="middle", weight="600")
    svg.text(665, 288, "Errors climb as the absent object gets more likely given the scene: a language prior, not vision.",
             11.5, MUTED)
    svg.save("pope_probe.svg")


def frame_sampling() -> None:
    """A to-scale 10 s timeline: 1 fps sampling misses an event that happens between samples."""
    svg = Svg(1000, 250, "Ten second timeline: one frame per second is sampled; the door opens between seconds 4 and 5")
    x0, x1, y = 60, 940, 120
    px = (x1 - x0) / 10  # pixels per second
    svg.add(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{MUTED}" stroke-width="2"/>')
    for k in range(301):  # every 30 fps frame, to scale
        x = x0 + k * px / 30
        svg.add(f'<line x1="{x:.1f}" y1="{y - 5}" x2="{x:.1f}" y2="{y + 5}" stroke="{DIM}" stroke-width="1"/>')
    for s in range(11):
        x = x0 + s * px
        svg.text(x, y + 30, f"{s} s", 11, MUTED)
    for s in range(10):  # sampled frames at 1 fps
        x = x0 + s * px
        svg.rect(x - 14, y - 58, 28, 36, BLUE, sw=2, rx=3)
        svg.add(f'<line x1="{x}" y1="{y - 22}" x2="{x}" y2="{y - 5}" stroke="{BLUE}" stroke-width="2"/>')
    # the event: door opens between 4.3 s and 4.8 s
    ea, eb = x0 + 4.3 * px, x0 + 4.8 * px
    svg.add(f'<rect x="{ea:.1f}" y="{y - 10}" width="{eb - ea:.1f}" height="20" fill="{RED}" fill-opacity="0.85"/>')
    svg.text((ea + eb) / 2, y + 54, "door opens (0.5 s)", 12, RED, weight="600")
    svg.text(x0 + 4 * px, y - 70, "door closed", 11, LIGHT)
    svg.text(x0 + 5 * px, y - 70, "door open", 11, LIGHT)
    svg.text(500, 30, "Sampled frames (1 fps): 10 of the 300 frames reach the model", 13, BLUE, weight="600")
    svg.text(500, 228, "The model sees the before and after, never the action. Who opened it, and how, is gone.",
             12, MUTED)
    svg.save("frame_sampling.svg")


def chart_probe() -> None:
    """A small bar chart with fine tick labels, for the perception-vs-reasoning side quest."""
    svg = Svg(420, 300, "Bar chart of quarterly revenue: Q1 42, Q2 47, Q3 61, Q4 58")
    x0, y0, h = 60, 250, 210
    vmax = 70
    svg.add(f'<line x1="{x0}" y1="{y0}" x2="{x0 + 330}" y2="{y0}" stroke="{MUTED}" stroke-width="1.5"/>')
    svg.add(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - h}" stroke="{MUTED}" stroke-width="1.5"/>')
    for v in range(0, 71, 10):
        y = y0 - h * v / vmax
        svg.add(f'<line x1="{x0 - 4}" y1="{y:.1f}" x2="{x0 + 330}" y2="{y:.1f}" stroke="{DIM}" stroke-width="0.8"/>')
        svg.text(x0 - 8, y + 3.5, str(v), 10, MUTED, anchor="end")
    for i, (q, v) in enumerate([("Q1", 42), ("Q2", 47), ("Q3", 61), ("Q4", 58)]):
        x = x0 + 22 + i * 78
        top = y0 - h * v / vmax
        svg.add(f'<rect x="{x}" y="{top:.1f}" width="50" height="{y0 - top:.1f}" fill="{BLUE}"/>')
        svg.text(x + 25, top - 6, str(v), 9, LIGHT)
        svg.text(x + 25, y0 + 16, q, 11.5, LIGHT)
    svg.text(x0 + 165, 22, "Revenue ($M)", 12.5, TEXT, weight="600")
    svg.save("chart_probe.svg")


if __name__ == "__main__":
    variants()
    full_stack()
    train_stages()
    token_types()
    vqvae()
    pope()
    frame_sampling()
    chart_probe()
