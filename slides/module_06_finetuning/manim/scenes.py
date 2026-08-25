"""Module 6 Manim animations: chat-template serialization, loss masking, LoRA.

Three step-through scenes (each split into clips with self.next_section):
  ChatTemplateScene (slug chat-template): raw turns -> role markers -> one stream
  LossMaskScene     (slug loss-mask):     formatted row -> mask prompt -> loss
  LoRAScene         (slug lora):          frozen W + low-rank B A -> merge
"""

from manim import *

# ---------------------------------------------------------------------------
# High-quality H.264 encoding patch (copied from Module 4).
#
# Manim 0.20.1 hardcodes the partial-movie encode at CRF 23; on flat dark
# backgrounds with sharp text that leaves "mosquito" ringing around glyph edges.
# Section videos are stream-COPIED from the partial files, so improving the
# partial encode fixes every section. We drop CRF to 15 with the "slow" preset
# and keep yuv420p (yuv444p H.264 does not decode reliably in browsers).
from manim.scene import scene_file_writer as _sfw


def _hq_open_partial_movie_stream(self, file_path=None):
    if file_path is None:
        file_path = self.partial_movie_files[self.renderer.num_plays]
    self.partial_movie_file_path = file_path

    fps = _sfw.to_av_frame_rate(_sfw.config.frame_rate)

    partial_movie_file_codec = "libx264"
    partial_movie_file_pix_fmt = "yuv420p"
    av_options = {
        "an": "1",  # ffmpeg: -an, no audio
        "crf": "23",
    }

    if _sfw.config.movie_file_extension == ".webm":
        partial_movie_file_codec = "libvpx-vp9"
        av_options["-auto-alt-ref"] = "1"
        if _sfw.config.transparent:
            partial_movie_file_pix_fmt = "yuva420p"
    elif _sfw.config.transparent:
        partial_movie_file_codec = "qtrle"
        partial_movie_file_pix_fmt = "argb"
    else:
        # Default mp4/H.264 path: crank quality for crisp text edges.
        av_options["crf"] = "15"
        av_options["preset"] = "slow"

    video_container = _sfw.av.open(file_path, mode="w")
    stream = video_container.add_stream(
        partial_movie_file_codec,
        rate=fps,
        options=av_options,
    )
    stream.pix_fmt = partial_movie_file_pix_fmt
    stream.width = _sfw.config.pixel_width
    stream.height = _sfw.config.pixel_height

    self.video_container = video_container
    self.video_stream = stream

    self.queue = _sfw.Queue()
    self.writer_thread = _sfw.Thread(target=self.listen_and_write, args=())
    self.writer_thread.start()


_sfw.SceneFileWriter.open_partial_movie_stream = _hq_open_partial_movie_stream


# ---------------------------------------------------------------------------
# Palette (matches the other modules) and the crisp-kerning Text shadow.
# ---------------------------------------------------------------------------
BG = "#0a0e1a"
TEXT = "#e8eaf0"
MUTED = "#8892a4"
PRIMARY = "#4a9eff"
SECONDARY = "#f5a623"
GREEN = "#3fb950"
RED = "#f25555"
LINE = "#2a3450"

FONT = "Helvetica Neue"

_BaseText = Text
_KERN_REF = 96.0


class Text(_BaseText):  # noqa: F811 - intentional shadow of manim.Text
    """Render at a large reference size and scale the vector down: tight kerning."""

    def __init__(self, text, *args, font_size=48, **kwargs):
        if font_size < _KERN_REF:
            super().__init__(text, *args, font_size=_KERN_REF, **kwargs)
            self.scale(font_size / _KERN_REF)
        else:
            super().__init__(text, *args, font_size=font_size, **kwargs)


# ---------------------------------------------------------------------------
# Shared helpers (LaTeX-free: every label is a Text mobject).
# ---------------------------------------------------------------------------

def label(text: str, size: int = 28, color: str = TEXT, weight=NORMAL) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, weight=weight)


def cell(text: str, w: float, h: float, color: str = PRIMARY,
         fill: float = 0.14, size: int = 22, tcolor: str = TEXT, sw: float = 2.0) -> VGroup:
    box = RoundedRectangle(width=w, height=h, corner_radius=0.10,
                           stroke_color=color, fill_color=color,
                           fill_opacity=fill, stroke_width=sw)
    t = label(text, size, tcolor)
    if t.width > w - 0.18:
        t.scale((w - 0.18) / t.width)
    t.move_to(box)
    return VGroup(box, t)


def title_bar(text: str, sub: str | None = None) -> VGroup:
    t = label(text, 34, TEXT, weight=BOLD).to_edge(UP, buff=0.35)
    grp = VGroup(t)
    if sub:
        s = label(sub, 20, MUTED).next_to(t, DOWN, buff=0.12)
        grp.add(s)
    return grp


class StepScene(Scene):
    """Dark background plus a single managed caption pinned to the bottom edge."""

    def setup_bg(self):
        self.camera.background_color = BG
        self.cap = None

    def caption(self, text, color=TEXT, size=23):
        new = label(text, size, color).to_edge(DOWN, buff=0.5)
        if new.width > 12.6:
            new.scale(12.6 / new.width)
            new.to_edge(DOWN, buff=0.5)
        if self.cap is None:
            self.play(FadeIn(new), run_time=0.4)
        else:
            self.play(FadeOut(self.cap), FadeIn(new), run_time=0.4)
        self.cap = new
        return new


# ===========================================================================
# Scene 1: ChatTemplateScene
# ===========================================================================


class ChatTemplateScene(StepScene):
    """Two raw conversation turns become one flat token stream with role markers."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("The Chat Template", "a conversation serialized into one token stream"))

        # ---- raw conversation turns ----
        self.next_section("turns", skip_animations=False)
        user_role = label("user", 22, PRIMARY, weight=BOLD)
        user_txt = label("hi", 26, TEXT)
        user_box = RoundedRectangle(width=4.6, height=1.0, corner_radius=0.12,
                                    stroke_color=PRIMARY, fill_color=PRIMARY,
                                    fill_opacity=0.12, stroke_width=2.0)
        user_grp = VGroup(user_role, user_txt).arrange(RIGHT, buff=0.45)
        user_grp.move_to(user_box)
        user_turn = VGroup(user_box, user_grp).move_to([-2.0, 1.7, 0])

        asst_role = label("assistant", 22, SECONDARY, weight=BOLD)
        asst_txt = label("HI", 26, TEXT)
        asst_box = RoundedRectangle(width=4.6, height=1.0, corner_radius=0.12,
                                    stroke_color=SECONDARY, fill_color=SECONDARY,
                                    fill_opacity=0.12, stroke_width=2.0)
        asst_grp = VGroup(asst_role, asst_txt).arrange(RIGHT, buff=0.45)
        asst_grp.move_to(asst_box)
        asst_turn = VGroup(asst_box, asst_grp).move_to([2.0, 0.3, 0])

        self.play(FadeIn(user_turn, shift=RIGHT * 0.2), run_time=0.6)
        self.play(FadeIn(asst_turn, shift=LEFT * 0.2), run_time=0.6)
        self.caption("Two raw turns: the user's instruction and the assistant's reply.")

        # ---- wrap each turn with role markers ----
        self.next_section("markers", skip_animations=False)
        self.play(user_turn.animate.move_to([-3.4, 1.7, 0]).scale(0.9),
                  asst_turn.animate.move_to([-3.4, 0.2, 0]).scale(0.9), run_time=0.6)

        marker_u = cell("<|user|>", 1.95, 0.6, SECONDARY, size=18)
        marker_e1 = cell("<|end|>", 1.7, 0.6, SECONDARY, size=18)
        marker_a = cell("<|assistant|>", 2.35, 0.6, SECONDARY, size=18)
        marker_e2 = cell("<|end|>", 1.7, 0.6, SECONDARY, size=18)
        markers = VGroup(marker_u, marker_e1, marker_a, marker_e2).arrange(DOWN, buff=0.35)
        markers.move_to([3.0, 0.9, 0])
        mlab = label("special tokens (one id each)", 18, SECONDARY).next_to(markers, UP, buff=0.25)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in markers],
                              lag_ratio=0.15), FadeIn(mlab), run_time=0.9)
        self.caption("Role markers and an end-of-turn token wrap each turn. Each is its own token id.")

        # ---- flatten into one token stream ----
        self.next_section("flatten", skip_animations=False)
        self.play(FadeOut(user_turn), FadeOut(asst_turn), FadeOut(markers), FadeOut(mlab),
                  run_time=0.4)

        seq = ["<|user|>", "h", "i", "<|end|>", "<|assistant|>", "H", "I", "<|end|>"]
        is_special = [True, False, False, True, True, False, False, True]
        widths = [1.75 if sp else 0.62 for sp in is_special]
        row = VGroup()
        x = 0.0
        for w in widths:
            x += w / 2
            row.add(RoundedRectangle(width=w, height=0.7, corner_radius=0.08).move_to([x, 0, 0]))
            x += w / 2 + 0.16
        row.move_to([0, 0.4, 0])
        boxes = []
        for i, (tok, sp) in enumerate(zip(seq, is_special)):
            color = MUTED
            box = row[i]
            box.set_stroke(color, width=2.0)
            box.set_fill(color, opacity=0.10)
            t = label(tok, 16 if sp else 22, TEXT)
            if t.width > box.width - 0.14:
                t.scale((box.width - 0.14) / t.width)
            t.move_to(box)
            boxes.append(VGroup(box, t))
        stream = VGroup(*boxes)
        idxrow = VGroup(*[label(str(i), 15, MUTED).next_to(boxes[i], DOWN, buff=0.18)
                          for i in range(len(seq))])
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.08), run_time=1.0)
        self.play(FadeIn(idxrow), run_time=0.4)
        self.caption("Flattened into one sequence: the model only ever sees a flat list of token ids.")

        # ---- highlight the special tokens ----
        self.next_section("highlight", skip_animations=False)
        anims = []
        for i, sp in enumerate(is_special):
            color = SECONDARY if sp else PRIMARY
            anims.append(boxes[i][0].animate.set_stroke(color, width=2.6).set_fill(color, opacity=0.18))
            anims.append(boxes[i][1].animate.set_color(color if sp else TEXT))
        self.play(*anims, run_time=0.8)
        self.caption("Special tokens (orange) carry structure; text tokens (blue) carry content.")
        self.wait(0.3)


# ===========================================================================
# Scene 2: LossMaskScene
# ===========================================================================


class LossMaskScene(StepScene):
    """Each position predicts the next token; only response predictions feed the loss."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Loss Masking", "train only on the assistant's response"))

        seq = ["<|user|>", "h", "i", "<|end|>", "<|assistant|>", "H", "I", "<|end|>"]
        is_special = [True, False, False, True, True, False, False, True]
        # The token each position predicts is the next one; the last predicts nothing.
        nexts = seq[1:] + ["·"]
        # Keep the predictions whose target is a response token (positions 4,5,6).
        keep = [False, False, False, False, True, True, True, False]

        widths = [1.7 if sp else 0.6 for sp in is_special]

        # ---- the token row ----
        self.next_section("row", skip_animations=False)
        trow = VGroup()
        x = -6.0
        for i, (tok, sp) in enumerate(zip(seq, is_special)):
            w = widths[i]
            x += w / 2
            color = SECONDARY if sp else PRIMARY
            box = RoundedRectangle(width=w, height=0.7, corner_radius=0.08,
                                   stroke_color=color, fill_color=color,
                                   fill_opacity=0.14, stroke_width=2.0).move_to([x, 1.9, 0])
            t = label(tok, 15 if sp else 22, color if sp else TEXT)
            if t.width > w - 0.14:
                t.scale((w - 0.14) / t.width)
            t.move_to(box)
            trow.add(VGroup(box, t))
            x += w / 2 + 0.16
        self.play(LaggedStart(*[FadeIn(c) for c in trow], lag_ratio=0.07), run_time=0.9)
        self.caption("One formatted example: prompt tokens (orange/blue) then the response.")

        # ---- each position predicts the next token ----
        self.next_section("predict", skip_animations=False)
        tgt_cells, arrows = VGroup(), VGroup()
        for i in range(len(seq)):
            src = trow[i][0]
            tcell = RoundedRectangle(width=widths[i], height=0.62, corner_radius=0.08,
                                     stroke_color=MUTED, fill_color=MUTED,
                                     fill_opacity=0.10, stroke_width=1.8)
            tcell.next_to(src, DOWN, buff=0.7)
            lab = label(nexts[i], 15 if len(nexts[i]) > 1 else 22, TEXT)
            if lab.width > widths[i] - 0.14:
                lab.scale((widths[i] - 0.14) / lab.width)
            lab.move_to(tcell)
            tgt_cells.add(VGroup(tcell, lab))
            arrows.add(Arrow(src.get_bottom(), tcell.get_top(), buff=0.06,
                             color=MUTED, stroke_width=2.0, max_tip_length_to_length_ratio=0.4))
        plab = label("target = next token", 18, MUTED).next_to(tgt_cells, LEFT, buff=0.0)
        plab.to_edge(LEFT, buff=0.2).shift(DOWN * 0.05)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.06),
                  LaggedStart(*[FadeIn(c) for c in tgt_cells], lag_ratio=0.06), run_time=1.0)
        self.caption("Every position is trained to predict the next token (shift by one).")

        # ---- mask the prompt predictions, keep the response ----
        self.next_section("mask", skip_animations=False)
        mask_anims = []
        for i in range(len(seq)):
            if keep[i]:
                mask_anims.append(tgt_cells[i][0].animate.set_stroke(GREEN, width=2.6).set_fill(GREEN, opacity=0.20))
                mask_anims.append(tgt_cells[i][1].animate.set_color(GREEN))
            else:
                cross = label("-100", 16, RED).move_to(tgt_cells[i])
                mask_anims.append(Transform(tgt_cells[i][1], cross))
                mask_anims.append(tgt_cells[i][0].animate.set_stroke(RED, width=1.6).set_fill(RED, opacity=0.08))
                mask_anims.append(arrows[i].animate.set_opacity(0.2))
        self.play(*mask_anims, run_time=0.9)
        brace = Brace(VGroup(tgt_cells[4], tgt_cells[6]), DOWN, color=GREEN)
        btxt = label("response tokens R", 20, GREEN).next_to(brace, DOWN, buff=0.12)
        self.play(GrowFromCenter(brace), FadeIn(btxt), run_time=0.6)
        self.caption("Prompt predictions are set to -100 and ignored; only the response counts.")

        # ---- into the loss ----
        self.next_section("loss", skip_animations=False)
        loss = label("L  =  - (1/|R|)  [ log p(H) + log p(I) + log p(<|end|>) ]", 24, TEXT)
        loss.to_edge(DOWN, buff=1.15)
        self.play(FadeOut(self.cap), FadeIn(loss), run_time=0.6)
        self.cap = None
        self.caption("The loss averages over the response tokens only.")
        self.wait(0.3)


# ===========================================================================
# Scene 3: LoRAScene
# ===========================================================================


class LoRAScene(StepScene):
    """Freeze W, learn a low-rank B A correction, then merge it back in."""

    @staticmethod
    def grid(rows: int, cols: int, cs: float, color: str, fill: float = 0.16) -> VGroup:
        """A rows x cols matrix drawn as a filled rectangle with grid lines."""
        w, h = cols * cs, rows * cs
        outer = Rectangle(width=w, height=h, stroke_color=color, stroke_width=2.2,
                          fill_color=color, fill_opacity=fill)
        lines = VGroup()
        for c in range(1, cols):
            x = -w / 2 + c * cs
            lines.add(Line([x, -h / 2, 0], [x, h / 2, 0], stroke_color=color, stroke_width=0.8))
        for r in range(1, rows):
            y = -h / 2 + r * cs
            lines.add(Line([-w / 2, y, 0], [w / 2, y, 0], stroke_color=color, stroke_width=0.8))
        return VGroup(outer, lines)

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Low-Rank Adaptation (LoRA)", "freeze W, learn a small B A correction"))

        OUT, IN, R, cs = 6, 8, 2, 0.34

        # ---- the pretrained weight matrix ----
        self.next_section("weight", skip_animations=False)
        W = self.grid(OUT, IN, cs, PRIMARY).move_to([-3.6, 0.2, 0])
        wlab = label("W", 30, PRIMARY, weight=BOLD).next_to(W, UP, buff=0.2)
        wdim = label("d_out x d_in", 20, MUTED).next_to(W, DOWN, buff=0.2)
        self.play(FadeIn(W), FadeIn(wlab), FadeIn(wdim), run_time=0.7)
        self.caption("A pretrained weight matrix W of shape d_out x d_in. A 70B model has thousands of these.")

        # ---- freeze it ----
        self.next_section("freeze", skip_animations=False)
        frozen = label("frozen", 20, MUTED).move_to(W).set_z_index(5)
        fbox = SurroundingRectangle(W, color=MUTED, buff=0.0, stroke_width=2.0)
        self.play(W.animate.set_fill(MUTED, opacity=0.10).set_stroke(MUTED),
                  Create(fbox), run_time=0.6)
        self.play(FadeIn(frozen), run_time=0.3)
        self.caption("Full finetuning would update every entry of W. LoRA freezes it instead.")

        # ---- add the low-rank pair B and A ----
        self.next_section("lowrank", skip_animations=False)
        plus = label("+", 36, TEXT).move_to([-1.1, 0.2, 0])
        B = self.grid(OUT, R, cs, SECONDARY).move_to([0.2, 0.2, 0])
        A = self.grid(R, IN, cs, GREEN).move_to([2.4, 0.2, 0])
        blab = label("B", 26, SECONDARY, weight=BOLD).next_to(B, UP, buff=0.2)
        bdim = label("d_out x r", 18, MUTED).next_to(B, DOWN, buff=0.2)
        alab = label("A", 26, GREEN, weight=BOLD).next_to(A, UP, buff=0.2)
        adim = label("r x d_in", 18, MUTED).next_to(A, DOWN, buff=0.2)
        self.play(FadeIn(plus), run_time=0.2)
        self.play(FadeIn(B, shift=UP * 0.1), FadeIn(blab), FadeIn(bdim),
                  FadeIn(A, shift=UP * 0.1), FadeIn(alab), FadeIn(adim), run_time=0.8)
        self.caption("Learn two small matrices instead: B (d_out x r) and A (r x d_in), with r much smaller than either.")

        # ---- r << d, the parameter saving ----
        self.next_section("rank", skip_animations=False)
        rbox = SurroundingRectangle(VGroup(B, A), color=SECONDARY, buff=0.18, stroke_width=2.0)
        rnote = label("r << d_out, d_in", 24, SECONDARY, weight=BOLD).next_to(rbox, UP, buff=0.25).shift(RIGHT * 0.4)
        self.play(Create(rbox), FadeIn(rnote), run_time=0.5)
        counts = VGroup(
            label("full:  d_out x d_in  weights", 22, PRIMARY),
            label("LoRA:  r x (d_out + d_in)  weights", 22, SECONDARY),
            label("attn proj 4096 x 4096, r = 8:   16.8M  ->  65K", 21, GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -2.35, 0])
        self.play(FadeOut(self.cap), LaggedStart(*[FadeIn(c) for c in counts], lag_ratio=0.2),
                  run_time=1.0)
        self.cap = None
        self.wait(0.2)

        # ---- merge back ----
        self.next_section("merge", skip_animations=False)
        self.play(FadeOut(counts), FadeOut(rbox), FadeOut(rnote), FadeOut(plus),
                  FadeOut(blab), FadeOut(bdim), FadeOut(alab), FadeOut(adim), run_time=0.4)
        eq = label("W'  =  W  +  (alpha / r) B A", 26, TEXT).move_to([0, -2.2, 0])
        Wmerged = self.grid(OUT, IN, cs, PRIMARY).move_to(W)
        self.play(B.animate.move_to(W.get_center()).scale(0.4).set_opacity(0.0),
                  A.animate.move_to(W.get_center()).scale(0.4).set_opacity(0.0),
                  FadeOut(frozen), FadeOut(fbox), run_time=0.7)
        self.play(W.animate.set_fill(PRIMARY, opacity=0.16).set_stroke(PRIMARY),
                  FadeIn(eq), run_time=0.6)
        self.caption("At inference, fold B A back into W: same d_out x d_in matrix, zero extra latency.")
        self.wait(0.3)
