"""Module 5 Manim animations: the pretraining loop, made visible.

Seven step-through scenes (each split into clips with self.next_section):
  next-token         shifted (input, target) pairs and one loss per position
  training-loop      forward -> loss -> backward -> update, loss descending
  sequence-packing   documents -> concat with EOS -> fixed-length blocks
  lr-schedule        warmup then cosine decay, traced by a moving dot
  scaling-laws       the power-law line, then the Chinchilla compute-optimal valley
  data-parallel      replicate the model, split the batch, all-reduce the gradients
  perplexity         cross-entropy -> perplexity (effective choices) -> bits per token

No LaTeX (every label is a Text mobject), no color gradients: matches the course style.
"""

from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# High-quality H.264 encoding patch (see module 4 notes): Manim 0.20.1 hardcodes
# the partial-movie encode at CRF 23, which leaves mosquito ringing around glyph
# edges on flat dark backgrounds. We drop to CRF 15 + preset slow on the default
# mp4/x264 path; section videos are stream-copied from these partials, so this
# fixes every clip. Keep yuv420p (yuv444p H.264 will not decode in Safari).
# ---------------------------------------------------------------------------
from manim.scene import scene_file_writer as _sfw


def _hq_open_partial_movie_stream(self, file_path=None):
    if file_path is None:
        file_path = self.partial_movie_files[self.renderer.num_plays]
    self.partial_movie_file_path = file_path

    fps = _sfw.to_av_frame_rate(_sfw.config.frame_rate)

    partial_movie_file_codec = "libx264"
    partial_movie_file_pix_fmt = "yuv420p"
    av_options = {"an": "1", "crf": "23"}

    if _sfw.config.movie_file_extension == ".webm":
        partial_movie_file_codec = "libvpx-vp9"
        av_options["-auto-alt-ref"] = "1"
        if _sfw.config.transparent:
            partial_movie_file_pix_fmt = "yuva420p"
    elif _sfw.config.transparent:
        partial_movie_file_codec = "qtrle"
        partial_movie_file_pix_fmt = "argb"
    else:
        av_options["crf"] = "15"
        av_options["preset"] = "slow"

    video_container = _sfw.av.open(file_path, mode="w")
    stream = video_container.add_stream(partial_movie_file_codec, rate=fps, options=av_options)
    stream.pix_fmt = partial_movie_file_pix_fmt
    stream.width = _sfw.config.pixel_width
    stream.height = _sfw.config.pixel_height

    self.video_container = video_container
    self.video_stream = stream

    self.queue = _sfw.Queue()
    self.writer_thread = _sfw.Thread(target=self.listen_and_write, args=())
    self.writer_thread.start()


_sfw.SceneFileWriter.open_partial_movie_stream = _hq_open_partial_movie_stream


BG = "#0a0e1a"
TEXT = "#e8eaf0"
MUTED = "#8892a4"
PRIMARY = "#4a9eff"
SECONDARY = "#f5a623"
GREEN = "#3fb950"
RED = "#e0533d"
LINE = "#2a3450"

FONT = "Helvetica Neue"


# ---------------------------------------------------------------------------
# Crisp-kerning Text: render at a large reference size and scale the vector down
# so inter-glyph advances stay tight at any display size (see module 4 notes).
# ---------------------------------------------------------------------------
_BaseText = Text
_KERN_REF = 96.0


class Text(_BaseText):  # noqa: F811 - intentional shadow of manim.Text
    def __init__(self, text, *args, font_size=48, **kwargs):
        if font_size < _KERN_REF:
            super().__init__(text, *args, font_size=_KERN_REF, **kwargs)
            self.scale(font_size / _KERN_REF)
        else:
            super().__init__(text, *args, font_size=font_size, **kwargs)


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


def sq(s: float, color: str = PRIMARY, fill: float = 0.5) -> Square:
    return Square(s, stroke_width=1.4, stroke_color=color, fill_color=color, fill_opacity=fill)


def title_bar(text: str, sub: str | None = None) -> VGroup:
    t = label(text, 34, TEXT, weight=BOLD).to_edge(UP, buff=0.35)
    grp = VGroup(t)
    if sub:
        s = label(sub, 20, MUTED).next_to(t, DOWN, buff=0.12)
        grp.add(s)
    return grp


def polyline(points, color=PRIMARY, width=4) -> VMobject:
    """A smooth-enough polyline through 3-D points (np arrays)."""
    vm = VMobject(stroke_color=color, stroke_width=width)
    vm.set_points_as_corners([np.array(p, dtype=float) for p in points])
    return vm


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


class NextTokenScene(StepScene):
    """The causal-LM objective: inputs and targets are the same text shifted by one."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Next-Token Prediction", "every position is a training example"))

        toks = ["The", "cat", "sat", "on", "the"]
        nexts = ["cat", "sat", "on", "the", "mat"]  # the token that follows each one
        cw, ch, buff = 1.32, 0.62, 0.30

        def row(words, color, y):
            g = VGroup(*[cell(w, cw, ch, color, size=24) for w in words]).arrange(RIGHT, buff=buff)
            g.move_to([0, y, 0])
            return g

        # ---- the sequence ----
        self.next_section("sequence", skip_animations=False)
        xrow = row(toks, PRIMARY, 2.25)
        xlab = label("input  x", 20, PRIMARY).next_to(xrow, LEFT, buff=0.4)
        self.play(LaggedStart(*[FadeIn(c) for c in xrow], lag_ratio=0.12), FadeIn(xlab), run_time=0.9)
        self.caption("Pretraining data is just a sequence of tokens.")

        # ---- shift by one to make the targets ----
        self.next_section("shift", skip_animations=False)
        yrow = row(nexts, SECONDARY, 1.30)
        ylab = label("target  y", 20, SECONDARY).next_to(yrow, LEFT, buff=0.4)
        arrows = VGroup(*[
            Arrow(xrow[i].get_bottom(), yrow[i].get_top(), buff=0.04, color=MUTED,
                  stroke_width=2.5, max_tip_length_to_length_ratio=0.5)
            for i in range(len(toks))
        ])
        self.play(FadeIn(ylab), run_time=0.3)
        # The first four targets are copies of x shifted left by one position.
        self.play(*[TransformFromCopy(xrow[i + 1], yrow[i]) for i in range(len(toks) - 1)],
                  run_time=0.9)
        # The last target is the token that continues the text.
        self.play(FadeIn(yrow[-1], shift=DOWN * 0.2), run_time=0.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1), run_time=0.7)
        self.caption("The label at each position is simply the next token: shift the sequence by one.")

        # ---- one position predicts a distribution ----
        self.next_section("predict", skip_animations=False)
        focus = 2  # the "sat" column predicts "on"
        dim = [0, 1, 3, 4]
        self.play(*[xrow[i].animate.set_opacity(0.30) for i in dim],
                  *[yrow[i].animate.set_opacity(0.30) for i in dim],
                  *[arrows[i].animate.set_opacity(0.25) for i in dim], run_time=0.5)
        col_hl = SurroundingRectangle(VGroup(xrow[focus], yrow[focus]), color=SECONDARY,
                                      buff=0.10, stroke_width=2.5)
        self.play(Create(col_hl), run_time=0.4)

        model = cell("model", 1.7, 0.7, PRIMARY, size=22).move_to([-3.4, -1.1, 0])
        feed = Arrow(xrow[focus].get_corner(DOWN), model.get_top() + RIGHT * 0.2,
                     buff=0.1, color=MUTED, stroke_width=2.5)
        cand = ["on", "in", "a", "the", "by"]
        probs = [0.62, 0.14, 0.10, 0.08, 0.06]
        bars = VGroup()
        blabs = VGroup()
        bx0, bdx = -0.7, 0.95
        for i, (c, p) in enumerate(zip(cand, probs)):
            col = SECONDARY if i == 0 else PRIMARY
            h = max(p * 3.2, 0.06)
            b = Rectangle(width=0.62, height=h, stroke_width=1.4, stroke_color=col,
                          fill_color=col, fill_opacity=0.6).move_to([bx0 + i * bdx, -2.0 + h / 2, 0])
            bars.add(b)
            blabs.add(label(c, 17, col if i == 0 else MUTED).move_to([bx0 + i * bdx, -2.35, 0]))
        toarr = Arrow(model.get_right(), bars.get_left() + LEFT * 0.1, buff=0.12,
                      color=MUTED, stroke_width=2.5)
        self.play(GrowArrow(feed), FadeIn(model), run_time=0.5)
        self.play(GrowArrow(toarr), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08),
                  FadeIn(blabs), run_time=0.9)
        self.caption("From the prefix, the model predicts a probability for every possible next token.")

        # ---- the loss at that position ----
        self.next_section("loss", skip_animations=False)
        truth = label("true next token: on", 20, SECONDARY).move_to([3.6, -1.0, 0])
        lossv = label("loss = -log p(on) = 0.48", 22, TEXT).move_to([3.6, -1.7, 0])
        self.play(FadeIn(truth), run_time=0.3)
        self.play(Indicate(bars[0], color=SECONDARY, scale_factor=1.15), FadeIn(lossv), run_time=0.7)
        self.caption("Cross-entropy rewards probability on the true token: one loss term per position.")
        self.wait(0.3)


class TrainingLoopScene(StepScene):
    """The optimization loop: forward, loss, backward, update, repeat; loss descends."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("The Training Loop"))

        F = cell("Forward", 2.0, 0.8, PRIMARY, size=22).move_to([0, 1.85, 0])
        L = cell("Loss", 2.0, 0.8, SECONDARY, size=22).move_to([3.8, 0.15, 0])
        B = cell("Backward", 2.0, 0.8, SECONDARY, size=22).move_to([0, -1.55, 0])
        U = cell("Update", 2.0, 0.8, GREEN, size=22).move_to([-3.8, 0.15, 0])
        nodes = VGroup(F, L, B, U)

        a_fl = CurvedArrow(F.get_right(), L.get_top(), angle=-0.9, color=PRIMARY, stroke_width=3, tip_length=0.2)
        a_lb = CurvedArrow(L.get_bottom(), B.get_right(), angle=-0.9, color=SECONDARY, stroke_width=3, tip_length=0.2)
        a_bu = CurvedArrow(B.get_left(), U.get_bottom(), angle=-0.9, color=SECONDARY, stroke_width=3, tip_length=0.2)
        a_uf = CurvedArrow(U.get_top(), F.get_left(), angle=-0.9, color=GREEN, stroke_width=3, tip_length=0.2)
        ring = VGroup(a_fl, a_lb, a_bu, a_uf)

        cap = label("training loss", 18, MUTED).move_to([0, 0.62, 0])
        num = label("--", 40, TEXT).move_to([0, -0.05, 0])

        # ---- the loop ----
        self.next_section("setup", skip_animations=False)
        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.1), run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in ring], lag_ratio=0.15), run_time=1.0)
        self.play(FadeIn(cap), FadeIn(num), run_time=0.4)
        self.caption("Pretraining repeats one loop, millions of times.")

        # ---- forward ----
        self.next_section("forward", skip_animations=False)
        batch = VGroup(*[sq(0.22, PRIMARY, 0.5) for _ in range(3)]).arrange(DOWN, buff=0.05)
        batch.next_to(F, UP, buff=0.35)
        blab = label("token batch", 16, MUTED).next_to(batch, RIGHT, buff=0.18)
        self.play(FadeIn(batch), FadeIn(blab), run_time=0.4)
        dot = batch.copy()
        self.play(dot.animate.move_to(F.get_center()).scale(0.6), run_time=0.4)
        self.play(dot.animate.move_to(L.get_center()), a_fl.animate.set_stroke(width=5), run_time=0.5)
        num4 = label("4.18", 40, SECONDARY).move_to(num)
        self.play(Transform(num, num4), FadeOut(dot), run_time=0.4)
        self.caption("Forward pass: run the batch through the model, then measure the loss.")

        # ---- backward ----
        self.next_section("backward", skip_animations=False)
        grad = Dot(color=SECONDARY, radius=0.13).move_to(L.get_center())
        self.play(a_fl.animate.set_stroke(width=3), FadeIn(grad, scale=1.4), run_time=0.3)
        self.play(MoveAlongPath(grad, a_lb), a_lb.animate.set_stroke(width=5), run_time=0.6)
        self.play(MoveAlongPath(grad, a_bu), a_bu.animate.set_stroke(width=5), run_time=0.6)
        self.caption("Backward pass: backpropagate the loss into a gradient for every weight.")

        # ---- update ----
        self.next_section("update", skip_animations=False)
        grid = VGroup(*[sq(0.20, GREEN, 0.35) for _ in range(9)]).arrange_in_grid(3, 3, buff=0.05)
        grid.next_to(U, DOWN, buff=0.3)
        gl = label("weights", 15, MUTED).next_to(grid, DOWN, buff=0.12)
        self.play(FadeOut(grad), FadeIn(grid), FadeIn(gl), run_time=0.4)
        self.play(Indicate(U, color=GREEN, scale_factor=1.08),
                  *[s.animate.set_fill(opacity=np.random.uniform(0.2, 0.7)) for s in grid],
                  a_bu.animate.set_stroke(width=3), run_time=0.5)
        num340 = label("3.40", 40, SECONDARY).move_to(num)
        self.play(a_uf.animate.set_stroke(width=5), Transform(num, num340), run_time=0.6)
        self.play(a_uf.animate.set_stroke(width=3), run_time=0.2)
        self.caption("Optimizer step: nudge every weight to lower the loss. Then repeat.")

        # ---- descend ----
        self.next_section("descend", skip_animations=False)
        spin = Dot(color=TEXT, radius=0.10)
        loop = Ellipse(width=7.6, height=3.4).move_to([0, 0.15, 0])
        spin.move_to(loop.point_from_proportion(0))
        self.play(FadeIn(spin), run_time=0.2)
        for val in ["2.29", "1.83", "1.64"]:
            newn = label(val, 40, SECONDARY).move_to(num)
            self.play(MoveAlongPath(spin, loop),
                      LaggedStart(*[a.animate.set_stroke(width=4.5) for a in ring], lag_ratio=0.05),
                      run_time=0.7, rate_func=linear)
            self.play(Transform(num, newn),
                      *[a.animate.set_stroke(width=3) for a in ring], run_time=0.3)
        self.play(FadeOut(spin), Indicate(num, color=GREEN, scale_factor=1.2), run_time=0.5)
        self.caption("Over many steps the loss falls: random weights become useful weights.")
        self.wait(0.3)


class SequencePackingScene(StepScene):
    """Documents of different lengths become one stream, chopped into fixed blocks."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Packing the Data", "documents to fixed-length training blocks"))

        side, p = 0.30, 0.345

        def cells(n, color, y, x_left):
            g = VGroup(*[sq(side, color, 0.55) for _ in range(n)]).arrange(RIGHT, buff=p - side)
            g.move_to([x_left + g.width / 2, y, 0])
            return g

        # ---- three documents ----
        self.next_section("docs", skip_animations=False)
        docA = cells(8, PRIMARY, 1.6, -3.9)
        docB = cells(4, GREEN, 0.7, -3.9)
        docC = cells(10, SECONDARY, -0.2, -3.9)
        labA = label("doc 1", 16, PRIMARY).next_to(docA, LEFT, buff=0.3)
        labB = label("doc 2", 16, GREEN).next_to(docB, LEFT, buff=0.3)
        labC = label("doc 3", 16, SECONDARY).next_to(docC, LEFT, buff=0.3)
        note = label("all different lengths", 18, MUTED).move_to([3.4, 0.7, 0])
        self.play(LaggedStart(FadeIn(VGroup(labA, docA)), FadeIn(VGroup(labB, docB)),
                              FadeIn(VGroup(labC, docC)), lag_ratio=0.3), run_time=1.0)
        self.play(FadeIn(note), run_time=0.3)
        self.caption("Raw documents have all different lengths.")

        # ---- concatenate into one stream with EOS separators ----
        self.next_section("concat", skip_animations=False)
        def slotx(i):
            return (i - 11.5) * p
        slots = [None] * 24
        # A -> slots 0..7 ; EOS 8 ; B -> 9..12 ; EOS 13 ; C -> 14..23
        a_slots = list(range(0, 8))
        b_slots = list(range(9, 13))
        c_slots = list(range(14, 24))
        for s, i in zip(docA, a_slots):
            slots[i] = s
        for s, i in zip(docB, b_slots):
            slots[i] = s
        for s, i in zip(docC, c_slots):
            slots[i] = s
        eos1 = sq(side, RED, 0.6).move_to([slotx(8), 0.9, 0])
        eos2 = sq(side, RED, 0.6).move_to([slotx(13), 0.9, 0])
        eos1d = label("E", 14, TEXT).move_to(eos1)
        eos2d = label("E", 14, TEXT).move_to(eos2)
        slots[8], slots[13] = VGroup(eos1, eos1d), VGroup(eos2, eos2d)
        self.play(
            FadeOut(note), FadeOut(labA), FadeOut(labB), FadeOut(labC),
            *[s.animate.move_to([slotx(i), 0.9, 0]) for s, i in zip(docA, a_slots)],
            *[s.animate.move_to([slotx(i), 0.9, 0]) for s, i in zip(docB, b_slots)],
            *[s.animate.move_to([slotx(i), 0.9, 0]) for s, i in zip(docC, c_slots)],
            run_time=1.0,
        )
        self.play(FadeIn(VGroup(eos1, eos1d, eos2, eos2d)), run_time=0.5)
        eoslab = label("E = end-of-text token marks each boundary", 18, RED).move_to([0, 1.7, 0])
        self.play(FadeIn(eoslab), run_time=0.4)
        self.caption("Concatenate into one long token stream, marking boundaries with an EOS token.")

        # ---- chop into fixed-length blocks ----
        self.next_section("chop", skip_animations=False)
        blocks = [VGroup(*slots[k * 6:(k + 1) * 6]) for k in range(4)]
        cuts = VGroup(*[
            DashedLine([(slotx(c) + slotx(c + 1)) / 2, 1.45, 0],
                       [(slotx(c) + slotx(c + 1)) / 2, 0.35, 0],
                       color=TEXT, stroke_width=2.5, dash_length=0.08)
            for c in (5, 11, 17)
        ])
        self.play(FadeOut(eoslab), Create(cuts), run_time=0.7)
        shifts = [-0.45, -0.15, 0.15, 0.45]
        self.play(*[blocks[k].animate.shift(RIGHT * shifts[k]) for k in range(4)],
                  FadeOut(cuts), run_time=0.6)
        self.caption("Chop the stream into fixed-length blocks (here, 6 tokens each).")

        # ---- stack into a batch tensor ----
        self.next_section("batch", skip_animations=False)
        targets = []
        for k in range(4):
            row_y = 0.9 - k * (side + 0.08)
            for j in range(6):
                targets.append(([(j - 2.5) * (side + 0.06) - 1.0, row_y, 0], blocks[k][j]))
        self.play(*[m.animate.move_to(pos) for pos, m in targets], run_time=1.0)
        grid_group = VGroup(*[m for _, m in targets])
        brace = SurroundingRectangle(grid_group, color=PRIMARY, buff=0.14, stroke_width=2)
        blab = label("one batch\n(4 x 6)", 20, PRIMARY).next_to(brace, RIGHT, buff=0.5)
        self.play(Create(brace), FadeIn(blab), run_time=0.6)
        self.caption("Fixed (batch x block) tensors: the regular shapes the GPU trains on.")
        self.wait(0.3)


class LRScheduleScene(StepScene):
    """Learning rate over training: a linear warmup, then a cosine decay."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Learning-Rate Schedule"))

        x0, x1 = -5.3, 5.3
        yb, yt = -2.3, 2.05
        wf, minf = 0.08, 0.1  # warmup fraction, min-lr fraction of peak

        def X(t):
            return x0 + t * (x1 - x0)

        def Y(lr):
            return yb + lr * (yt - yb)

        def lr_of(t):
            if t < wf:
                return t / wf
            d = (t - wf) / (1 - wf)
            return minf + 0.5 * (1 + np.cos(np.pi * d)) * (1 - minf)

        xaxis = Line([x0, yb, 0], [x1 + 0.2, yb, 0], color=MUTED, stroke_width=2)
        yaxis = Line([x0, yb, 0], [x0, yt + 0.35, 0], color=MUTED, stroke_width=2)
        xlab = label("training step", 18, MUTED).next_to(xaxis, DOWN, buff=0.15).align_to(xaxis, RIGHT)
        ylab = label("learning rate", 18, MUTED).rotate(PI / 2).next_to(yaxis, LEFT, buff=0.15)

        # ---- axes ----
        self.next_section("axes", skip_animations=False)
        self.play(Create(xaxis), Create(yaxis), FadeIn(xlab), FadeIn(ylab), run_time=0.8)
        self.caption("The learning rate is not constant; it follows a schedule.")

        # ---- warmup ----
        self.next_section("warmup", skip_animations=False)
        wpts = [[X(t), Y(lr_of(t)), 0] for t in np.linspace(0, wf, 24)]
        wpath = polyline(wpts, PRIMARY, 4)
        shade = Rectangle(width=X(wf) - x0, height=yt - yb, stroke_width=0,
                          fill_color=PRIMARY, fill_opacity=0.07)
        shade.move_to([(x0 + X(wf)) / 2, (yb + yt) / 2, 0])
        wlab = label("warmup", 18, PRIMARY).next_to([X(wf), yt, 0], UP, buff=0.1)
        dot = Dot(color=SECONDARY, radius=0.12).move_to(wpts[0])
        self.play(FadeIn(shade), FadeIn(wlab), run_time=0.3)
        self.play(Create(wpath), MoveAlongPath(dot, wpath), FadeIn(dot), run_time=0.9, rate_func=linear)
        self.caption("Warmup: ramp the step size up slowly so the first updates do not blow up.")

        # ---- cosine decay ----
        self.next_section("cosine", skip_animations=False)
        dpts = [[X(t), Y(lr_of(t)), 0] for t in np.linspace(wf, 1.0, 90)]
        dpath = polyline(dpts, SECONDARY, 4)
        dlab = label("cosine decay", 18, SECONDARY).move_to([X(0.55), Y(lr_of(0.42)) + 0.45, 0])
        self.play(Create(dpath), MoveAlongPath(dot, dpath), FadeIn(dlab), run_time=1.5, rate_func=linear)
        self.caption("Cosine decay: lower the step size smoothly to settle into a good minimum.")

        # ---- mark peak and floor ----
        self.next_section("annotate", skip_animations=False)
        peak = DashedLine([x0, Y(1.0), 0], [X(wf), Y(1.0), 0], color=MUTED, stroke_width=2, dash_length=0.1)
        floor = DashedLine([x0, Y(minf), 0], [x1, Y(minf), 0], color=MUTED, stroke_width=2, dash_length=0.1)
        peakl = label("max LR", 16, TEXT).next_to(peak, LEFT, buff=0.12).shift(UP * 0.0)
        floorl = label("min LR", 16, MUTED).next_to([x1, Y(minf), 0], UP, buff=0.08)
        self.play(Create(peak), Create(floor), FadeIn(peakl), FadeIn(floorl), run_time=0.7)
        self.caption("A single peak, then a long glide down to a small floor.")
        self.wait(0.3)


class ScalingLawScene(StepScene):
    """The power-law line in compute, then the Chinchilla valley at fixed compute."""

    def axes(self, xname, yname):
        x0, x1, yb, yt = -5.0, 5.0, -2.2, 1.9
        xa = Line([x0, yb, 0], [x1 + 0.2, yb, 0], color=MUTED, stroke_width=2)
        ya = Line([x0, yb, 0], [x0, yt + 0.4, 0], color=MUTED, stroke_width=2)
        xl = label(xname, 18, MUTED).next_to(xa, DOWN, buff=0.15).align_to(xa, RIGHT)
        yl = label(yname, 18, MUTED).rotate(PI / 2).next_to(ya, LEFT, buff=0.12)
        return VGroup(xa, ya, xl, yl), (x0, x1, yb, yt)

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Scaling Laws"))

        # ---- power law (loss vs compute) ----
        self.next_section("powerlaw", skip_animations=False)
        ax1, (x0, x1, yb, yt) = self.axes("compute  (log scale)", "loss  (log)")
        self.play(Create(ax1[:2]), FadeIn(ax1[2:]), run_time=0.7)
        p_start, p_end = [-4.2, 1.55, 0], [3.2, -1.35, 0]
        line = Line(p_start, p_end, color=PRIMARY, stroke_width=4)
        dot_t = np.linspace(0.06, 0.94, 4)
        dots = VGroup(*[Dot(line.point_from_proportion(t), color=SECONDARY, radius=0.09) for t in dot_t])
        self.play(Create(line), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.2), run_time=0.7)
        self.caption("Empirically, loss falls as a near-straight line in compute on log-log axes.")

        # ---- extrapolate ----
        self.next_section("extrapolate", skip_animations=False)
        ext = DashedLine(p_end, [4.6, -2.05, 0], color=PRIMARY, stroke_width=4, dash_length=0.14)
        arrow = Arrow([2.0, -2.3, 0], [4.2, -2.3, 0], buff=0, color=MUTED, stroke_width=3)
        moret = label("more compute, bigger models", 17, MUTED).next_to(arrow, UP, buff=0.08)
        self.play(Create(ext), GrowArrow(arrow), FadeIn(moret), run_time=0.8)
        self.caption("This predictability is what justified spending more on scale.")

        # ---- the chinchilla valley ----
        self.next_section("chinchilla", skip_animations=False)
        self.play(FadeOut(VGroup(ax1, line, dots, ext, arrow, moret)), run_time=0.5)
        ax2, (x0, x1, yb, yt) = self.axes("model size  (parameters)", "loss")
        budget = label("fixed compute budget", 18, SECONDARY).move_to([2.7, 1.6, 0])
        self.play(Create(ax2[:2]), FadeIn(ax2[2:]), FadeIn(budget), run_time=0.7)
        xm, ymin, a = 0.3, -1.45, 0.135
        uxs = np.linspace(-4.2, 4.4, 80)
        upts = [[x, min(ymin + a * (x - xm) ** 2, 1.7), 0] for x in uxs]
        ucurve = polyline(upts, PRIMARY, 4)
        self.play(Create(ucurve), run_time=1.2)
        self.caption("For a FIXED compute budget, loss versus model size is U-shaped.")

        # ---- the optimum and the rule of thumb ----
        self.next_section("rule", skip_animations=False)
        minpt = [xm, ymin, 0]
        optdot = Dot(minpt, color=SECONDARY, radius=0.13)
        vmark = DashedLine([xm, yb, 0], minpt, color=MUTED, stroke_width=2, dash_length=0.1)
        optlab = label("compute-optimal", 18, SECONDARY).next_to(optdot, UP, buff=0.2)
        self.play(FadeIn(optdot, scale=1.5), Create(vmark), FadeIn(optlab), run_time=0.7)
        left = label("too small:\nunderfits", 15, MUTED).move_to([-3.25, -0.95, 0])
        right = label("too big:\ntoo few tokens", 15, MUTED).move_to([3.45, 0.55, 0])
        self.play(FadeIn(left), FadeIn(right), run_time=0.5)
        self.caption("The optimum balances parameters against training tokens: about 20 tokens per parameter.")
        self.wait(0.3)


class DataParallelScene(StepScene):
    """Data parallelism: replicate the model, split the batch, all-reduce gradients."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Data Parallelism", "many GPUs, one shared model"))

        gx = [-4.6, -1.55, 1.55, 4.6]
        gy = 1.75
        gpus = VGroup(*[cell(f"GPU {k}\nmodel copy", 2.3, 1.0, PRIMARY, size=19).move_to([x, gy, 0])
                        for k, x in enumerate(gx)])

        # ---- replicas ----
        self.next_section("replicas", skip_animations=False)
        self.play(LaggedStart(*[FadeIn(g) for g in gpus], lag_ratio=0.12), run_time=0.9)
        self.caption("Data parallelism: put an identical copy of the model on every GPU.")

        # ---- split the batch ----
        self.next_section("split", skip_animations=False)
        batch = VGroup(*[sq(0.4, SECONDARY, 0.55) for _ in range(8)]).arrange(RIGHT, buff=0.06)
        batch.move_to([0, -2.45, 0])
        blab = label("one big batch", 17, SECONDARY).next_to(batch, LEFT, buff=0.3)
        self.play(FadeIn(batch), FadeIn(blab), run_time=0.5)
        shards = [VGroup(batch[2 * k], batch[2 * k + 1]) for k in range(4)]
        feeds = VGroup(*[Arrow([gx[k], -1.6, 0], [gx[k], gpus[k].get_bottom()[1] - 0.05, 0],
                               buff=0.05, color=MUTED, stroke_width=3) for k in range(4)])
        self.play(*[shards[k].animate.move_to([gx[k], -0.95, 0]) for k in range(4)], run_time=0.8)
        self.play(LaggedStart(*[GrowArrow(a) for a in feeds], lag_ratio=0.1), FadeOut(blab), run_time=0.6)
        self.caption("Split the batch into shards: each GPU processes a different slice.")

        # ---- local gradients ----
        self.next_section("localgrad", skip_animations=False)
        grads = VGroup(*[cell("grad", 1.0, 0.5, GREEN, size=18).move_to([gx[k], 0.45, 0]) for k in range(4)])
        self.play(FadeOut(VGroup(*shards)), FadeOut(feeds),
                  LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in grads], lag_ratio=0.12), run_time=0.8)
        self.caption("Each GPU backpropagates its shard into a local gradient.")

        # ---- all-reduce ----
        self.next_section("allreduce", skip_animations=False)
        hub = cell("all-reduce\naverage", 2.2, 0.85, SECONDARY, size=18).move_to([0, -0.7, 0])
        self.play(FadeIn(hub), run_time=0.3)
        self.play(*[g.animate.move_to(hub.get_center()).scale(0.5).set_opacity(0.6) for g in grads],
                  run_time=0.8)
        self.play(FadeOut(grads), Indicate(hub, color=SECONDARY, scale_factor=1.1), run_time=0.4)
        avg = VGroup(*[cell("avg", 0.95, 0.48, GREEN, size=17).move_to(hub.get_center()) for _ in range(4)])
        outs = VGroup(*[Arrow(hub.get_top(), [gx[k], gpus[k].get_bottom()[1] - 0.05, 0],
                              buff=0.1, color=GREEN, stroke_width=3,
                              max_tip_length_to_length_ratio=0.12) for k in range(4)])
        self.play(LaggedStart(*[GrowArrow(a) for a in outs], lag_ratio=0.08),
                  *[avg[k].animate.move_to([gx[k], 0.55, 0]) for k in range(4)], run_time=0.9)
        self.play(*[Indicate(g, color=GREEN, scale_factor=1.06) for g in gpus], run_time=0.6)
        self.caption("All-reduce averages the gradients, so every replica updates identically.")
        self.wait(0.3)


class PerplexityScene(StepScene):
    """Cross-entropy becomes perplexity (effective choices) and bits per token."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Loss, Perplexity, and Bits", "after the prefix 'to b'"))

        toks = ["e", "a", "o", " ", "r", "u", "l", "y"]
        before = np.array([0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.08])
        after = np.array([0.78, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
        bx0, bdx, unit, base = -5.4, 0.62, 2.6, -1.4

        def bars_of(probs):
            g = VGroup()
            for i, p in enumerate(probs):
                col = SECONDARY if i == 0 else PRIMARY
                h = max(p * unit, 0.05)
                g.add(Rectangle(width=0.46, height=h, stroke_width=1.4, stroke_color=col,
                                fill_color=col, fill_opacity=0.6).move_to([bx0 + i * bdx, base + h / 2, 0]))
            return g

        tlabels = VGroup(*[label(t if t != " " else "_", 17, MUTED).move_to([bx0 + i * bdx, base - 0.28, 0])
                           for i, t in enumerate(toks)])

        def panel(loss, ppl, bits, ppl_note):
            rows = VGroup(
                label(f"loss = {loss}  nats", 22, TEXT),
                label(f"perplexity = exp(loss) = {ppl}", 22, SECONDARY),
                label(ppl_note, 15, MUTED),
                label(f"bits / token = loss / ln 2 = {bits}", 22, PRIMARY),
            ).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
            rows[2].next_to(rows[1], DOWN, buff=0.06, aligned_edge=LEFT)
            rows.move_to([2.6, 0.8, 0])
            return rows

        # ---- before: spread out ----
        self.next_section("spread", skip_animations=False)
        bars = bars_of(before)
        truth = label("true next token", 15, SECONDARY).next_to(bars[0], UP, buff=0.15)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.06),
                  FadeIn(tlabels), run_time=1.0)
        self.play(FadeIn(truth), run_time=0.3)
        self.caption("Before training, the model is unsure: probability is spread across many tokens.")

        # ---- perplexity as effective choices ----
        self.next_section("perplexity", skip_animations=False)
        p_before = panel("4.18", "65", "6.03", "the effective number of next-token guesses")
        self.play(FadeIn(p_before), run_time=0.6)
        self.caption("Perplexity = exp(loss): the effective number of guesses. Here, nearly all 65 characters.")

        # ---- after training: sharpened ----
        self.next_section("sharpen", skip_animations=False)
        bars_after = bars_of(after)
        p_after = panel("1.64", "5.1", "2.36", "now only a handful of plausible characters")
        self.play(Transform(bars, bars_after), Transform(p_before, p_after),
                  truth.animate.next_to(bars_after[0], UP, buff=0.15), run_time=1.1)
        self.caption("Training sharpens the distribution: lower loss, and far lower perplexity.")

        # ---- bits and compression ----
        self.next_section("bits", skip_animations=False)
        self.play(Indicate(p_before[3], color=PRIMARY, scale_factor=1.12), run_time=0.6)
        comp = label("fewer bits per token  =  the text compresses better", 19, PRIMARY)
        comp.to_edge(DOWN, buff=1.15)
        self.play(FadeIn(comp), run_time=0.5)
        self.caption("Bits per token is the same loss in Shannon's units: the bridge back to Module 1.")
        self.wait(0.3)


# ======= APPEND SCENES BELOW THIS LINE =======
