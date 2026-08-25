"""Module 8 Manim animations: ViT patchification and the video token budget.

Two step-through scenes (each split into clips with self.next_section):
  PatchifyScene (slug patchify):    image -> patch grid -> flattened sequence ->
                                    projected patch embeddings
  VideoBudgetScene (slug videobudget): one frame's tokens -> 300 frames ->
                                    the 100k context budget -> overflow at 448px
"""

from manim import *

# ---------------------------------------------------------------------------
# High-quality H.264 encoding patch (copied from Module 6): CRF 15, preset slow,
# yuv420p, so flat dark backgrounds with sharp text stay crisp.
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


def label(text: str, size: int = 28, color: str = TEXT, weight=NORMAL) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, weight=weight)


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
# Scene 1: PatchifyScene (slug patchify)
# ===========================================================================


class PatchifyScene(StepScene):
    """An image is cut into patches, flattened into a sequence, then projected."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("ViT Tokenization", "an image becomes a sequence of patch tokens"))

        # ---- image (a grid of many pixels) ----
        self.next_section("image", skip_animations=False)
        PIX = 16          # 16 x 16 pixels
        PATCH = 4         # 4 x 4 pixels per patch -> 16 patches
        NP = PIX // PATCH
        ps = 0.2          # pixel square size
        ycen = 1.05

        def pix_color(r, c):
            if r >= 12:
                return GREEN               # ground
            if 4 <= r <= 9 and 5 <= c <= 10:
                return RED                 # a red object
            return PRIMARY                 # sky

        def pix_pos(r, c):
            return [(c - (PIX - 1) / 2) * ps, ycen + ((PIX - 1) / 2 - r) * ps, 0]

        pixels = VGroup()
        cell = [[None] * PIX for _ in range(PIX)]
        for r in range(PIX):
            for c in range(PIX):
                sq = Square(side_length=ps, stroke_width=0.4, stroke_color=BG,
                            fill_color=pix_color(r, c), fill_opacity=0.92)
                sq.move_to(pix_pos(r, c))
                cell[r][c] = sq
                pixels.add(sq)
        self.play(FadeIn(pixels, shift=DOWN * 0.15), run_time=0.7)
        dim_lbl = label("16 x 16 pixels", 22, MUTED).move_to([3.9, 1.6, 0])
        self.play(FadeIn(dim_lbl), run_time=0.35)
        self.caption("An image is a grid of pixels: H x W x C, not a list of words.")

        # ---- group the pixels into patches ----
        self.next_section("grid", skip_animations=False)
        self.play(FadeOut(dim_lbl), run_time=0.25)
        patch_groups, borders = [], VGroup()
        for pr in range(NP):
            for pc in range(NP):
                g = VGroup(*[cell[r][c]
                             for r in range(pr * PATCH, pr * PATCH + PATCH)
                             for c in range(pc * PATCH, pc * PATCH + PATCH)])
                patch_groups.append(g)
                borders.add(SurroundingRectangle(g, color=SECONDARY, stroke_width=2.6, buff=0.0))
        self.play(Create(borders), run_time=0.6)
        seps = []
        for i, (g, b) in enumerate(zip(patch_groups, borders)):
            pr, pc = divmod(i, NP)
            seps.append(VGroup(g, b).animate.shift([(pc - 1.5) * 0.06, (1.5 - pr) * 0.06, 0]))
        self.play(*seps, run_time=0.7)
        count = label("16 patches, each 4 x 4 pixels", 22, SECONDARY).move_to([3.7, 1.6, 0])
        self.play(FadeIn(count), run_time=0.35)
        self.caption("Group the pixels into non-overlapping P x P patches. Here, 16.")

        # ---- flatten each patch into one token in a sequence ----
        self.next_section("flatten", skip_animations=False)
        self.play(FadeOut(count), run_time=0.25)
        ts = 0.56
        step = 0.63
        tokens = VGroup()
        transforms = []
        for i, (g, b) in enumerate(zip(patch_groups, borders)):
            pr, pc = divmod(i, NP)
            cols = [pix_color(r, c)
                    for r in range(pr * PATCH, pr * PATCH + PATCH)
                    for c in range(pc * PATCH, pc * PATCH + PATCH)]
            maj = max(set(cols), key=cols.count)
            tok = Square(side_length=ts, stroke_width=1.2, stroke_color=LINE,
                         fill_color=maj, fill_opacity=0.85).move_to([(i - 7.5) * step, -0.85, 0])
            tokens.add(tok)
            transforms.append(ReplacementTransform(VGroup(g, b), tok))
        self.play(LaggedStart(*transforms, lag_ratio=0.05), run_time=1.3)
        seq = tokens
        seq_lbl = label("a sequence of 16 patch tokens", 22, TEXT).next_to(seq, UP, buff=0.35)
        self.play(FadeIn(seq_lbl), run_time=0.4)
        self.caption("Flatten row by row: the 2-D image becomes a 1-D sequence.")

        # ---- project each patch into an embedding ----
        self.next_section("project", skip_animations=False)
        vecs = VGroup()
        arrows = VGroup()
        for i in range(16):
            x = (i - 7.5) * step
            bar = VGroup()
            for s in range(3):
                seg = Rectangle(width=0.30, height=0.28, stroke_width=0.8, stroke_color=LINE,
                                fill_color=[PRIMARY, GREEN, SECONDARY][s], fill_opacity=0.75)
                seg.move_to([x, -2.35 - s * 0.30, 0])
                bar.add(seg)
            vecs.add(bar)
            arrows.add(Arrow([x, -1.15, 0], [x, -1.95, 0], buff=0.05, stroke_width=2.5,
                             color=MUTED, max_tip_length_to_length_ratio=0.35))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.04), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(v, shift=DOWN * 0.1) for v in vecs], lag_ratio=0.04),
                  run_time=0.9)
        eq = label("z = W flatten(p) + b", 30, PRIMARY, weight=BOLD).move_to([0, 1.75, 0])
        self.play(FadeIn(eq), run_time=0.4)
        self.caption("Project each patch into an embedding: the visual token embedding.")
        self.wait(0.3)


# ===========================================================================
# Scene 2: VideoBudgetScene (slug videobudget)
# ===========================================================================


class VideoBudgetScene(StepScene):
    """One frame is 196 tokens; 300 frames overflow a 100k context window."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("The Video Token Budget", "why video explodes the context"))

        # ---- one frame ----
        self.next_section("frame", skip_animations=False)
        frame = Square(side_length=1.7, stroke_width=2.0, stroke_color=PRIMARY,
                       fill_color=PRIMARY, fill_opacity=0.12).move_to([-4.3, 1.3, 0])
        grid = VGroup()
        for k in range(1, 4):
            grid.add(Line(frame.get_corner(UL) + RIGHT * k * 1.7 / 4,
                          frame.get_corner(DL) + RIGHT * k * 1.7 / 4,
                          stroke_width=1.0, color=LINE))
            grid.add(Line(frame.get_corner(UL) + DOWN * k * 1.7 / 4,
                          frame.get_corner(UR) + DOWN * k * 1.7 / 4,
                          stroke_width=1.0, color=LINE))
        one = label("224 x 224, 16 x 16 patches", 22, TEXT).move_to([1.6, 1.9, 0])
        one2 = label("= 196 tokens per frame", 26, SECONDARY, weight=BOLD).move_to([1.6, 1.1, 0])
        self.play(FadeIn(frame), Create(grid), run_time=0.6)
        self.play(FadeIn(one), FadeIn(one2), run_time=0.5)
        self.caption("One video frame is about 196 patch tokens.")

        # ---- 300 frames ----
        self.next_section("frames", skip_animations=False)
        strip = VGroup()
        for k in range(11):
            f = Square(side_length=1.2, stroke_width=1.6, stroke_color=PRIMARY,
                       fill_color=BG, fill_opacity=1.0)
            f.move_to([-5.2 + k * 0.42, -0.9 + k * 0.11, 0])
            strip.add(f)
        cnt = label("300 frames  (10 s at 30 fps)", 24, TEXT).move_to([2.4, -0.5, 0])
        mult = label("300 x 196 = 58,800 tokens", 28, SECONDARY, weight=BOLD).move_to([2.4, -1.4, 0])
        self.play(FadeOut(frame), FadeOut(grid), FadeOut(one), FadeOut(one2), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.1) for f in strip], lag_ratio=0.06),
                  run_time=1.0)
        self.play(FadeIn(cnt), FadeIn(mult), run_time=0.5)
        self.caption("A 10-second clip at 30 fps is 300 frames: 58,800 visual tokens.")

        # ---- the 100k budget ----
        self.next_section("budget", skip_animations=False)
        self.play(FadeOut(strip), FadeOut(cnt), FadeOut(mult), run_time=0.3)
        BARW = 9.0
        x0 = -BARW / 2
        box = Rectangle(width=BARW, height=1.0, stroke_width=2.2, stroke_color=TEXT,
                        fill_opacity=0).move_to([0, 0.5, 0]).set_z_index(5)
        boxlbl = label("100k context window", 22, MUTED).next_to(box, UP, buff=0.18)
        fillw = BARW * 0.588
        fill = Rectangle(width=fillw, height=1.0, stroke_width=0,
                         fill_color=SECONDARY, fill_opacity=0.85)
        fill.align_to(box, LEFT).set_y(0.5)
        flbl = label("58,800 tokens = 59% of the window, before any text", 22, SECONDARY,
                     weight=BOLD).next_to(box, DOWN, buff=0.30)
        self.play(Create(box), FadeIn(boxlbl), run_time=0.5)
        self.play(GrowFromEdge(fill, LEFT), run_time=0.8)
        self.play(FadeIn(flbl), run_time=0.4)
        self.caption("That already fills more than half a 100k context window.")

        # ---- overflow at 448 x 448 ----
        self.next_section("overflow", skip_animations=False)
        self.play(FadeOut(fill), FadeOut(flbl), run_time=0.3)
        over = Rectangle(width=BARW + 3.4, height=1.0, stroke_width=0,
                         fill_color=RED, fill_opacity=0.8)
        over.align_to(box, LEFT).set_y(0.5)
        arrow = Arrow([x0 + BARW, -0.8, 0], [x0 + BARW + 1.7, -0.8, 0], buff=0.1,
                      stroke_width=4, color=RED, max_tip_length_to_length_ratio=0.3)
        olbl = label("448 x 448: 784 tokens/frame  ->  235,200 tokens", 24, RED,
                     weight=BOLD).next_to(box, DOWN, buff=0.30)
        self.play(GrowFromEdge(over, LEFT), run_time=0.8)
        self.play(GrowArrow(arrow), FadeIn(olbl), run_time=0.5)
        self.caption("Double the resolution and it no longer fits, and attention is quadratic.")
        self.wait(0.3)
