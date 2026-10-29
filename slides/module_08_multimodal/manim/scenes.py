"""Module 8 Manim animations: patches, positions, video budget, and audio tokens.

Four step-through scenes (each split into clips with self.next_section):
  PatchifyScene (slug patchify):    image -> patch grid -> flattened sequence ->
                                    projected patch embeddings
  VideoBudgetScene (slug videobudget): one frame's tokens -> 300 frames ->
                                    the 100k context budget -> overflow at 448px
  PosEmbedScene (slug posembed):    patch grid -> flattened sequence -> adding
                                    2-D position vectors -> similarity of codes
  AudioTokensScene (slug audiotokens): waveform -> log-mel spectrogram ->
                                    encoder vectors or codec ids -> one sequence
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


# ===========================================================================
# Scene 3: PosEmbedScene (slug posembed)
# ===========================================================================


def sincos_1d(pos, dim):
    """1-D sine-cosine position code: half sines, half cosines (Vaswani et al.)."""
    omega = 1.0 / 10000 ** (np.arange(dim // 2) / (dim / 2))
    angles = np.outer(pos, omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def sincos_2d(grid, dim):
    """2-D code as in MAE: half the dimensions encode the row, half the column."""
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    return np.concatenate([sincos_1d(rows, dim // 2), sincos_1d(cols, dim // 2)], axis=1)


def heat_rgb(t, lo=BG, hi=PRIMARY, top="#dcecff"):
    """Single-hue sequential scale: dark to blue to pale blue, as an RGB byte triple."""
    if t < 0.6:
        col = interpolate_color(ManimColor(lo), ManimColor(hi), t / 0.6)
    else:
        col = interpolate_color(ManimColor(hi), ManimColor(top), (t - 0.6) / 0.4)
    return (np.array(col.to_rgb()) * 255).astype(np.uint8)


def pixel_image(rgb, height):
    """Show an H x W x 3 array as a crisp (nearest-neighbor) image of given height."""
    img = ImageMobject(rgb)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = height
    return img


class PosEmbedScene(StepScene):
    """Patches lose their 2-D layout when flattened; 2-D position codes restore it."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("2-D Position Embeddings", "putting the layout back after flattening"))
        N = 4
        cs = 0.95

        # ---- the patch grid, each cell labeled (row, col) ----
        self.next_section("grid", skip_animations=False)
        cells = VGroup()
        for i in range(N * N):
            r, c = divmod(i, N)
            sq = Square(side_length=cs, stroke_width=2.0, stroke_color=LINE,
                        fill_color=PRIMARY, fill_opacity=0.14)
            sq.move_to([-3.4 + (c - 1.5) * cs, 0.1 + (1.5 - r) * cs, 0])
            txt = label(f"({r},{c})", 18, TEXT).move_to(sq)
            cells.add(VGroup(sq, txt))
        self.play(LaggedStart(*[FadeIn(g) for g in cells], lag_ratio=0.03), run_time=0.8)
        info = VGroup(
            label("16 patches in a 4 x 4 grid", 24, TEXT),
            label("each patch knows its (row, col)", 24, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.0, 0.3, 0])
        self.play(FadeIn(info), run_time=0.4)
        self.caption("In the image, every patch has a row and a column.")

        # ---- flatten: vertical neighbors end up far apart ----
        self.next_section("flatten", skip_animations=False)
        self.play(FadeOut(info), run_time=0.25)
        self.play(cells.animate.scale(0.62).move_to([-4.7, 1.25, 0]), run_time=0.6)
        hl = [1, 5]
        self.play(*[cells[i][0].animate.set_fill(SECONDARY, opacity=0.55) for i in hl],
                  run_time=0.4)
        step = 0.74
        seq = VGroup()
        for i in range(N * N):
            sq = Square(side_length=0.62, stroke_width=1.6, stroke_color=LINE,
                        fill_color=SECONDARY if i in hl else PRIMARY,
                        fill_opacity=0.55 if i in hl else 0.14)
            sq.move_to([(i - 7.5) * step, -1.75, 0])
            seq.add(VGroup(sq, label(str(i), 18, TEXT).move_to(sq)))
        self.play(TransformFromCopy(cells, seq), run_time=1.1)
        arc = ArcBetweenPoints(seq[5].get_top() + UP * 0.08, seq[1].get_top() + UP * 0.08,
                               angle=PI / 2.2, color=SECONDARY, stroke_width=3)
        gap = label("4 positions apart", 20, SECONDARY).next_to(arc, RIGHT, buff=0.2)
        side = VGroup(
            label("patches 1 and 5 touch vertically", 22, TEXT),
            label("but sit 4 steps apart in the sequence", 22, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([2.6, 1.2, 0])
        self.play(Create(arc), FadeIn(gap), FadeIn(side), run_time=0.6)
        self.caption("Flattening keeps the patches but throws away who is above whom.")

        # ---- add a position vector to every patch embedding ----
        self.next_section("add", skip_animations=False)
        self.play(FadeOut(cells), FadeOut(arc), FadeOut(gap), FadeOut(side), run_time=0.3)
        self.play(seq.animate.move_to([0, 1.55, 0]), run_time=0.5)
        z_lbl = label("z", 22, PRIMARY, weight=BOLD).next_to(seq, LEFT, buff=0.25)
        rows_codes = VGroup()
        for i in range(N * N):
            r, c = divmod(i, N)
            x = seq[i].get_x()
            top = Rectangle(width=0.62, height=0.36, stroke_width=1.0, stroke_color=LINE,
                            fill_color=GREEN, fill_opacity=0.15 + 0.22 * r).move_to([x, 0.3, 0])
            bot = Rectangle(width=0.62, height=0.36, stroke_width=1.0, stroke_color=LINE,
                            fill_color=SECONDARY, fill_opacity=0.15 + 0.22 * c).move_to([x, -0.06, 0])
            rows_codes.add(VGroup(top, bot))
        p_lbl = label("p", 22, GREEN, weight=BOLD).next_to(rows_codes, LEFT, buff=0.25)
        self.play(FadeIn(z_lbl), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(v, shift=UP * 0.1) for v in rows_codes], lag_ratio=0.04),
                  FadeIn(p_lbl), run_time=0.9)
        legend = VGroup(
            VGroup(Square(0.26, stroke_width=0, fill_color=GREEN, fill_opacity=0.8),
                   label("half the dimensions encode the row", 20, TEXT)).arrange(RIGHT, buff=0.2),
            VGroup(Square(0.26, stroke_width=0, fill_color=SECONDARY, fill_opacity=0.8),
                   label("half the dimensions encode the column", 20, TEXT)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([-2.6, -1.35, 0])
        eq = label("x_i = z_i + p(row, col)", 30, TEXT, weight=BOLD).move_to([3.4, -1.35, 0])
        self.play(FadeIn(legend), FadeIn(eq), run_time=0.5)
        self.caption("Each patch embedding gets a position vector added: row part and column part.")

        # ---- the similarity of every position vector to every other ----
        self.next_section("similarity", skip_animations=False)
        self.play(*[FadeOut(m) for m in [seq, z_lbl, rows_codes, p_lbl, legend, eq]],
                  run_time=0.4)
        G, DIM, CELL, GAP = 7, 64, 10, 4
        P = sincos_2d(G, DIM)
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        S = P @ P.T
        # Normalize each patch's tile on its own so every tile uses the full scale.
        S = (S - S.min(axis=1, keepdims=True)) / np.ptp(S, axis=1, keepdims=True)
        S = S ** 2  # stretch the top of the scale so the near neighborhood stands out
        side_px = G * (G * CELL) + (G + 1) * GAP
        img = np.zeros((side_px, side_px, 3), dtype=np.uint8)
        img[:, :] = (np.array(ManimColor(LINE).to_rgb()) * 255).astype(np.uint8)
        for q in range(G * G):
            qr, qc = divmod(q, G)
            y0 = GAP + qr * (G * CELL + GAP)
            x0 = GAP + qc * (G * CELL + GAP)
            for k in range(G * G):
                kr, kc = divmod(k, G)
                img[y0 + kr * CELL:y0 + (kr + 1) * CELL,
                    x0 + kc * CELL:x0 + (kc + 1) * CELL] = heat_rgb(S[q, k])
        heat = pixel_image(img, 5.1).move_to([-2.4, -0.25, 0])
        row_ax = label("patch row", 20, MUTED).rotate(PI / 2).next_to(heat, LEFT, buff=0.2)
        col_ax = label("patch column", 20, MUTED).next_to(heat, UP, buff=0.12)
        self.play(FadeIn(heat), FadeIn(row_ax), FadeIn(col_ax), run_time=0.8)
        scale = VGroup(*[
            Square(0.3, stroke_width=0, fill_color=rgb_to_color(heat_rgb(t) / 255),
                   fill_opacity=1.0)
            for t in np.linspace(0, 1, 9)
        ]).arrange(RIGHT, buff=0)
        ends = VGroup(label("less similar", 18, MUTED).next_to(scale, LEFT, buff=0.15),
                      label("more", 18, MUTED).next_to(scale, RIGHT, buff=0.15))
        key = VGroup(scale, ends).move_to([3.8, -1.6, 0])
        notes = VGroup(
            label("tile (r, c) = patch (r, c); each cell in it", 21, TEXT),
            label("compares its position vector to one patch", 21, TEXT),
            label("2-D sine-cosine codes, 7 x 7 grid, 64 dims", 19, MUTED),
            label("each tile scaled to its own range", 19, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.8, 0.6, 0])
        self.play(FadeIn(notes), FadeIn(key), run_time=0.5)
        self.caption("Nearby patches get similar position vectors: rows and columns are recoverable.")
        self.wait(0.3)


# ===========================================================================
# Scene 4: AudioTokensScene (slug audiotokens)
# ===========================================================================

SR = 16000


def speech_like(seconds=1.5, seed=0):
    """Synthetic 'speech': voiced syllables (gliding harmonics) plus one hiss burst."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 50 * np.sin(2 * np.pi * 0.9 * t)           # pitch glides
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = np.zeros_like(t)
    for h in range(1, 25):
        formant = np.exp(-((h * f0 - 700) / 450) ** 2) + 0.6 * np.exp(-((h * f0 - 1800) / 600) ** 2)
        voiced += (0.15 + formant) / h * np.sin(h * phase)
    syll = np.zeros_like(t)
    for c, w in [(0.22, 0.12), (0.55, 0.14), (1.2, 0.16)]:
        syll += np.exp(-((t - c) / w) ** 2)
    hiss = rng.normal(size=t.size)
    hiss = np.convolve(hiss, [1, -1.6, 1], mode="same")      # crude high-pass
    hiss *= 0.35 * np.exp(-((t - 0.88) / 0.07) ** 2)
    return t, voiced * syll / 3 + hiss


def log_mel(x, n_fft=400, hop=160, n_mels=80):
    """Whisper-style front end: 25 ms windows, 10 ms hop, 80 mel bins, log scale."""
    win = np.hanning(n_fft)
    frames = [x[i:i + n_fft] * win for i in range(0, len(x) - n_fft, hop)]
    power = np.abs(np.fft.rfft(np.array(frames), axis=1)) ** 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = inv(np.linspace(mel(0), mel(SR / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mels, bins.size))
    for m in range(n_mels):
        lo, mid, hi = edges[m:m + 3]
        fb[m] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None)
    spec = np.log10(power @ fb.T + 1e-6)
    return spec.T  # (mel bins, frames)


def token_box(text, color, w=0.62, h=0.5, size=17, opacity=0.25):
    box = Rectangle(width=w, height=h, stroke_width=1.6, stroke_color=color,
                    fill_color=color, fill_opacity=opacity)
    return VGroup(box, label(text, size, TEXT).move_to(box))


def block(text, sub, color, w=2.4, h=1.0):
    box = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_width=2.2,
                           stroke_color=color, fill_color=BG, fill_opacity=1.0)
    t = label(text, 20, TEXT, weight=BOLD)
    s = label(sub, 16, MUTED)
    VGroup(t, s).arrange(DOWN, buff=0.1).move_to(box)
    return VGroup(box, t, s)


def arrow(a, b, color=MUTED):
    return Arrow(a, b, buff=0.08, stroke_width=3, color=color, max_tip_length_to_length_ratio=0.25)


class AudioTokensScene(StepScene):
    """Waveform -> spectrogram -> encoder vectors (continuous) or codec ids (discrete)."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("How Audio Becomes LLM Input", "two routes: continuous vectors or discrete codec tokens"))
        t, x = speech_like()

        # ---- the raw waveform ----
        self.next_section("wave", skip_animations=False)
        idx = np.linspace(0, len(x) - 1, 1400).astype(int)
        xs = -6.0 + 12.0 * idx / (len(x) - 1)
        ys = 0.2 + 1.3 * x[idx] / np.abs(x).max()
        wave = VMobject(stroke_color=PRIMARY, stroke_width=1.6)
        wave.set_points_as_corners([[a, b, 0] for a, b in zip(xs, ys)])
        axis = Line([-6.0, 0.2, 0], [6.0, 0.2, 0], stroke_width=1, color=LINE)
        self.play(FadeIn(axis), Create(wave), run_time=1.2)
        facts = VGroup(
            label("16,000 samples per second", 24, TEXT),
            label("30 s = 480,000 numbers", 24, SECONDARY, weight=BOLD),
        ).arrange(RIGHT, buff=1.2).move_to([0, -1.9, 0])
        self.play(FadeIn(facts), run_time=0.4)
        self.caption("Raw audio is a long list of amplitudes: far too long to feed in directly.")

        # ---- log-mel spectrogram ----
        self.next_section("spectrogram", skip_animations=False)
        spec = log_mel(x)
        lo, hi = np.percentile(spec, 5), spec.max()
        norm = np.clip((spec - lo) / (hi - lo), 0, 1)[::-1]   # low frequencies at the bottom
        rgb = np.zeros(norm.shape + (3,), dtype=np.uint8)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                rgb[i, j] = heat_rgb(norm[i, j], hi=SECONDARY, top="#fff1d6")
        spec_img = pixel_image(rgb, 3.3)
        spec_img.stretch_to_fit_width(8.4).move_to([-1.6, 0.1, 0])
        self.play(FadeOut(facts), wave.animate.stretch_to_fit_width(8.4).stretch_to_fit_height(0.9)
                  .move_to([-1.6, 2.15, 0]), FadeOut(axis), run_time=0.6)
        self.play(FadeIn(spec_img), run_time=0.7)
        t_ax = label("time", 18, MUTED).next_to(spec_img, DOWN, buff=0.12)
        f_ax = label("frequency (mel)", 18, MUTED).rotate(PI / 2).next_to(spec_img, LEFT, buff=0.15)
        side = VGroup(
            label("80 mel bins", 22, TEXT),
            label("one frame every 10 ms", 22, TEXT),
            label("= 100 frames/s", 22, SECONDARY, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.8, 0.1, 0])
        self.play(FadeIn(t_ax), FadeIn(f_ax), FadeIn(side), run_time=0.4)
        self.caption("A log-mel spectrogram: energy per frequency band, frame by frame.")

        # ---- continuous route: encoder vectors -> projector -> LLM prefix ----
        self.next_section("encoder", skip_animations=False)
        self.play(FadeOut(wave), FadeOut(t_ax), FadeOut(f_ax), FadeOut(side), run_time=0.3)
        self.play(spec_img.animate.stretch_to_fit_width(2.6).stretch_to_fit_height(1.1)
                  .move_to([-5.3, 1.3, 0]), run_time=0.6)
        enc = block("Audio encoder", "conv stem, stride 2", PRIMARY).move_to([-2.1, 1.3, 0])
        vecs = VGroup(*[
            Rectangle(width=0.22, height=0.9, stroke_width=1, stroke_color=LINE,
                      fill_color=GREEN, fill_opacity=0.65) for _ in range(10)
        ]).arrange(RIGHT, buff=0.08).move_to([1.4, 1.3, 0])
        vlbl = VGroup(label("1,500 vectors per 30 s", 20, GREEN),
                      label("(50 per second)", 18, GREEN)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(vecs, RIGHT, buff=0.3)
        self.play(GrowArrow(arrow([-3.95, 1.3, 0], [-3.35, 1.3, 0])), FadeIn(enc), run_time=0.5)
        self.play(GrowArrow(arrow([-0.85, 1.3, 0], vecs.get_left())),
                  LaggedStart(*[FadeIn(v) for v in vecs], lag_ratio=0.05), run_time=0.7)
        self.play(FadeIn(vlbl), run_time=0.3)
        proj = block("Projector", "to LLM width", GREEN, w=2.2, h=0.9).move_to([1.4, -0.45, 0])
        seq = VGroup(*[token_box("", GREEN, w=0.42, opacity=0.6) for _ in range(6)],
                     *[token_box(w, PRIMARY, w=1.0) for w in ["What", "is", "said?"]]
                     ).arrange(RIGHT, buff=0.08)
        seq.shift([1.4 - seq[:6].get_x(), -2.0 - seq.get_y(), 0])
        pre = label("audio prefix", 18, GREEN).next_to(seq[:6], DOWN, buff=0.15)
        txt = label("text tokens", 18, PRIMARY).next_to(seq[6:], DOWN, buff=0.15)
        self.play(GrowArrow(arrow(vecs.get_bottom(), proj.get_top())), FadeIn(proj), run_time=0.5)
        self.play(GrowArrow(arrow(proj.get_bottom(), seq[:6].get_top())),
                  FadeIn(seq), FadeIn(pre), FadeIn(txt), run_time=0.6)
        self.caption("Continuous route: encoder vectors are projected in, just like the image prefix.")
        self.wait(0.3)

        # ---- discrete route: a neural codec with residual vector quantization ----
        self.next_section("codec", skip_animations=False)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (self.cap,)
                    and m not in self.mobjects[:1]], run_time=0.4)
        mini = VMobject(stroke_color=PRIMARY, stroke_width=1.4)
        mini.set_points_as_corners([[a, b, 0] for a, b in zip(
            np.linspace(-6.4, -4.6, 300), 1.8 + 0.4 * x[idx[::4][:300]] / np.abs(x).max())])
        codec = block("Codec encoder", "EnCodec, SoundStream", PRIMARY, w=2.7).move_to([-2.4, 1.8, 0])
        frame = VGroup(*[
            Square(0.26, stroke_width=1, stroke_color=LINE, fill_color=PRIMARY,
                   fill_opacity=o) for o in [0.8, 0.3, 0.6, 0.45, 0.9, 0.2]
        ]).arrange(DOWN, buff=0).move_to([0.6, 1.8, 0])
        flbl = label("one frame = 1/75 s", 18, MUTED).next_to(frame, RIGHT, buff=0.25)
        self.play(Create(mini), FadeIn(codec), GrowArrow(arrow([-4.5, 1.8, 0], [-3.8, 1.8, 0])),
                  run_time=0.6)
        self.play(GrowArrow(arrow([-1.0, 1.8, 0], [0.4, 1.8, 0])), FadeIn(frame), FadeIn(flbl),
                  run_time=0.5)
        steps = [
            ("codebook 1: nearest of 1,024 entries", "517"),
            ("codebook 2: quantize what is left over", "88"),
            ("...", "..."),
            ("codebook 8: the last residual", "903"),
        ]
        rows = VGroup()
        for k, (desc, tok) in enumerate(steps):
            y = 0.35 - k * 0.68
            d = label(desc, 20, TEXT if desc != "..." else MUTED)
            d.move_to([-1.9, y, 0], aligned_edge=LEFT).shift(LEFT * 2.5)
            b = token_box(tok, SECONDARY, w=0.9, h=0.5, size=19, opacity=0.35).move_to([1.7, y, 0])
            rows.add(VGroup(d, b))
        for r in rows:
            self.play(FadeIn(r[0]), FadeIn(r[1], shift=RIGHT * 0.1), run_time=0.35)
        stack = VGroup(*[token_box(s[1], SECONDARY, w=0.9, h=0.5, size=19, opacity=0.35)
                         for s in steps]).arrange(DOWN, buff=0.08).move_to([4.4, -0.65, 0])
        slbl = VGroup(label("8 ids per frame", 19, SECONDARY, weight=BOLD),
                      label("75 frames/s at 24 kHz", 18, MUTED)
                      ).arrange(DOWN, buff=0.08).next_to(stack, UP, buff=0.2)
        self.play(TransformFromCopy(VGroup(*[r[1] for r in rows]), stack), FadeIn(slbl),
                  run_time=0.7)
        dec = VGroup(label("a decoder turns ids", 18, MUTED), label("back into a waveform", 18, MUTED)
                     ).arrange(DOWN, buff=0.08).next_to(stack, DOWN, buff=0.22)
        self.play(FadeIn(dec), run_time=0.3)
        self.caption("Discrete route: integer codebook ids the LLM can read and also generate.")

        # ---- both routes end as one sequence in the same transformer ----
        self.next_section("sequence", skip_animations=False)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (self.cap,)
                    and m not in self.mobjects[:1]], run_time=0.4)
        llm = RoundedRectangle(corner_radius=0.15, width=11.6, height=1.0, stroke_width=2.4,
                               stroke_color=TEXT, fill_color=BG, fill_opacity=1).move_to([0, 0.35, 0])
        llm_t = label("Language model: the same transformer", 22, TEXT, weight=BOLD).move_to(llm)
        ins = VGroup(token_box("Transcribe", PRIMARY, w=1.6),
                     token_box("this:", PRIMARY, w=1.0),
                     *[token_box(i, SECONDARY, w=0.75) for i in ["517", "88", "903", "212", "45", "760"]],
                     token_box("Answer:", PRIMARY, w=1.4),
                     ).arrange(RIGHT, buff=0.08).move_to([0, -1.35, 0])
        in_lbl = label("input: text tokens and audio tokens, interleaved", 19, MUTED).next_to(ins, DOWN, buff=0.2)
        outs = VGroup(token_box("hello", PRIMARY, w=1.1), token_box("world", PRIMARY, w=1.1),
                      label("or", 18, MUTED),
                      *[token_box(i, SECONDARY, w=0.75) for i in ["31", "640", "7"]],
                      ).arrange(RIGHT, buff=0.12).move_to([-3.0, 2.05, 0])
        out_lbl = label("output: text, or audio ids for a decoder to speak", 19, MUTED).next_to(outs, RIGHT, buff=0.3)
        self.play(FadeIn(llm), FadeIn(llm_t), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in ins], lag_ratio=0.06),
                  FadeIn(in_lbl), run_time=0.8)
        self.play(GrowArrow(arrow(ins.get_top() + UP * 0.02, llm.get_bottom())),
                  run_time=0.3)
        self.play(GrowArrow(arrow([outs.get_x(), llm.get_top()[1], 0], outs.get_bottom() + DOWN * 0.02)),
                  FadeIn(outs), FadeIn(out_lbl), run_time=0.5)
        self.caption("Either route ends the same way: audio becomes a sequence beside the words.")
        self.wait(0.3)
