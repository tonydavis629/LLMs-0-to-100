"""
Module 2: Perceptrons and Optimization - Manim Animations

Scenes:
  - LinearSeparabilityScene: AND / OR separable with one line, XOR fails
  - OverfittingCurveScene: Train/val curves + 1D fits

Other concepts that used to be manim are now interactive HTML/JS widgets
(folding, the MLP decision boundary, and the loss landscape) or a static
image (the exercise datasets).
"""

from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# High-quality H.264 encoding patch.
#
# Manim 0.20.1 hardcodes the partial-movie encode at CRF 23 (see
# SceneFileWriter.open_partial_movie_stream). CRF 23 is tuned for natural video;
# on our flat dark backgrounds with sharp text it leaves "mosquito" ringing
# around glyph edges, which reads as fuzzy / unevenly-spaced letters once the
# clip is scaled up in the deck. Section videos are stream-COPIED from these
# partial files (combine_files uses add_stream_from_template, no re-encode), so
# improving the partial encode is enough to fix every section.
#
# We drop CRF to 15 (near-transparent for this kind of content) and use the
# "slow" preset so the lower CRF does not balloon file size. We deliberately
# keep yuv420p: yuv444p H.264 does not decode reliably in browsers (Safari in
# particular), and our text is light-on-dark, so the full-resolution luma plane
# carries the edges regardless of chroma subsampling. Only the default mp4/x264
# path is touched; the webm and transparent branches are preserved verbatim.
# Re-render with --disable_caching or Manim reuses the old CRF-23 partial files.
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


# Theme colors matching slides
BG = "#0a0e1a"
TEXT = "#e8eaf0"
MUTED = "#8892a4"
PRIMARY = "#4a9eff"
SECONDARY = "#f5a623"
RED_C = "#e74c3c"
GREEN_C = "#3fb950"
LINE_C = "#2a3450"

HEADING_FONT = "Helvetica Neue"
BODY_FONT = "Helvetica Neue"


# ---------------------------------------------------------------------------
# Crisp-kerning Text.
#
# Manim renders a Text mobject's glyphs at a pixel size tied to `font_size`; at
# small sizes the inter-glyph advances get coarsely rounded, so text drawn at a
# small font_size and then shown near full slide width looks loosely and
# unevenly spaced ("bad kerning"). Rendering at a large reference size and
# scaling the VECTOR result down yields accurate, tight kerning at any display
# size. We shadow Text so every call benefits without touching call sites;
# layout code reads `.width`/`.height` after the scale, so it keeps working.
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


class LinearSeparabilityScene(Scene):
    """Show that AND and OR are linearly separable, but XOR is not.

    Sections (click-through):
      1. Perceptron as a line: the decision boundary.
      2. AND: one line separates all four points.
      3. OR: one line separates all four points.
      4. XOR: a rotating line always misclassifies.

    Deliberately minimal: just the data points and a single separating
    line per section (no half-plane shading).
    """

    def construct(self):
        self.camera.background_color = BG

        axes = Axes(
            x_range=[-0.5, 1.5, 0.5],
            y_range=[-0.5, 1.5, 0.5],
            x_length=5,
            y_length=5,
            axis_config={
                "color": MUTED,
                "stroke_width": 2,
                "include_ticks": True,
                "tick_size": 0.08,
            },
            tips=False,
        ).shift(LEFT * 0.5)

        x_label = axes.get_x_axis_label(
            MathTex("x_1", color=TEXT, font_size=28), edge=DOWN, direction=DOWN, buff=0.3
        )
        y_label = axes.get_y_axis_label(
            MathTex("x_2", color=TEXT, font_size=28), edge=LEFT, direction=LEFT, buff=0.3
        )

        # ---- helpers ----
        def build_line(a, b, c):
            """Return a Line for a*x + b*y = c clipped to the frame."""
            pts = []
            bounds = [(-0.5, 1.5), (-0.5, 1.5)]
            if abs(b) > 1e-9:
                y = (c - a * bounds[0][0]) / b
                if bounds[1][0] <= y <= bounds[1][1]:
                    pts.append((bounds[0][0], y))
            if abs(b) > 1e-9:
                y = (c - a * bounds[0][1]) / b
                if bounds[1][0] <= y <= bounds[1][1]:
                    pts.append((bounds[0][1], y))
            if abs(a) > 1e-9:
                x = (c - b * bounds[1][0]) / a
                if bounds[0][0] <= x <= bounds[0][1]:
                    pts.append((x, bounds[1][0]))
            if abs(a) > 1e-9:
                x = (c - b * bounds[1][1]) / a
                if bounds[0][0] <= x <= bounds[0][1]:
                    pts.append((x, bounds[1][1]))
            pts = list({(round(p[0], 6), round(p[1], 6)) for p in pts})
            if len(pts) < 2:
                return Line(ORIGIN, ORIGIN, color=SECONDARY)
            pts.sort(key=lambda p: np.arctan2(p[1] - 0.5, p[0] - 0.5))
            return Line(
                axes.c2p(pts[0][0], pts[0][1]),
                axes.c2p(pts[1][0], pts[1][1]),
                color=SECONDARY,
                stroke_width=4,
            )

        def make_points(coords):
            dots = VGroup()
            lbls = VGroup()
            for (x, y, cls) in coords:
                color = GREEN_C if cls == 1 else RED_C
                pos = axes.c2p(x, y)
                dot = Dot(pos, radius=0.14, color=color, fill_opacity=0.95)
                dots.add(dot)
                txt = Text(
                    f"({x},{y})={cls}", font=BODY_FONT, font_size=16, color=TEXT
                ).next_to(dot, UR, buff=0.08)
                lbls.add(txt)
            return dots, lbls

        legend = VGroup(
            Dot(radius=0.08, color=GREEN_C),
            Text("Class 1", font=BODY_FONT, font_size=18, color=TEXT),
            Dot(radius=0.08, color=RED_C).shift(RIGHT * 0.3),
            Text("Class 0", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.15).to_corner(UR, buff=0.5)
        self.add(legend)

        # ---- Section 1: Perceptron as a line ----
        self.next_section("perceptron_as_line", skip_animations=False)
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.8)
        line_intro = build_line(1, 1, 0.8)
        cap_intro = Text(
            "The perceptron is a linear classifier: the line is its decision boundary",
            font=BODY_FONT, font_size=20, color=TEXT,
        ).to_edge(DOWN, buff=0.6)
        self.play(Create(line_intro), Write(cap_intro), run_time=1.0)
        self.wait(0.3)

        # ---- Section 2: AND separable ----
        self.next_section("and_separable", skip_animations=False)
        self.play(FadeOut(line_intro), FadeOut(cap_intro), run_time=0.3)
        and_pts = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        and_dots, and_lbls = make_points(and_pts)
        line_and = build_line(1, 1, 1.5)
        cap_and = Text(
            "AND: one line separates the classes", font=BODY_FONT, font_size=22, color=GREEN_C
        ).to_edge(DOWN, buff=0.6)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in and_dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(l) for l in and_lbls], lag_ratio=0.2),
            run_time=0.8,
        )
        self.play(Create(line_and), Write(cap_and), run_time=0.6)
        self.wait(0.3)

        # ---- Section 3: OR separable ----
        self.next_section("or_separable", skip_animations=False)
        self.play(FadeOut(and_dots), FadeOut(and_lbls), FadeOut(line_and), FadeOut(cap_and), run_time=0.3)
        or_pts = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        or_dots, or_lbls = make_points(or_pts)
        line_or = build_line(1, 1, 0.5)
        cap_or = Text(
            "OR: one line separates the classes", font=BODY_FONT, font_size=22, color=GREEN_C
        ).to_edge(DOWN, buff=0.6)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in or_dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(l) for l in or_lbls], lag_ratio=0.2),
            run_time=0.8,
        )
        self.play(Create(line_or), Write(cap_or), run_time=0.6)
        self.wait(0.3)

        # ---- Section 4: XOR fails ----
        self.next_section("xor_fails", skip_animations=False)
        self.play(FadeOut(or_dots), FadeOut(or_lbls), FadeOut(line_or), FadeOut(cap_or), run_time=0.3)
        xor_pts = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        xor_dots, xor_lbls = make_points(xor_pts)
        fail_text = Text("always misclassifies one point", font=BODY_FONT, font_size=20, color=RED_C).to_edge(DOWN, buff=0.6)
        cap_xor = Text("XOR: no single line works", font=BODY_FONT, font_size=22, color=SECONDARY).to_edge(DOWN, buff=0.6)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in xor_dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(l) for l in xor_lbls], lag_ratio=0.2),
            run_time=0.8,
        )

        cx, cy = 0.5, 0.5
        line_len = 2.5
        angles_deg = [20, 70, 110, 160]

        def make_rot_line(angle_deg):
            rad = angle_deg * DEGREES
            dx = line_len * np.cos(rad)
            dy = line_len * np.sin(rad)
            return Line(axes.c2p(cx - dx, cy - dy), axes.c2p(cx + dx, cy + dy), color=SECONDARY, stroke_width=4)

        current_line = make_rot_line(angles_deg[0])
        self.play(Create(current_line), run_time=0.4)
        self.play(FadeIn(fail_text), run_time=0.3)
        for angle in angles_deg[1:]:
            new_line = make_rot_line(angle)
            self.play(Transform(current_line, new_line), run_time=0.6)
            self.wait(0.25)

        self.play(FadeOut(fail_text), FadeOut(current_line), run_time=0.3)
        self.play(Write(cap_xor), run_time=0.6)
        self.wait(0.5)


class OverfittingCurveScene(Scene):
    """Training/validation loss curves, best-model marker, and 1D fits.

    Sections (click-through):
      1. Axes and legend.
      2. Training loss curve.
      3. Validation loss curve.
      4. Best-model marker + overfitting region.
      5. 1D underfit / good / overfit panels.
    """

    def construct(self):
        self.camera.background_color = BG

        def train_loss(x):
            return 2.0 * np.exp(-0.04 * x) + 0.1

        def val_loss(x):
            return 1.8 * np.exp(-0.06 * x) + 0.003 * (x - 30) ** 2 * (x > 30) + 0.3

        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 2.5, 0.5],
            x_length=9,
            y_length=5,
            axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True, "tick_size": 0.06},
            tips=False,
        ).shift(DOWN * 0.2)

        x_label = Text("Epoch", font=BODY_FONT, font_size=20, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.25)
        y_label = Text("Loss", font=BODY_FONT, font_size=20, color=MUTED).next_to(axes.y_axis, LEFT, buff=0.25).shift(UP * 0.5)

        title = Text("Overfitting", font=HEADING_FONT, font_size=30, color=TEXT, weight=BOLD).to_edge(UP, buff=0.3)

        train_curve = axes.plot(train_loss, x_range=[1, 100], color=PRIMARY, stroke_width=3)
        val_curve = axes.plot(val_loss, x_range=[1, 100], color=RED_C, stroke_width=3)

        train_legend = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=PRIMARY, stroke_width=3),
            Text("Training loss", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.1)
        val_legend = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=RED_C, stroke_width=3),
            Text("Validation loss", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.1)
        legend = VGroup(train_legend, val_legend).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_corner(UR, buff=0.5)

        xs_fine = np.linspace(1, 100, 2000)
        sweet_x = float(xs_fine[np.argmin(val_loss(xs_fine))])

        sweet_line = DashedLine(
            axes.c2p(sweet_x, 0), axes.c2p(sweet_x, 2.5),
            color=SECONDARY, stroke_width=2, dash_length=0.1,
        )
        sweet_label = Text("Best model", font=BODY_FONT, font_size=18, color=SECONDARY).next_to(sweet_line, UP, buff=0.1)

        overfit_region = Polygon(
            axes.c2p(sweet_x, 0), axes.c2p(100, 0),
            axes.c2p(100, 2.5), axes.c2p(sweet_x, 2.5),
            fill_color=RED_C, fill_opacity=0.08, stroke_width=0,
        )
        overfit_label = Text("Overfitting", font=BODY_FONT, font_size=20, color=RED_C).move_to(axes.c2p(65, 2.2))

        # ---- Section 1: Axes ----
        self.next_section("overfit_axes", skip_animations=False)
        self.play(Write(title), Create(axes), Write(x_label), Write(y_label), run_time=0.6)
        self.wait(0.2)

        # ---- Section 2: Train curve ----
        self.next_section("overfit_train", skip_animations=False)
        self.play(Create(train_curve), FadeIn(train_legend), run_time=1.0)
        self.wait(0.3)

        # ---- Section 3: Val curve ----
        self.next_section("overfit_val", skip_animations=False)
        self.play(Create(val_curve), FadeIn(val_legend), run_time=1.0)
        self.wait(0.3)

        # ---- Section 4: Region ----
        self.next_section("overfit_region", skip_animations=False)
        self.play(
            Create(sweet_line), Write(sweet_label),
            FadeIn(overfit_region), Write(overfit_label),
            run_time=0.8,
        )
        self.wait(0.5)

        # ---- Section 5: 1D fits ----
        self.next_section("fits", skip_animations=False)
        main_group = VGroup(axes, x_label, y_label, train_curve, val_curve, legend,
                            sweet_line, sweet_label, overfit_region, overfit_label)
        # Fade the title too, so it does not overlap the "Model Complexity" title.
        self.play(FadeOut(main_group), FadeOut(title), run_time=0.4)

        rng = np.random.default_rng(42)
        n_samp = 15
        samp_x = np.linspace(0, 6, n_samp)
        samp_y = np.sin(1.2 * samp_x) + rng.normal(0, 0.25, n_samp)
        true_fn = lambda x: np.sin(1.2 * x)

        ax_cfg = dict(
            x_range=[0, 6, 1], y_range=[-2, 2, 1],
            x_length=2.8, y_length=2.0,
            axis_config={"color": MUTED, "stroke_width": 1.5, "include_ticks": False},
            tips=False,
        )
        ax_u = Axes(**ax_cfg).shift(LEFT * 3.5 + DOWN * 0.2)
        ax_g = Axes(**ax_cfg).shift(DOWN * 0.2)
        ax_o = Axes(**ax_cfg).shift(RIGHT * 3.5 + DOWN * 0.2)

        for ax in (ax_u, ax_g, ax_o):
            tc = ax.plot(true_fn, x_range=[0, 6], color=MUTED, stroke_width=1.5, stroke_opacity=0.4)
            self.add(tc)

        for ax in (ax_u, ax_g, ax_o):
            for xi, yi in zip(samp_x, samp_y):
                self.add(Dot(ax.c2p(xi, yi), radius=0.04, color=TEXT, fill_opacity=0.8))

        c1 = np.polyfit(samp_x, samp_y, 1)
        c5 = np.polyfit(samp_x, samp_y, 5)
        c14 = np.polyfit(samp_x, samp_y, min(14, n_samp - 1))

        def make_poly_fn(coeffs):
            return lambda x: np.clip(np.polyval(coeffs, x), -3, 3)

        fit1 = ax_u.plot(make_poly_fn(c1), x_range=[0, 6], color=PRIMARY, stroke_width=2)
        fit5 = ax_g.plot(make_poly_fn(c5), x_range=[0, 6], color=GREEN_C, stroke_width=2)
        fit14 = ax_o.plot(make_poly_fn(c14), x_range=[0, 6], color=RED_C, stroke_width=2)

        lbl_u = Text("Underfit", font=BODY_FONT, font_size=18, color=PRIMARY).next_to(ax_u, DOWN, buff=0.2)
        lbl_g = Text("Good fit", font=BODY_FONT, font_size=18, color=GREEN_C).next_to(ax_g, DOWN, buff=0.2)
        lbl_o = Text("Overfit", font=BODY_FONT, font_size=18, color=RED_C).next_to(ax_o, DOWN, buff=0.2)

        fits_title = Text("Model Complexity", font=HEADING_FONT, font_size=26, color=TEXT, weight=BOLD).to_edge(UP, buff=0.4)

        self.play(Write(fits_title), run_time=0.4)
        self.play(
            Create(ax_u), Create(ax_g), Create(ax_o),
            run_time=0.5,
        )
        self.play(
            Create(fit1), Write(lbl_u),
            Create(fit5), Write(lbl_g),
            Create(fit14), Write(lbl_o),
            run_time=0.8,
        )
        self.wait(0.5)
