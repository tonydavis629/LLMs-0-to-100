"""
Module 2: Perceptrons and Optimization - Manim Animations

Scenes:
  - LinearSeparabilityScene: AND / OR separable, XOR fails
  - MLPBoundaryScene: MLP boundary = many straight lines
  (Folding is now an interactive HTML/JS widget, not a Manim scene.)
  - OptimizerLandscapeScene: 3D loss surface with GD, SGD, Adam
  - OverfittingCurveScene: Train/val curves + 1D fits
  - DataVisualizationScene: Scatter plots of exercise datasets (UNCHANGED)
"""

from manim import *
import numpy as np


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


class LinearSeparabilityScene(Scene):
    """Show that AND and OR are linearly separable, but XOR is not.

    Sections (click-through):
      1. Perceptron as a line + half-plane tint.
      2. AND: one line separates all four points.
      3. OR: one line separates all four points.
      4. XOR: rotating line always misclassifies.
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
            # x = xmin
            if abs(b) > 1e-9:
                y = (c - a * bounds[0][0]) / b
                if bounds[1][0] <= y <= bounds[1][1]:
                    pts.append((bounds[0][0], y))
            # x = xmax
            if abs(b) > 1e-9:
                y = (c - a * bounds[0][1]) / b
                if bounds[1][0] <= y <= bounds[1][1]:
                    pts.append((bounds[0][1], y))
            # y = ymin
            if abs(a) > 1e-9:
                x = (c - b * bounds[1][0]) / a
                if bounds[0][0] <= x <= bounds[0][1]:
                    pts.append((x, bounds[1][0]))
            # y = ymax
            if abs(a) > 1e-9:
                x = (c - b * bounds[1][1]) / a
                if bounds[0][0] <= x <= bounds[0][1]:
                    pts.append((x, bounds[1][1]))
            pts = list({(round(p[0], 6), round(p[1], 6)) for p in pts})
            if len(pts) < 2:
                return Line(ORIGIN, ORIGIN, color=SECONDARY)
            pts.sort(key=lambda p: np.arctan2(p[1] - 0.5, p[0] - 0.5))
            return Line(axes.c2p(pts[0][0], pts[0][1]), axes.c2p(pts[1][0], pts[1][1]), color=SECONDARY, stroke_width=3)

        def build_tint(w1, w2, b):
            tint = VGroup()
            xs = np.linspace(-0.3, 1.3, 14)
            ys = np.linspace(-0.3, 1.3, 14)
            for x in xs:
                for y in ys:
                    s = w1 * x + w2 * y + b
                    if abs(s) < 0.06:
                        continue
                    color = GREEN_C if s > 0 else RED_C
                    sq = Square(
                        side_length=0.25,
                        fill_color=color,
                        fill_opacity=0.18,
                        stroke_width=0,
                    ).move_to(axes.c2p(x, y))
                    tint.add(sq)
            return tint

        def make_points(coords, labels=None):
            dots = VGroup()
            lbls = VGroup()
            for (x, y, cls) in coords:
                color = GREEN_C if cls == 1 else RED_C
                pos = axes.c2p(x, y)
                dot = Dot(pos, radius=0.14, color=color, fill_opacity=0.9)
                dots.add(dot)
                txt = Text(f"({x},{y})={cls}", font=BODY_FONT, font_size=16, color=TEXT).next_to(dot, UR, buff=0.08)
                lbls.add(txt)
            return dots, lbls

        legend = VGroup(
            Dot(radius=0.08, color=GREEN_C),
            Text("Class 1", font=BODY_FONT, font_size=18, color=TEXT),
            Dot(radius=0.08, color=RED_C).shift(RIGHT * 0.3),
            Text("Class 0", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.15).to_corner(UR, buff=0.5)

        # Legend stays for all sections
        self.add(legend)

        # ---- Section 1: Perceptron as line ----
        self.next_section("perceptron_as_line", skip_animations=False)
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.8)
        line_intro = build_line(1, 1, -1)
        tint_intro = build_tint(1, 1, -1)
        cap_intro = Text(
            "The perceptron is a linear classifier; the line is its decision boundary",
            font=BODY_FONT, font_size=20, color=TEXT,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(tint_intro), Create(line_intro), Write(cap_intro), run_time=1.0)
        self.wait(0.3)

        # ---- Section 2: AND separable ----
        self.next_section("and_separable", skip_animations=False)
        self.play(FadeOut(tint_intro), FadeOut(line_intro), FadeOut(cap_intro), run_time=0.3)
        and_pts = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        and_dots, and_lbls = make_points(and_pts)
        line_and = build_line(1, 1, -1.5)
        cap_and = Text("AND: one line works", font=BODY_FONT, font_size=22, color=GREEN_C).to_edge(DOWN, buff=0.6)
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
        line_or = build_line(1, 1, -0.5)
        cap_or = Text("OR: one line works", font=BODY_FONT, font_size=22, color=GREEN_C).to_edge(DOWN, buff=0.6)
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
        fail_text = Text("misclassifies!", font=BODY_FONT, font_size=20, color=RED_C).to_edge(DOWN, buff=0.6)
        cap_xor = Text("XOR: no single line works", font=BODY_FONT, font_size=22, color=SECONDARY).to_edge(DOWN, buff=0.6)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in xor_dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(l) for l in xor_lbls], lag_ratio=0.2),
            run_time=0.8,
        )

        # Rotate a line through several angles
        cx, cy = 0.5, 0.5
        line_len = 2.5
        angles_deg = [20, 70, 110, 160]

        def make_rot_line(angle_deg):
            rad = angle_deg * DEGREES
            dx = line_len * np.cos(rad)
            dy = line_len * np.sin(rad)
            return Line(axes.c2p(cx - dx, cy - dy), axes.c2p(cx + dx, cy + dy), color=SECONDARY, stroke_width=3)

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


class MLPBoundaryScene(Scene):
    """Show that an MLP boundary is composed of multiple straight lines.

    Sections (click-through):
      1. Scatter points: class 1 inside a diamond.
      2. Add 4 hidden-neuron lines one at a time.
      3. Shade the intersection region.
      4. Highlight the final polygon boundary.
    """

    def construct(self):
        self.camera.background_color = BG

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6.5,
            y_length=6.5,
            axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True, "tick_size": 0.06},
            tips=False,
        ).shift(DOWN * 0.2)

        title = Text("Many Lines Make a Curve", font=HEADING_FONT, font_size=28, color=TEXT, weight=BOLD).to_edge(UP, buff=0.3)

        # ---- Generate data: diamond interior = class 1 ----
        rng = np.random.default_rng(42)
        n = 60
        dots = VGroup()
        c0, c1 = [], []
        while len(c0) < n or len(c1) < n:
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            inside = abs(x) + abs(y) < 1.5
            if inside and len(c1) < n:
                c1.append((x, y))
                dots.add(Dot(axes.c2p(x, y), radius=0.05, color=GREEN_C, fill_opacity=0.8))
            elif not inside and len(c0) < n:
                c0.append((x, y))
                dots.add(Dot(axes.c2p(x, y), radius=0.05, color=RED_C, fill_opacity=0.8))

        legend = VGroup(
            Dot(radius=0.06, color=GREEN_C),
            Text("Class 1", font=BODY_FONT, font_size=16, color=TEXT),
            Dot(radius=0.06, color=RED_C).shift(RIGHT * 0.2),
            Text("Class 0", font=BODY_FONT, font_size=16, color=TEXT),
        ).arrange(RIGHT, buff=0.1).to_corner(UR, buff=0.4)

        # ---- Diamond edges as lines ----
        # x+y=1.5, -x+y=1.5, x+y=-1.5, x-y=1.5  => equivalently written as:
        lines_data = [
            (1, 1, 1.5, r"h_1"),
            (-1, 1, 1.5, r"h_2"),
            (1, 1, -1.5, r"h_3"),
            (1, -1, 1.5, r"h_4"),
        ]

        def frame_line(a, b, c):
            pts = []
            bounds = [(-3, 3), (-3, 3)]
            if abs(b) > 1e-9:
                y = (c - a * bounds[0][0]) / b
                if bounds[1][0] <= y <= bounds[1][1]:
                    pts.append((bounds[0][0], y))
                y = (c - a * bounds[0][1]) / b
                if bounds[1][0] <= y <= bounds[1][1]:
                    pts.append((bounds[0][1], y))
            if abs(a) > 1e-9:
                x = (c - b * bounds[1][0]) / a
                if bounds[0][0] <= x <= bounds[0][1]:
                    pts.append((x, bounds[1][0]))
                x = (c - b * bounds[1][1]) / a
                if bounds[0][0] <= x <= bounds[0][1]:
                    pts.append((x, bounds[1][1]))
            pts = list({(round(p[0], 6), round(p[1], 6)) for p in pts})
            if len(pts) < 2:
                return None
            # sort for consistent line drawing
            pts.sort(key=lambda p: np.arctan2(p[1], p[0]))
            return Line(axes.c2p(pts[0][0], pts[0][1]), axes.c2p(pts[1][0], pts[1][1]), color=PRIMARY, stroke_width=2.5)

        line_objs = []
        line_labels = []
        for a, b, c, lab in lines_data:
            ln = frame_line(a, b, c)
            if ln is None:
                ln = Line(ORIGIN, ORIGIN, color=PRIMARY)
            line_objs.append(ln)
            mid = ln.get_center()
            lbl = MathTex(lab, color=PRIMARY, font_size=22).move_to(mid + UR * 0.25)
            line_labels.append(lbl)

        # Diamond polygon fill
        poly = Polygon(
            axes.c2p(1.5, 0), axes.c2p(0, 1.5),
            axes.c2p(-1.5, 0), axes.c2p(0, -1.5),
            fill_color=GREEN_C, fill_opacity=0.15, stroke_width=0,
        )

        # ---- Section 1: Data ----
        self.next_section("data", skip_animations=False)
        self.play(Write(title), Create(axes), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.005),
            FadeIn(legend),
            run_time=1.0,
        )
        self.wait(0.3)

        # ---- Section 2: Lines ----
        self.next_section("lines", skip_animations=False)
        for ln, lbl in zip(line_objs, line_labels):
            self.play(Create(ln), Write(lbl), run_time=0.5)
            self.wait(0.15)

        # ---- Section 3: Regions ----
        self.next_section("regions", skip_animations=False)
        cap_region = Text("Intersection of half-planes", font=BODY_FONT, font_size=20, color=MUTED).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(poly), Write(cap_region), run_time=0.6)
        self.wait(0.3)

        # ---- Section 4: Boundary ----
        self.next_section("boundary", skip_animations=False)
        cap_bnd = Text(
            "The curved-looking boundary is straight lines stitched together",
            font=BODY_FONT, font_size=20, color=TEXT,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeOut(cap_region), run_time=0.2)
        # Highlight polygon border
        poly_border = Polygon(
            axes.c2p(1.5, 0), axes.c2p(0, 1.5),
            axes.c2p(-1.5, 0), axes.c2p(0, -1.5),
            fill_color=GREEN_C, fill_opacity=0.0, stroke_color=GREEN_C, stroke_width=4,
        )
        self.play(Create(poly_border), Write(cap_bnd), run_time=0.8)
        self.wait(0.5)


class OptimizerLandscapeScene(ThreeDScene):
    """3D loss surface with GD, SGD, and Adam trajectories.

    Sections (click-through):
      1. Show 2-weight perceptron diagram.
      2. Show 3D loss surface + brief ball movement.
      3. Gradient Descent trajectory.
      4. SGD trajectory.
      5. Adam trajectory.
    """

    def construct(self):
        self.camera.background_color = BG

        # ---- Fixed dataset ----
        X = np.array([[-1.2, -0.9], [-0.8, -1.1], [-0.5, -0.8],
                      [1.1, 0.8], [0.9, 1.1], [0.8, 0.5],
                      [-0.7, 1.0], [1.0, -0.7]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=float)

        def loss_surface(u, v):
            w = np.array([u, v])
            z = 0.0
            for xi, yi in zip(X, y):
                z_i = np.dot(w, xi)
                p = 1.0 / (1.0 + np.exp(-z_i))
                eps = 1e-12
                bce = -(yi * np.log(p + eps) + (1 - yi) * np.log(1 - p + eps))
                z += bce
            z = z / len(y) + 0.05 * np.dot(w, w)
            return float(np.clip(z, 0.0, 6.0))

        def compute_grad(w, idxs=None):
            if idxs is None:
                idxs = range(len(y))
            grad = np.zeros(2)
            for i in idxs:
                z_i = np.dot(w, X[i])
                p = 1.0 / (1.0 + np.exp(-z_i))
                grad += (p - y[i]) * X[i]
            grad = grad / len(idxs) + 0.1 * w
            return grad

        def compute_path(name, steps, lr, start_w):
            w = np.array(start_w, dtype=float)
            path = [(w[0], w[1], loss_surface(w[0], w[1]))]
            m = np.zeros(2)
            v = np.zeros(2)
            t = 0
            rng = np.random.default_rng(123)
            for _ in range(steps):
                if name == "gd":
                    grad = compute_grad(w)
                    w = w - lr * grad
                elif name == "sgd":
                    batch = rng.choice(len(y), size=3, replace=False)
                    grad = compute_grad(w, batch)
                    w = w - lr * grad
                elif name == "adam":
                    t += 1
                    grad = compute_grad(w)
                    m = 0.9 * m + 0.1 * grad
                    v = 0.999 * v + 0.001 * (grad ** 2)
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    w = w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                w = np.clip(w, -3.0, 3.0)
                path.append((w[0], w[1], loss_surface(w[0], w[1])))
            return path

        # ---- 3D axes ----
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 5, 1],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"color": MUTED, "stroke_width": 1.5},
        )
        axes.x_axis.set_color(PRIMARY)
        axes.y_axis.set_color(SECONDARY)

        surface = axes.plot_surface(
            loss_surface,
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(28, 28),
            colorscale=[PRIMARY, "#2a6bbf", SECONDARY, RED_C],
            fill_opacity=0.75,
        )

        # ---- Fixed-in-frame diagram ----
        neuron = Circle(radius=0.25, color=TEXT, stroke_width=2)
        x1_lab = MathTex("x_1", color=TEXT, font_size=22).next_to(neuron, LEFT, buff=1.0).shift(UP * 0.35)
        x2_lab = MathTex("x_2", color=TEXT, font_size=22).next_to(neuron, LEFT, buff=1.0).shift(DOWN * 0.35)
        w1_line = Line(x1_lab.get_right(), neuron.get_left(), color=PRIMARY, stroke_width=2)
        w2_line = Line(x2_lab.get_right(), neuron.get_left(), color=SECONDARY, stroke_width=2)
        w1_txt = MathTex("w_1", color=PRIMARY, font_size=18).next_to(w1_line, UP, buff=0.08)
        w2_txt = MathTex("w_2", color=SECONDARY, font_size=18).next_to(w2_line, DOWN, buff=0.08)
        diag = VGroup(x1_lab, x2_lab, neuron, w1_line, w2_line, w1_txt, w2_txt)
        diag.to_corner(UL, buff=0.4)
        diag_cap = Text("(w_1, w_2) is one point in weight space", font=BODY_FONT, font_size=16, color=MUTED)
        diag_cap.next_to(diag, DOWN, buff=0.15).align_to(diag, LEFT)
        self.add_fixed_in_frame_mobjects(diag, diag_cap)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        # ---- Section 1: Perceptron diagram ----
        self.next_section("perceptron", skip_animations=False)
        self.play(FadeIn(diag), FadeIn(diag_cap), run_time=0.4)
        self.wait(0.3)

        # ---- Section 2: Surface ----
        self.next_section("surface", skip_animations=False)
        self.play(Create(axes), run_time=0.5)
        self.play(Create(surface), run_time=1.5)

        start_w = [-2.5, 2.5]
        ball = Dot3D(
            axes.c2p(start_w[0], start_w[1], loss_surface(start_w[0], start_w[1])),
            radius=0.12, color=WHITE,
        )
        self.play(GrowFromCenter(ball), run_time=0.3)
        self.play(
            ball.animate.move_to(axes.c2p(0, 0, loss_surface(0, 0))),
            run_time=0.8,
        )
        self.play(
            ball.animate.move_to(axes.c2p(start_w[0], start_w[1], loss_surface(start_w[0], start_w[1]))),
            run_time=0.6,
        )
        self.wait(0.3)

        # ---- Section 3: GD ----
        self.next_section("gradient_descent", skip_animations=False)
        gd_label = Text("Gradient Descent", font=BODY_FONT, font_size=20, color=PRIMARY)
        gd_label.to_corner(DR, buff=0.4)
        self.add_fixed_in_frame_mobjects(gd_label)
        self.play(FadeIn(gd_label), run_time=0.3)

        gd_pts = compute_path("gd", steps=12, lr=0.8, start_w=start_w)
        gd_path = VMobject(color=PRIMARY, stroke_width=3)
        gd_path.set_points_as_corners([axes.c2p(p[0], p[1], p[2]) for p in gd_pts])
        self.play(Create(gd_path), MoveAlongPath(ball, gd_path), run_time=2.5)
        self.wait(0.3)

        # ---- Section 4: SGD ----
        self.next_section("sgd", skip_animations=False)
        self.play(FadeOut(gd_label), FadeOut(gd_path), run_time=0.3)
        sgd_label = Text("Stochastic GD", font=BODY_FONT, font_size=20, color=SECONDARY)
        sgd_label.to_corner(DR, buff=0.4)
        self.add_fixed_in_frame_mobjects(sgd_label)
        self.play(FadeIn(sgd_label), run_time=0.3)

        ball2 = Dot3D(
            axes.c2p(start_w[0], start_w[1], loss_surface(start_w[0], start_w[1])),
            radius=0.12, color=WHITE,
        )
        sgd_pts = compute_path("sgd", steps=15, lr=1.0, start_w=start_w)
        sgd_path = VMobject(color=SECONDARY, stroke_width=3)
        sgd_path.set_points_as_corners([axes.c2p(p[0], p[1], p[2]) for p in sgd_pts])
        self.play(GrowFromCenter(ball2), run_time=0.3)
        self.play(Create(sgd_path), MoveAlongPath(ball2, sgd_path), run_time=2.5)
        self.wait(0.3)

        # ---- Section 5: Adam ----
        self.next_section("adam", skip_animations=False)
        self.play(FadeOut(sgd_label), FadeOut(sgd_path), run_time=0.3)
        adam_label = Text("Adam", font=BODY_FONT, font_size=20, color=GREEN_C)
        adam_label.to_corner(DR, buff=0.4)
        self.add_fixed_in_frame_mobjects(adam_label)
        self.play(FadeIn(adam_label), run_time=0.3)

        ball3 = Dot3D(
            axes.c2p(start_w[0], start_w[1], loss_surface(start_w[0], start_w[1])),
            radius=0.12, color=WHITE,
        )
        adam_pts = compute_path("adam", steps=12, lr=1.0, start_w=start_w)
        adam_path = VMobject(color=GREEN_C, stroke_width=3)
        adam_path.set_points_as_corners([axes.c2p(p[0], p[1], p[2]) for p in adam_pts])
        self.play(GrowFromCenter(ball3), run_time=0.3)
        self.play(Create(adam_path), MoveAlongPath(ball3, adam_path), run_time=2.5)
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

        # ---- Compute best-model x numerically ----
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
        self.play(FadeOut(main_group), run_time=0.4)

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

        # True function (faint)
        for ax in (ax_u, ax_g, ax_o):
            tc = ax.plot(true_fn, x_range=[0, 6], color=MUTED, stroke_width=1.5, stroke_opacity=0.4)
            self.add(tc)

        # Sample points
        for ax in (ax_u, ax_g, ax_o):
            for xi, yi in zip(samp_x, samp_y):
                self.add(Dot(ax.c2p(xi, yi), radius=0.04, color=TEXT, fill_opacity=0.8))

        # Fits
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


class DataVisualizationScene(Scene):
    """Scatter plots of the exercise datasets: linearly separable and XOR.

    Sections:
      1. Show linearly separable data
      2. Show non-linearly separable (XOR) data
    """

    def construct(self):
        self.camera.background_color = BG

        # --- Linearly separable data ---
        ax_lin = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5.5,
            y_length=4.5,
            axis_config={"color": MUTED, "stroke_width": 1.5, "include_ticks": True, "tick_size": 0.05},
            tips=False,
        ).shift(LEFT * 3.2 + DOWN * 0.3)

        lin_title = Text(
            "Linearly Separable", font=HEADING_FONT, font_size=24, color=PRIMARY, weight=BOLD
        ).next_to(ax_lin, UP, buff=0.3)

        # Generate linearly separable data (same seed as exercise)
        rng = np.random.default_rng(42)
        n = 75  # per class
        c0_x = rng.normal(-1.5, 0.8, n)
        c0_y = rng.normal(-1.0, 0.7, n)
        c1_x = rng.normal(1.5, 0.8, n)
        c1_y = rng.normal(1.0, 0.7, n)

        lin_dots = VGroup()
        for i in range(n):
            lin_dots.add(Dot(ax_lin.c2p(c0_x[i], c0_y[i]), radius=0.04, color=RED_C, fill_opacity=0.7))
            lin_dots.add(Dot(ax_lin.c2p(c1_x[i], c1_y[i]), radius=0.04, color=GREEN_C, fill_opacity=0.7))

        # --- XOR data ---
        ax_xor = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5.5,
            y_length=4.5,
            axis_config={"color": MUTED, "stroke_width": 1.5, "include_ticks": True, "tick_size": 0.05},
            tips=False,
        ).shift(RIGHT * 3.2 + DOWN * 0.3)

        xor_title = Text(
            "Non-Linearly Separable (XOR)", font=HEADING_FONT, font_size=24, color=PRIMARY, weight=BOLD
        ).next_to(ax_xor, UP, buff=0.3)

        # XOR-pattern data (same structure as exercise)
        rng2 = np.random.default_rng(42)
        n_xor = 40  # per cluster
        xor_clusters = [
            (-1.5, -1.5, 0), (-1.5, 1.5, 1), (1.5, -1.5, 1), (1.5, 1.5, 0),
        ]
        xor_dots = VGroup()
        for cx, cy, cls in xor_clusters:
            xs = rng2.normal(cx, 0.5, n_xor)
            ys = rng2.normal(cy, 0.5, n_xor)
            color = GREEN_C if cls == 1 else RED_C
            for i in range(n_xor):
                xor_dots.add(Dot(ax_xor.c2p(xs[i], ys[i]), radius=0.04, color=color, fill_opacity=0.7))

        # Legend
        legend = VGroup(
            Dot(radius=0.06, color=GREEN_C),
            Text("Class 1", font=BODY_FONT, font_size=16, color=TEXT),
            Dot(radius=0.06, color=RED_C).shift(RIGHT * 0.2),
            Text("Class 0", font=BODY_FONT, font_size=16, color=TEXT),
        ).arrange(RIGHT, buff=0.1).to_edge(DOWN, buff=0.4)

        # --- Section 1: Linearly separable ---
        self.next_section("data_linear", skip_animations=False)
        self.play(Create(ax_lin), Write(lin_title), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in lin_dots], lag_ratio=0.005),
            run_time=1.0,
        )
        self.play(FadeIn(legend), run_time=0.3)
        self.wait(0.3)

        # --- Section 2: XOR data ---
        self.next_section("data_xor", skip_animations=False)
        self.play(Create(ax_xor), Write(xor_title), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in xor_dots], lag_ratio=0.005),
            run_time=1.0,
        )
        self.wait(0.5)
