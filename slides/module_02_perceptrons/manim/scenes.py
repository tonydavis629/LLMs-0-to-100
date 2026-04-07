"""
Module 2: Perceptrons and Optimization - Manim Animations

Scenes:
  - XORVisualizationScene: XOR points and rotating decision boundary
  - FoldingScene: Hidden layer folds space to separate entangled classes
  - GradientDescentScene: Ball stepping downhill on loss curve
  - LossLandscapeScene: 3D loss surface with sharp/flat minima, saddle point
  - OverfittingCurveScene: Training vs validation loss curves
  - DataVisualizationScene: Scatter plots of the exercise datasets
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


class XORVisualizationScene(Scene):
    """Show XOR points on a 2D grid and animate a decision boundary rotating
    through positions, demonstrating that no single line can separate them.

    Sections (click-through):
      1. Show axes and four XOR points with labels
      2. Animate line rotating through several angles, each misclassifying
      3. Show conclusion text
    """

    def construct(self):
        self.camera.background_color = BG

        # --- Setup axes ---
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

        # XOR points: (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=0
        points_data = [
            (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
        ]
        dots = VGroup()
        labels = VGroup()
        for x, y, cls in points_data:
            color = GREEN_C if cls == 1 else RED_C
            pos = axes.c2p(x, y)
            dot = Dot(pos, radius=0.15, color=color, fill_opacity=0.9)
            label = Text(
                f"({x},{y})={cls}", font=BODY_FONT, font_size=16, color=TEXT
            ).next_to(dot, UR, buff=0.1)
            dots.add(dot)
            labels.add(label)

        # Legend
        legend = VGroup(
            Dot(radius=0.08, color=GREEN_C),
            Text("Class 1", font=BODY_FONT, font_size=18, color=TEXT),
            Dot(radius=0.08, color=RED_C).shift(RIGHT * 0.3),
            Text("Class 0", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.15).to_corner(UR, buff=0.5)

        # --- Section 1: Show points ---
        self.next_section("xor_points", skip_animations=False)
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.8)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2),
            FadeIn(legend),
            run_time=1.0,
        )
        self.wait(0.3)

        # --- Section 2: Rotating decision boundary ---
        self.next_section("xor_line_rotate", skip_animations=False)

        # Try several line angles: none can separate XOR
        angles_deg = [20, 70, 110, 160]

        # Create a line that passes through center of the grid
        cx, cy = 0.5, 0.5
        line_length = 2.5

        def make_line(angle_deg):
            rad = angle_deg * DEGREES
            dx = line_length * np.cos(rad)
            dy = line_length * np.sin(rad)
            start = axes.c2p(cx - dx, cy - dy)
            end = axes.c2p(cx + dx, cy + dy)
            return Line(start, end, color=SECONDARY, stroke_width=3)

        current_line = make_line(angles_deg[0])
        fail_text = Text(
            "misclassifies!", font=BODY_FONT, font_size=20, color=RED_C
        ).to_edge(DOWN, buff=0.6)

        self.play(Create(current_line), run_time=0.5)
        self.play(FadeIn(fail_text), run_time=0.3)

        for angle in angles_deg[1:]:
            new_line = make_line(angle)
            self.play(Transform(current_line, new_line), run_time=0.6)
            self.wait(0.3)

        self.wait(0.3)

        # --- Section 3: Conclusion ---
        self.next_section("xor_conclusion", skip_animations=False)

        self.play(FadeOut(fail_text), FadeOut(current_line), run_time=0.3)

        conclusion = Text(
            "No single line can separate XOR",
            font=HEADING_FONT, font_size=30, color=SECONDARY, weight=BOLD,
        ).to_edge(DOWN, buff=0.6)

        self.play(Write(conclusion), run_time=0.8)
        self.wait(0.5)


class FoldingScene(Scene):
    """Demonstrate how a hidden layer transforms (folds) the input space
    so that previously entangled classes become linearly separable.

    Uses a 2D XOR pattern. Shows:
      1. Original input space with mixed classes
      2. Arrow indicating transformation
      3. Hidden representation space where classes are separated
      4. A line separating them in the new space
    """

    def construct(self):
        self.camera.background_color = BG

        # --- Input space (left) ---
        ax_in = Axes(
            x_range=[-0.5, 1.5, 0.5],
            y_range=[-0.5, 1.5, 0.5],
            x_length=4,
            y_length=4,
            axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True, "tick_size": 0.06},
            tips=False,
        ).shift(LEFT * 3.5)

        in_title = Text(
            "Input Space", font=HEADING_FONT, font_size=24, color=PRIMARY, weight=BOLD
        ).next_to(ax_in, UP, buff=0.3)

        # XOR points with some noise for visual interest
        rng = np.random.default_rng(42)
        n_per_cluster = 8
        clusters = [
            (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
        ]
        input_dots = VGroup()
        input_coords = []
        input_classes = []
        for cx, cy, cls in clusters:
            for _ in range(n_per_cluster):
                x = cx + rng.normal(0, 0.08)
                y = cy + rng.normal(0, 0.08)
                color = GREEN_C if cls == 1 else RED_C
                dot = Dot(ax_in.c2p(x, y), radius=0.07, color=color, fill_opacity=0.85)
                input_dots.add(dot)
                input_coords.append((x, y))
                input_classes.append(cls)

        # --- Hidden space (right) ---
        ax_out = Axes(
            x_range=[-0.2, 1.2, 0.5],
            y_range=[-0.2, 1.2, 0.5],
            x_length=4,
            y_length=4,
            axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True, "tick_size": 0.06},
            tips=False,
        ).shift(RIGHT * 3.5)

        out_title = Text(
            "Hidden Space", font=HEADING_FONT, font_size=24, color=PRIMARY, weight=BOLD
        ).next_to(ax_out, UP, buff=0.3)

        h1_label = MathTex("h_1", color=TEXT, font_size=24).next_to(ax_out, DOWN, buff=0.3)
        h2_label = MathTex("h_2", color=TEXT, font_size=24).next_to(ax_out, LEFT, buff=0.3)

        # Compute hidden layer activations using a simple 2-neuron hidden layer
        # that roughly solves XOR: h1 = sigmoid(x1 + x2 - 0.5), h2 = sigmoid(-x1 - x2 + 1.5)
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        hidden_dots = VGroup()
        for i, (x, y) in enumerate(input_coords):
            h1 = sigmoid(10 * (x + y - 0.5))
            h2 = sigmoid(10 * (-x - y + 1.5))
            color = GREEN_C if input_classes[i] == 1 else RED_C
            dot = Dot(ax_out.c2p(h1, h2), radius=0.07, color=color, fill_opacity=0.85)
            hidden_dots.add(dot)

        # Arrow between spaces
        arrow = Arrow(
            ax_in.get_right() + RIGHT * 0.2,
            ax_out.get_left() + LEFT * 0.2,
            color=SECONDARY, stroke_width=3, buff=0.1,
        )
        arrow_label = MathTex(r"\sigma(W\mathbf{x} + \mathbf{b})", color=SECONDARY, font_size=22).next_to(arrow, UP, buff=0.15)

        # Separating line in hidden space
        sep_start = ax_out.c2p(-0.1, 0.5)
        sep_end = ax_out.c2p(1.1, 0.5)
        sep_line = DashedLine(sep_start, sep_end, color=SECONDARY, stroke_width=2.5)

        # --- Section 1: Show input space ---
        self.next_section("fold_input", skip_animations=False)
        self.play(Create(ax_in), Write(in_title), run_time=0.6)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in input_dots], lag_ratio=0.02),
            run_time=0.8,
        )
        self.wait(0.3)

        # --- Section 2: Show transformation ---
        self.next_section("fold_transform", skip_animations=False)
        self.play(Create(ax_out), Write(out_title), Write(h1_label), Write(h2_label), run_time=0.6)
        self.play(GrowArrow(arrow), Write(arrow_label), run_time=0.5)

        # Animate dots moving from input to hidden space
        hidden_dot_copies = VGroup()
        for i in range(len(input_dots)):
            d = input_dots[i].copy()
            hidden_dot_copies.add(d)

        self.play(
            *[
                hidden_dot_copies[i].animate.move_to(hidden_dots[i].get_center())
                for i in range(len(hidden_dots))
            ],
            run_time=1.5,
            rate_func=smooth,
        )
        self.wait(0.3)

        # --- Section 3: Show separating line ---
        self.next_section("fold_separate", skip_animations=False)
        sep_label = Text(
            "Now linearly separable!", font=BODY_FONT, font_size=22, color=GREEN_C
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(sep_line), Write(sep_label), run_time=0.6)
        self.wait(0.5)


class GradientDescentScene(Scene):
    """Animate gradient descent on a 1D loss curve.

    Sections:
      1. Show loss curve and starting point
      2. Show gradient (tangent) arrow
      3. Animate steps toward minimum
      4. Show convergence
    """

    def construct(self):
        self.camera.background_color = BG

        # Loss function: a slightly asymmetric bowl
        def loss_fn(x):
            return 0.15 * (x - 1) ** 2 + 0.05 * np.sin(2 * x) + 0.5

        axes = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 3, 0.5],
            x_length=9,
            y_length=5,
            axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True, "tick_size": 0.06},
            tips=False,
        ).shift(DOWN * 0.3)

        x_label = Text("weight", font=BODY_FONT, font_size=20, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.25)
        y_label = Text("loss", font=BODY_FONT, font_size=20, color=MUTED).next_to(axes.y_axis, LEFT, buff=0.25).shift(UP * 0.5)

        curve = axes.plot(loss_fn, x_range=[-2.5, 4.5], color=PRIMARY, stroke_width=3)

        title = Text(
            "Gradient Descent", font=HEADING_FONT, font_size=30, color=TEXT, weight=BOLD
        ).to_edge(UP, buff=0.3)

        # --- Section 1: Show curve and starting point ---
        self.next_section("gd_curve", skip_animations=False)
        self.play(Write(title), Create(axes), Write(x_label), Write(y_label), run_time=0.6)
        self.play(Create(curve), run_time=0.8)

        # Starting point
        x_val = 4.0
        ball = Dot(axes.c2p(x_val, loss_fn(x_val)), radius=0.12, color=SECONDARY, fill_opacity=1.0)
        self.play(GrowFromCenter(ball), run_time=0.4)
        self.wait(0.3)

        # --- Section 2: Show gradient arrow ---
        self.next_section("gd_gradient", skip_animations=False)

        # Approximate gradient
        eps = 0.01
        grad = (loss_fn(x_val + eps) - loss_fn(x_val - eps)) / (2 * eps)
        # Arrow pointing in negative gradient direction
        arrow_dx = -1.5 * np.sign(grad)
        grad_arrow = Arrow(
            axes.c2p(x_val, loss_fn(x_val)),
            axes.c2p(x_val + arrow_dx, loss_fn(x_val)),
            color=GREEN_C, stroke_width=3, buff=0,
        )
        grad_label = MathTex(r"-\eta \frac{\partial L}{\partial w}", color=GREEN_C, font_size=24).next_to(grad_arrow, UP, buff=0.15)

        self.play(GrowArrow(grad_arrow), Write(grad_label), run_time=0.5)
        self.wait(0.3)

        # --- Section 3: Animate steps ---
        self.next_section("gd_steps", skip_animations=False)

        self.play(FadeOut(grad_arrow), FadeOut(grad_label), run_time=0.3)

        lr = 0.5
        trajectory_dots = VGroup()
        path_lines = VGroup()

        for step in range(8):
            grad = (loss_fn(x_val + eps) - loss_fn(x_val - eps)) / (2 * eps)
            x_new = x_val - lr * grad
            x_new = np.clip(x_new, -2.5, 4.5)

            # Draw path line
            line = Line(
                axes.c2p(x_val, loss_fn(x_val)),
                axes.c2p(x_new, loss_fn(x_new)),
                color=SECONDARY, stroke_width=1.5, stroke_opacity=0.5,
            )
            path_lines.add(line)

            # Small dot at each step
            step_dot = Dot(
                axes.c2p(x_val, loss_fn(x_val)),
                radius=0.06, color=SECONDARY, fill_opacity=0.4,
            )
            trajectory_dots.add(step_dot)

            self.play(
                ball.animate.move_to(axes.c2p(x_new, loss_fn(x_new))),
                Create(line),
                FadeIn(step_dot),
                run_time=0.4,
            )

            x_val = x_new

        self.wait(0.3)

        # --- Section 4: Convergence ---
        self.next_section("gd_converged", skip_animations=False)

        converged = Text(
            "Converged to minimum", font=BODY_FONT, font_size=22, color=GREEN_C
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(converged), run_time=0.5)
        self.wait(0.5)


class LossLandscapeScene(ThreeDScene):
    """3D loss surface visualization showing sharp minima, flat minima,
    and a saddle point.

    Sections:
      1. Show 3D surface with rotation
      2. Label sharp minimum
      3. Label flat minimum
      4. Label saddle point
    """

    def construct(self):
        self.camera.background_color = BG

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-1, 4, 1],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"color": MUTED, "stroke_width": 1.5},
        )

        # Loss surface with interesting features:
        # - A sharp minimum near (-1.5, -1.5)
        # - A flat minimum near (1.5, 1.5)
        # - A saddle point near (0, 0)
        def loss_surface(u, v):
            # Saddle component
            saddle = 0.15 * (u ** 2 - v ** 2)
            # Sharp minimum
            r_sharp = np.sqrt((u + 1.5) ** 2 + (v + 1.5) ** 2)
            sharp = -1.5 * np.exp(-2 * r_sharp ** 2)
            # Flat minimum
            r_flat = np.sqrt((u - 1.5) ** 2 + (v - 1.5) ** 2)
            flat = -0.8 * np.exp(-0.3 * r_flat ** 2)
            # Base bowl
            base = 0.05 * (u ** 2 + v ** 2)
            return base + saddle + sharp + flat + 2.0

        surface = axes.plot_surface(
            loss_surface,
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(40, 40),
            colorscale=[PRIMARY, "#2a6bbf", SECONDARY, RED_C],
            fill_opacity=0.75,
        )

        title = Text(
            "Loss Landscape", font=HEADING_FONT, font_size=28, color=TEXT, weight=BOLD
        )
        title.to_corner(UL, buff=0.4)

        # --- Section 1: Show surface ---
        self.next_section("landscape_surface", skip_animations=False)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.3)
        self.play(Create(axes), run_time=0.5)
        self.play(Create(surface), run_time=1.5)
        self.wait(0.3)

        # --- Section 2: Label sharp minimum ---
        self.next_section("landscape_sharp", skip_animations=False)
        sharp_pos = axes.c2p(-1.5, -1.5, loss_surface(-1.5, -1.5))
        sharp_dot = Dot3D(sharp_pos, radius=0.08, color=RED_C)
        sharp_label = Text(
            "Sharp minimum", font=BODY_FONT, font_size=20, color=RED_C
        )
        sharp_label.to_corner(DR, buff=0.5)
        self.add_fixed_in_frame_mobjects(sharp_label)
        self.play(GrowFromCenter(sharp_dot), Write(sharp_label), run_time=0.5)
        self.wait(0.3)

        # --- Section 3: Label flat minimum ---
        self.next_section("landscape_flat", skip_animations=False)
        flat_pos = axes.c2p(1.5, 1.5, loss_surface(1.5, 1.5))
        flat_dot = Dot3D(flat_pos, radius=0.08, color=GREEN_C)
        flat_label = Text(
            "Flat minimum (generalizes better)", font=BODY_FONT, font_size=20, color=GREEN_C
        )
        flat_label.next_to(sharp_label, UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(flat_label)
        self.play(GrowFromCenter(flat_dot), Write(flat_label), run_time=0.5)
        self.wait(0.3)

        # --- Section 4: Label saddle point ---
        self.next_section("landscape_saddle", skip_animations=False)
        saddle_pos = axes.c2p(0, 0, loss_surface(0, 0))
        saddle_dot = Dot3D(saddle_pos, radius=0.08, color=SECONDARY)
        saddle_label = Text(
            "Saddle point", font=BODY_FONT, font_size=20, color=SECONDARY
        )
        saddle_label.next_to(flat_label, UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(saddle_label)
        self.play(GrowFromCenter(saddle_dot), Write(saddle_label), run_time=0.5)

        # Slow camera rotation to show the 3D structure
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)
        self.stop_ambient_camera_rotation()


class OverfittingCurveScene(Scene):
    """Training loss vs validation loss curves, showing the overfitting point.

    Sections:
      1. Show axes and legend
      2. Draw training loss curve
      3. Draw validation loss curve (diverges)
      4. Mark the overfitting region
    """

    def construct(self):
        self.camera.background_color = BG

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

        title = Text(
            "Overfitting", font=HEADING_FONT, font_size=30, color=TEXT, weight=BOLD
        ).to_edge(UP, buff=0.3)

        # Training loss: monotonically decreasing
        def train_loss(x):
            return 2.0 * np.exp(-0.04 * x) + 0.1

        # Validation loss: decreasing then increasing
        def val_loss(x):
            return 1.8 * np.exp(-0.06 * x) + 0.003 * (x - 30) ** 2 * (x > 30) + 0.3

        train_curve = axes.plot(train_loss, x_range=[1, 100], color=PRIMARY, stroke_width=3)
        val_curve = axes.plot(val_loss, x_range=[1, 100], color=RED_C, stroke_width=3)

        # Legend
        train_legend = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=PRIMARY, stroke_width=3),
            Text("Training loss", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.1)
        val_legend = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=RED_C, stroke_width=3),
            Text("Validation loss", font=BODY_FONT, font_size=18, color=TEXT),
        ).arrange(RIGHT, buff=0.1)
        legend = VGroup(train_legend, val_legend).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_corner(UR, buff=0.5)

        # --- Section 1: Show axes ---
        self.next_section("overfit_axes", skip_animations=False)
        self.play(Write(title), Create(axes), Write(x_label), Write(y_label), run_time=0.6)
        self.wait(0.2)

        # --- Section 2: Training loss ---
        self.next_section("overfit_train", skip_animations=False)
        self.play(Create(train_curve), FadeIn(train_legend), run_time=1.0)
        self.wait(0.3)

        # --- Section 3: Validation loss ---
        self.next_section("overfit_val", skip_animations=False)
        self.play(Create(val_curve), FadeIn(val_legend), run_time=1.0)
        self.wait(0.3)

        # --- Section 4: Mark overfitting region ---
        self.next_section("overfit_region", skip_animations=False)

        # Vertical dashed line at the sweet spot (~30 epochs)
        sweet_x = 30
        sweet_line = DashedLine(
            axes.c2p(sweet_x, 0), axes.c2p(sweet_x, 2.5),
            color=SECONDARY, stroke_width=2, dash_length=0.1,
        )
        sweet_label = Text(
            "Best model", font=BODY_FONT, font_size=18, color=SECONDARY
        ).next_to(sweet_line, UP, buff=0.1)

        # Shaded overfitting region
        overfit_region = Polygon(
            axes.c2p(sweet_x, 0), axes.c2p(100, 0),
            axes.c2p(100, 2.5), axes.c2p(sweet_x, 2.5),
            fill_color=RED_C, fill_opacity=0.08,
            stroke_width=0,
        )
        overfit_label = Text(
            "Overfitting", font=BODY_FONT, font_size=20, color=RED_C
        ).move_to(axes.c2p(65, 2.2))

        self.play(
            Create(sweet_line), Write(sweet_label),
            FadeIn(overfit_region), Write(overfit_label),
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
