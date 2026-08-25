"""Module 7 Manim animations: the GRPO loop and the pass@k crossover.

Two step-through scenes (each split into clips with self.next_section):
  GRPOScene (slug grpo):  prompt -> group of completions -> rewards ->
                          group-relative advantages -> policy update
  PassKScene (slug passk): axes -> base-model pass@k curve -> RL pass@k curve ->
                          the crossover where the base model overtakes at large k
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
# Scene 1: GRPOScene (slug grpo)
# ===========================================================================


class GRPOScene(StepScene):
    """One prompt, a group of completions, rewards, advantages, and the update."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("GRPO: One Step", "sample a group, score it, update toward the winners"))

        # ---- prompt ----
        self.next_section("prompt", skip_animations=False)
        prompt_box = RoundedRectangle(width=4.4, height=0.9, corner_radius=0.12,
                                      stroke_color=PRIMARY, fill_color=PRIMARY,
                                      fill_opacity=0.12, stroke_width=2.0)
        prompt_txt = label("reverse: cat", 26, TEXT)
        prompt_txt.move_to(prompt_box)
        prompt = VGroup(prompt_box, prompt_txt).move_to([0, 2.35, 0])
        self.play(FadeIn(prompt, shift=DOWN * 0.2), run_time=0.5)
        self.caption("One prompt. The verified answer is \"tac\".")

        # ---- group of G = 4 completions ----
        self.next_section("group", skip_animations=False)
        comps = ["tac", "tca", "tac", "cta"]
        xs = [-4.5, -1.5, 1.5, 4.5]
        boxes = VGroup()
        for c, x in zip(comps, xs):
            b = RoundedRectangle(width=2.2, height=0.9, corner_radius=0.12,
                                 stroke_color=MUTED, fill_color=MUTED,
                                 fill_opacity=0.10, stroke_width=2.0)
            t = label(c, 26, TEXT).move_to(b)
            boxes.add(VGroup(b, t).move_to([x, 0.7, 0]))
        arrows = VGroup(*[Arrow(prompt.get_bottom(), boxes[i].get_top(),
                                buff=0.12, stroke_width=3, color=MUTED,
                                max_tip_length_to_length_ratio=0.12)
                          for i in range(4)])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.15) for b in boxes], lag_ratio=0.12),
                  run_time=0.8)
        self.caption("Sample a group of G = 4 completions from the policy.")

        # ---- rewards ----
        self.next_section("reward", skip_animations=False)
        rewards = [1.0, 0.0, 1.0, 0.0]
        rmarks = VGroup()
        for i, r in enumerate(rewards):
            ok = r > 0.5
            sym = label("correct" if ok else "wrong", 18, GREEN if ok else RED, weight=BOLD)
            val = label(f"R = {r:.0f}", 20, GREEN if ok else RED)
            grp = VGroup(sym, val).arrange(DOWN, buff=0.10).next_to(boxes[i], DOWN, buff=0.22)
            rmarks.add(grp)
            boxes[i][0].set_stroke(GREEN if ok else RED)
        self.play(LaggedStart(*[FadeIn(m) for m in rmarks], lag_ratio=0.12), run_time=0.8)
        self.caption("Score each with the verifier: reward 1 if it reverses correctly, else 0.")

        # ---- advantages ----
        self.next_section("advantage", skip_animations=False)
        mean_txt = label("group mean = 0.5", 24, SECONDARY, weight=BOLD).move_to([0, -1.7, 0])
        self.play(FadeIn(mean_txt, shift=UP * 0.15), run_time=0.5)
        advs = ["+1", "-1", "+1", "-1"]
        amarks = VGroup()
        for i, a in enumerate(advs):
            col = GREEN if a.startswith("+") else RED
            t = label(f"A = {a}", 20, col, weight=BOLD).next_to(rmarks[i], DOWN, buff=0.20)
            amarks.add(t)
        self.play(LaggedStart(*[FadeIn(m) for m in amarks], lag_ratio=0.12), run_time=0.7)
        self.caption("Advantage = reward minus the group mean (then normalized).")

        # ---- update ----
        self.next_section("update", skip_animations=False)
        # Clear the gray sampling arrows so the colored update arrows read cleanly.
        self.play(FadeOut(arrows), run_time=0.3)
        ups = VGroup()
        for i, a in enumerate(advs):
            up = a.startswith("+")
            top = boxes[i].get_top()
            if up:
                arr = Arrow(top + UP * 0.08, top + UP * 0.62, buff=0.0, stroke_width=7,
                            color=GREEN, max_tip_length_to_length_ratio=0.4)
            else:
                arr = Arrow(top + UP * 0.62, top + UP * 0.08, buff=0.0, stroke_width=7,
                            color=RED, max_tip_length_to_length_ratio=0.4)
            ups.add(arr)
        self.play(LaggedStart(*[GrowArrow(a) for a in ups], lag_ratio=0.1), run_time=0.8)
        self.caption("Push up the probability of winners, push down the losers. One step.")
        self.wait(0.3)


# ===========================================================================
# Scene 2: PassKScene (slug passk)
# ===========================================================================


class PassKScene(StepScene):
    """pass@k: RL wins at small k, the base model catches up and overtakes at large k."""

    # plot region in scene coordinates
    X0, X1 = -5.2, 4.6
    Y0, Y1 = -2.2, 2.4

    def _pt(self, t, p):
        """Map t in [0,1] (k position) and p in [0,1] (pass@k) to scene coords."""
        x = self.X0 + t * (self.X1 - self.X0)
        y = self.Y0 + p * (self.Y1 - self.Y0)
        return np.array([x, y, 0.0])

    def _curve(self, pts, color):
        c = VMobject(stroke_color=color, stroke_width=5)
        c.set_points_smoothly([self._pt(t, p) for t, p in pts])
        return c

    def construct(self):
        self.setup_bg()
        self.add(title_bar("pass@k: Sharpening, Not Expanding",
                           "RL wins at small k; the base model overtakes at large k"))

        # ---- axes ----
        self.next_section("axes", skip_animations=False)
        x_axis = Line(self._pt(0, 0), self._pt(1, 0), stroke_color=LINE, stroke_width=3)
        y_axis = Line(self._pt(0, 0), self._pt(0, 1), stroke_color=LINE, stroke_width=3)
        x_lab = label("k  (samples per problem, log scale)", 20, MUTED)
        x_lab.next_to(x_axis, DOWN, buff=0.30)
        y_lab = label("pass@k", 20, MUTED).rotate(PI / 2)
        y_lab.next_to(y_axis, LEFT, buff=0.30)
        k1 = label("k = 1", 18, MUTED).next_to(self._pt(0, 0), DOWN, buff=0.30)
        kbig = label("large k", 18, MUTED).next_to(self._pt(1, 0), DOWN, buff=0.30)
        self.play(Create(x_axis), Create(y_axis), run_time=0.6)
        self.play(FadeIn(x_lab), FadeIn(y_lab), FadeIn(k1), FadeIn(kbig), run_time=0.5)
        self.caption("pass@k: does any of k samples solve the problem? Coverage, not the top guess.")

        # ---- base-model curve ----
        self.next_section("base", skip_animations=False)
        base_pts = [(0.0, 0.16), (0.25, 0.34), (0.5, 0.56), (0.75, 0.78), (1.0, 0.93)]
        base = self._curve(base_pts, MUTED)
        base_lab = label("base model", 22, MUTED, weight=BOLD)
        base_lab.next_to(self._pt(1.0, 0.93), UP, buff=0.18).shift(LEFT * 0.6)
        self.play(Create(base), run_time=1.0)
        self.play(FadeIn(base_lab), run_time=0.4)
        self.caption("The base model starts low at k = 1 but keeps climbing with more samples.")

        # ---- RL curve ----
        self.next_section("rl", skip_animations=False)
        rl_pts = [(0.0, 0.50), (0.25, 0.63), (0.5, 0.69), (0.75, 0.71), (1.0, 0.72)]
        rl = self._curve(rl_pts, PRIMARY)
        rl_lab = label("RL-trained", 22, PRIMARY, weight=BOLD)
        rl_lab.next_to(self._pt(0.0, 0.50), UP, buff=0.20).shift(RIGHT * 0.7)
        self.play(Create(rl), run_time=1.0)
        self.play(FadeIn(rl_lab), run_time=0.4)
        self.caption("RL wins big at k = 1 (better top guess) but plateaus fast.")

        # ---- crossover ----
        self.next_section("cross", skip_animations=False)
        cx = self._pt(0.63, 0.705)
        dot = Dot(cx, radius=0.09, color=SECONDARY)
        ring = Circle(radius=0.28, stroke_color=SECONDARY, stroke_width=3).move_to(cx)
        cross_lab = label("base overtakes here", 20, SECONDARY, weight=BOLD)
        cross_lab.next_to(cx, RIGHT, buff=0.25).shift(DOWN * 0.35)
        self.play(FadeIn(dot), Create(ring), run_time=0.6)
        self.play(FadeIn(cross_lab), run_time=0.4)
        self.caption("Same coverage was already in the base model. RL sharpened it, did not expand it.")
        self.wait(0.3)


# ===========================================================================
# Scene 3: REINFORCEScene (slug reinforce)
# ===========================================================================


class REINFORCEScene(StepScene):
    """The policy-gradient core: scale each completion's log-prob gradient by its
    reward. Every reward here is positive, so every arrow points UP — the
    high-variance failure mode the baseline (next slide) repairs."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("REINFORCE: Reward-Weighted Gradient",
                           "scale each log-prob gradient by its reward"))

        comps = ["tac", "tca", "act", "cat"]
        rewards = [1.0, 0.3, 0.6, 0.1]
        xs = [-4.5, -1.5, 1.5, 4.5]

        # ---- sample a few completions ----
        self.next_section("samples", skip_animations=False)
        prompt_box = RoundedRectangle(width=4.0, height=0.8, corner_radius=0.12,
                                      stroke_color=PRIMARY, fill_color=PRIMARY,
                                      fill_opacity=0.12, stroke_width=2.0)
        prompt_txt = label("reverse: cat", 24, TEXT).move_to(prompt_box)
        prompt = VGroup(prompt_box, prompt_txt).move_to([0, 2.35, 0])
        boxes = VGroup()
        for c, x in zip(comps, xs):
            b = RoundedRectangle(width=2.0, height=0.8, corner_radius=0.12,
                                 stroke_color=MUTED, fill_color=MUTED,
                                 fill_opacity=0.10, stroke_width=2.0)
            t = label(c, 24, TEXT).move_to(b)
            boxes.add(VGroup(b, t).move_to([x, 0.85, 0]))
        arrows = VGroup(*[Arrow(prompt.get_bottom(), boxes[i].get_top(), buff=0.12,
                                stroke_width=3, color=MUTED,
                                max_tip_length_to_length_ratio=0.12) for i in range(4)])
        self.play(FadeIn(prompt, shift=DOWN * 0.2), run_time=0.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.15) for b in boxes], lag_ratio=0.12),
                  run_time=0.7)
        self.caption("Sample completions from the policy. No labels, no correct token to copy.")

        # ---- scalar rewards ----
        self.next_section("reward", skip_animations=False)
        rmarks = VGroup()
        for i, r in enumerate(rewards):
            col = GREEN if r >= 0.5 else SECONDARY
            t = label(f"R = {r:.1f}", 22, col, weight=BOLD).next_to(boxes[i], DOWN, buff=0.28)
            rmarks.add(t)
        self.play(LaggedStart(*[FadeIn(m) for m in rmarks], lag_ratio=0.12), run_time=0.7)
        self.caption("Each earns a scalar reward — any number, not just 0 or 1.")

        # ---- reward-weighted gradient: every arrow points UP ----
        self.next_section("weight", skip_animations=False)
        self.play(FadeOut(arrows), run_time=0.3)
        ups = VGroup()
        for i, r in enumerate(rewards):
            top = boxes[i].get_top()
            length = 0.30 + 1.05 * r
            arr = Arrow(top + UP * 0.08, top + UP * (0.08 + length), buff=0.0,
                        stroke_width=7, color=GREEN,
                        max_tip_length_to_length_ratio=0.30)
            ups.add(arr)
        self.play(LaggedStart(*[GrowArrow(a) for a in ups], lag_ratio=0.1), run_time=0.8)
        self.caption("Push every completion UP, by an amount proportional to its reward.")

        # ---- the estimator and the catch ----
        self.next_section("estimator", skip_animations=False)
        formula = label("update  =  reward  x  (gradient of log-probability)", 24, TEXT)
        formula.move_to([0, -2.0, 0])
        self.play(Write(formula), run_time=0.8)
        self.caption("Correct, but high-variance: even mediocre samples get pushed up. The baseline fixes this.")
        self.wait(0.3)


# ===========================================================================
# Scene 4: PPOScene (slug ppo)
# ===========================================================================


class PPOScene(StepScene):
    """Actor-critic with a clipped step: a learned value network supplies the
    baseline (advantage = reward minus value), and importance-ratio clipping
    bounds how far the policy can move in one update."""

    def _box(self, w, h, color, txt, sub=None):
        b = RoundedRectangle(width=w, height=h, corner_radius=0.12,
                             stroke_color=color, fill_color=color,
                             fill_opacity=0.12, stroke_width=2.0)
        t = label(txt, 22, TEXT, weight=BOLD).move_to(b)
        grp = VGroup(b, t)
        if sub:
            s = label(sub, 16, MUTED).next_to(t, DOWN, buff=0.10)
            grp.add(s)
        return grp

    def construct(self):
        self.setup_bg()
        self.add(title_bar("PPO: Actor, Critic, and a Clipped Step",
                           "a value network gives the baseline; clipping bounds the update"))

        # ---- actor produces a completion, scored by the reward model ----
        self.next_section("actor", skip_animations=False)
        actor = self._box(3.0, 1.0, PRIMARY, "Actor", "policy pi_theta").move_to([-4.2, 1.5, 0])
        comp = self._box(2.4, 0.9, MUTED, "completion").move_to([-0.3, 1.5, 0])
        rbox = label("R = 0.8", 24, GREEN, weight=BOLD).move_to([3.7, 1.5, 0])
        a1 = Arrow(actor.get_right(), comp.get_left(), buff=0.15, stroke_width=4, color=MUTED)
        a2 = Arrow(comp.get_right(), rbox.get_left(), buff=0.18, stroke_width=4, color=MUTED)
        self.play(FadeIn(actor, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(a1), FadeIn(comp), run_time=0.5)
        self.play(GrowArrow(a2), FadeIn(rbox), run_time=0.5)
        self.caption("The actor samples a completion; the reward model scores it: R = 0.8.")

        # ---- critic estimates the value (the baseline) ----
        self.next_section("critic", skip_animations=False)
        critic = self._box(3.0, 1.0, SECONDARY, "Critic", "value net V").move_to([-4.2, -0.4, 0])
        vbox = label("V = 0.5", 24, SECONDARY, weight=BOLD).move_to([-1.5, -0.4, 0])
        a3 = Arrow(critic.get_right(), vbox.get_left(), buff=0.18, stroke_width=4, color=MUTED)
        self.play(FadeIn(critic, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(a3), FadeIn(vbox), run_time=0.5)
        self.caption("A separate value network predicts the expected reward: the learned baseline.")

        # ---- advantage = reward minus value ----
        self.next_section("advantage", skip_animations=False)
        adv = label("Advantage  A = R - V = +0.3", 24, GREEN, weight=BOLD).move_to([3.1, -0.4, 0])
        self.play(Write(adv), run_time=0.7)
        self.caption("Advantage = reward minus the critic's baseline. Positive: better than expected.")

        # ---- clipped step: ratio clamped into a trust region ----
        self.next_section("clip", skip_animations=False)
        x0, x1, y = -4.6, 4.6, -2.1
        line = Line([x0, y, 0], [x1, y, 0], stroke_color=LINE, stroke_width=3)

        def rx(r):  # map ratio in [0.6, 1.5] to x
            return x0 + (r - 0.6) / (1.5 - 0.6) * (x1 - x0)
        band = Rectangle(width=rx(1.2) - rx(0.8), height=0.34,
                         stroke_width=0, fill_color=GREEN, fill_opacity=0.18)
        band.move_to([(rx(0.8) + rx(1.2)) / 2, y, 0])
        ticks = VGroup()
        for r, lab in [(0.8, "1-eps"), (1.0, "1"), (1.2, "1+eps")]:
            tk = Line([rx(r), y - 0.12, 0], [rx(r), y + 0.12, 0], stroke_color=MUTED, stroke_width=3)
            tl = label(lab, 16, MUTED).next_to(tk, DOWN, buff=0.12)
            ticks.add(VGroup(tk, tl))
        rlbl = label("policy / old-policy ratio", 18, MUTED).next_to(line, UP, buff=0.10).shift(LEFT * 3.2)
        self.play(Create(line), FadeIn(band), FadeIn(ticks), FadeIn(rlbl), run_time=0.8)
        dot = Dot([rx(1.0), y, 0], radius=0.10, color=PRIMARY)
        self.play(FadeIn(dot), run_time=0.3)
        want = Arrow([rx(1.0), y + 0.55, 0], [rx(1.5), y + 0.55, 0], buff=0.0,
                     stroke_width=5, color=RED, max_tip_length_to_length_ratio=0.25)
        self.play(GrowArrow(want), dot.animate.move_to([rx(1.5), y, 0]), run_time=0.7)
        self.play(dot.animate.move_to([rx(1.2), y, 0]), run_time=0.6)
        self.caption("The update wants a big step; clipping clamps the ratio into [1-eps, 1+eps]. One safe step.")
        self.wait(0.3)


# ===========================================================================
# Scene 5: DPOScene (slug dpo)
# ===========================================================================


class DPOScene(StepScene):
    """No RL loop: a fixed chosen/rejected pair, an implicit reward read off the
    policy's log-prob ratio to a frozen reference, and a logistic loss that
    widens the margin between winner and loser. Pure SFT-style gradient descent."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("DPO: Widen the Margin, No Sampling",
                           "a fixed pair, an implicit reward, a classification loss"))

        # ---- a fixed preference pair (no sampling) ----
        self.next_section("pair", skip_animations=False)
        prompt = label("prompt: \"reverse: cat\"", 22, MUTED).move_to([0, 2.6, 0])
        win = RoundedRectangle(width=4.6, height=0.95, corner_radius=0.12,
                               stroke_color=GREEN, fill_color=GREEN,
                               fill_opacity=0.12, stroke_width=2.0).move_to([0, 1.35, 0])
        win_t = label("chosen   y_w  =  \"tac\"", 22, TEXT).move_to(win)
        lose = RoundedRectangle(width=4.6, height=0.95, corner_radius=0.12,
                                stroke_color=RED, fill_color=RED,
                                fill_opacity=0.12, stroke_width=2.0).move_to([0, 0.1, 0])
        lose_t = label("rejected   y_l  =  \"cta\"", 22, TEXT).move_to(lose)
        self.play(FadeIn(prompt), run_time=0.4)
        self.play(FadeIn(VGroup(win, win_t), shift=DOWN * 0.1),
                  FadeIn(VGroup(lose, lose_t), shift=DOWN * 0.1), run_time=0.7)
        self.caption("A fixed pair from a dataset: one preferred, one rejected. Nothing is sampled.")

        # ---- implicit reward = beta * log (policy / reference) ----
        self.next_section("reward", skip_animations=False)
        rw = label("r_w = +0.4", 22, GREEN, weight=BOLD).next_to(win, RIGHT, buff=0.5)
        rl = label("r_l = +0.1", 22, RED, weight=BOLD).next_to(lose, RIGHT, buff=0.5)
        formula = label("implicit reward   r = beta log( pi_theta / pi_ref )", 22, TEXT)
        formula.move_to([0, -1.35, 0])
        self.play(FadeIn(rw), FadeIn(rl), run_time=0.6)
        self.play(Write(formula), run_time=0.8)
        self.caption("The reward is implicit: the policy's log-prob ratio to a frozen reference. No reward model.")

        # ---- the margin and the logistic loss ----
        self.next_section("margin", skip_animations=False)
        brace = BraceBetweenPoints(lose.get_left() + LEFT * 0.2, win.get_left() + LEFT * 0.2,
                                   direction=LEFT, color=SECONDARY)
        mlab = label("margin  r_w - r_l", 18, SECONDARY, weight=BOLD).next_to(brace, LEFT, buff=0.15)
        loss = label("loss  =  - log sigma( beta ( r_w - r_l ) )", 22, TEXT).move_to([0, -2.15, 0])
        self.play(GrowFromCenter(brace), FadeIn(mlab), run_time=0.6)
        self.play(Write(loss), run_time=0.8)
        self.caption("A logistic loss rewards a larger margin: chosen above rejected. This is SFT-style descent.")

        # ---- the update widens the gap ----
        self.next_section("update", skip_animations=False)
        up = Arrow(win.get_top() + UP * 0.05, win.get_top() + UP * 0.5, buff=0.0,
                   stroke_width=7, color=GREEN, max_tip_length_to_length_ratio=0.45)
        down = Arrow(lose.get_bottom() + DOWN * 0.05, lose.get_bottom() + DOWN * 0.5, buff=0.0,
                     stroke_width=7, color=RED, max_tip_length_to_length_ratio=0.45)
        self.play(GrowArrow(up), GrowArrow(down), run_time=0.7)
        self.caption("Raise the chosen, lower the rejected. No sampling, no critic, no RL loop.")
        self.wait(0.3)
