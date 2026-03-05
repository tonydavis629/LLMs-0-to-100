"""
Module 1: Course Introduction - Manim Animations
Scenes covering AI history, Shannon's prediction, n-gram models, and compression.
"""

from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLD_COLOR = BLUE_C          # AI winter
WARM_COLOR = ORANGE           # AI summer
ERA_GOFAI = BLUE_D
ERA_NEURAL = GOLD_C
ERA_DEEP = GREEN_C
ERA_LLM = RED_C
ERA_RL = PURPLE_C


class AITimelineScene(Scene):
    """Animated timeline of key milestones in AI / LLM history.
    Items are spaced proportionally to actual year differences."""

    def construct(self):
        self.next_section("title")
        # ---- title ----
        title = Text("Key Milestones in AI & LLMs", font_size=40, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)

        # ---- timeline data (year, label, color) ----
        milestones = [
            (1948, "Information\nTheory", WARM_COLOR),
            (1965, "GOFAI\nEra", COLD_COLOR),
            (1986, "Back-\npropagation", WARM_COLOR),
            (2012, "AlexNet\n+ GPUs", ERA_DEEP),
            (2017, "Transformers\n(Attention)", ERA_LLM),
            (2018, "BERT /\nGPT", ERA_LLM),
            (2022, "InstructGPT /\nChatGPT", ERA_LLM),
            (2024, "RL Post-\nTraining", ERA_RL),
        ]

        self.next_section("axis")
        # ---- horizontal axis ----
        line_left = LEFT * 6.2
        line_right = RIGHT * 6.2
        axis = Line(line_left, line_right, color=GREY_B, stroke_width=3)
        axis.shift(DOWN * 0.3)
        self.play(Create(axis), run_time=0.8)

        # ---- compute proportional x positions ----
        year_min = milestones[0][0]   # 1948
        year_max = milestones[-1][0]  # 2024
        year_span = year_max - year_min  # 76 years
        x_left = line_left[0]
        x_right = line_right[0]
        x_span = x_right - x_left

        def year_to_x(year):
            return x_left + (year - year_min) / year_span * x_span

        # ---- year tick labels for axis ----
        year_display = ["1948", "~1965", "1986", "2012", "2017", "2018", "2022", "2024"]

        # ---- dots and labels ----
        dots = VGroup()
        year_labels = VGroup()
        desc_labels = VGroup()

        for i, (year, desc, color) in enumerate(milestones):
            x = year_to_x(year)
            pos = np.array([x, axis.get_center()[1], 0])

            dot = Dot(pos, radius=0.1, color=color)
            dots.add(dot)

            y_label = Text(year_display[i], font_size=20, color=color)
            y_label.next_to(dot, DOWN, buff=0.25)
            year_labels.add(y_label)

            d_label = Text(desc, font_size=16, color=color, line_spacing=0.3)
            # alternate above / below for readability
            if i % 2 == 0:
                d_label.next_to(dot, UP, buff=0.3)
            else:
                d_label.next_to(dot, UP, buff=0.9)
            desc_labels.add(d_label)

        # Animate milestones in one by one
        milestone_names = [
            "milestone_information_theory",
            "milestone_gofai_era",
            "milestone_backpropagation",
            "milestone_alexnet_gpus",
            "milestone_transformers",
            "milestone_bert_gpt",
            "milestone_instructgpt_chatgpt",
            "milestone_rl_post_training",
        ]
        for i in range(len(milestones)):
            self.next_section(milestone_names[i])
            self.play(
                GrowFromCenter(dots[i]),
                FadeIn(year_labels[i], shift=DOWN * 0.2),
                FadeIn(desc_labels[i], shift=UP * 0.2),
                run_time=0.6,
            )

        self.wait(0.5)

        self.next_section("highlight_ai_winter")
        # ---- highlight AI winter region (around GOFAI dot) ----
        winter_cx = dots[1].get_center()[0]
        winter_rect = Rectangle(
            width=1.2,
            height=3.2,
            color=BLUE_E,
            fill_color=BLUE_E,
            fill_opacity=0.15,
            stroke_width=1,
        ).move_to(
            np.array([winter_cx, axis.get_center()[1] + 0.5, 0])
        )
        winter_label = Text("AI Winter", font_size=16, color=BLUE_E)
        winter_label.next_to(winter_rect, DOWN, buff=0.05)
        self.play(FadeIn(winter_rect), FadeIn(winter_label), run_time=0.6)

        self.next_section("highlight_ai_summer")
        # ---- highlight AI summer region (2017-2024) ----
        summer_left = dots[4].get_center()[0] - 0.3
        summer_right = dots[7].get_center()[0] + 0.3
        summer_rect = Rectangle(
            width=summer_right - summer_left + 0.4,
            height=3.2,
            color=ORANGE,
            fill_color=ORANGE,
            fill_opacity=0.12,
            stroke_width=1,
        ).move_to(
            np.array([(summer_left + summer_right) / 2, axis.get_center()[1] + 0.5, 0])
        )
        summer_label = Text("AI Summer", font_size=16, color=ORANGE)
        summer_label.next_to(summer_rect, DOWN, buff=0.05)
        self.play(FadeIn(summer_rect), FadeIn(summer_label), run_time=0.6)

        self.wait(2)


class ShannonPredictionScene(Scene):
    """Shannon's character-level prediction experiment.
    Layout fixed so bar charts and text do not overlap."""

    def construct(self):
        self.next_section("title")
        # ---- title ----
        title = Text("Shannon's Prediction Game", font_size=38, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        self.next_section("partial_sentence")
        # ---- partial sentence ----
        sentence_text = "THE NEXT LETTER IS PROBABLY _"
        sentence = Text(sentence_text, font_size=28, font="Monospace")
        sentence.next_to(title, DOWN, buff=0.35)
        self.play(Write(sentence), run_time=1.2)
        self.wait(0.3)

        self.next_section("context_aware_distribution")
        # ---- context-aware probability bars ----
        subtitle_ctx = Text("With context (English statistics):", font_size=20, color=YELLOW)
        subtitle_ctx.next_to(sentence, DOWN, buff=0.35).align_to(sentence, LEFT)
        self.play(FadeIn(subtitle_ctx), run_time=0.4)

        chars_ctx = ["' '", "E", "A", "T", "S", "N", "other"]
        probs_ctx = [0.22, 0.18, 0.12, 0.10, 0.08, 0.07, 0.23]
        bar_ctx = self._make_bar_chart(
            chars_ctx, probs_ctx, GREEN_C, max_bar_width=3.5
        )
        bar_ctx.next_to(subtitle_ctx, DOWN, buff=0.15).align_to(subtitle_ctx, LEFT)
        self.play(
            LaggedStart(*[FadeIn(row) for row in bar_ctx], lag_ratio=0.10),
            run_time=1.0,
        )
        self.wait(0.4)

        self.next_section("highlight_most_likely")
        # ---- highlight space and E ----
        highlight_rect = SurroundingRectangle(
            VGroup(bar_ctx[0], bar_ctx[1]), color=YELLOW, buff=0.06
        )
        note = Text("Most likely!", font_size=18, color=YELLOW)
        note.next_to(highlight_rect, RIGHT, buff=0.2)
        self.play(Create(highlight_rect), FadeIn(note), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(highlight_rect), FadeOut(note), run_time=0.3)

        self.next_section("uniform_distribution")
        # ---- uniform distribution for contrast ----
        subtitle_uni = Text("Without context (uniform):", font_size=20, color=RED_C)
        subtitle_uni.next_to(bar_ctx, DOWN, buff=0.3).align_to(subtitle_ctx, LEFT)
        self.play(FadeIn(subtitle_uni), run_time=0.4)

        n_chars = 27  # 26 letters + space
        uniform_prob = 1.0 / n_chars
        chars_uni = ["' '", "E", "A", "T", "S", "N", "other"]
        probs_uni = [uniform_prob] * len(chars_uni)
        bar_uni = self._make_bar_chart(
            chars_uni, probs_uni, RED_C, max_bar_width=3.5
        )
        bar_uni.next_to(subtitle_uni, DOWN, buff=0.15).align_to(subtitle_uni, LEFT)
        self.play(
            LaggedStart(*[FadeIn(row) for row in bar_uni], lag_ratio=0.10),
            run_time=0.8,
        )
        self.wait(0.4)

        self.next_section("punchline_context_matters")
        # ---- punchline ----
        punchline = Text(
            "Context makes prediction MUCH easier!",
            font_size=26,
            color=YELLOW,
            weight=BOLD,
        )
        punchline.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(punchline, shift=UP * 0.3), run_time=0.8)
        self.wait(2)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_bar_chart(labels, probs, color, max_bar_width=3.5):
        """Return a VGroup of rows, each row = label + bar + pct text."""
        rows = VGroup()
        max_p = max(probs) if max(probs) > 0 else 1
        for lbl, p in zip(labels, probs):
            label_mob = Text(lbl, font_size=16, font="Monospace")
            label_mob.set_width(min(label_mob.width, 0.6))
            bar_width = max(0.05, (p / max_p) * max_bar_width)
            bar = Rectangle(
                width=bar_width,
                height=0.18,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=0,
            )
            pct = Text(f"{p:.0%}", font_size=14, color=GREY_A)
            # Fixed-width label column: align all labels to the same right edge
            label_mob.align_to(ORIGIN, RIGHT)
            bar.next_to(label_mob, RIGHT, buff=0.15)
            bar.align_to(label_mob, DOWN)
            pct.next_to(bar, RIGHT, buff=0.1)
            row = VGroup(label_mob, bar, pct)
            rows.add(row)
        rows.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        return rows


class NgramOrderScene(Scene):
    """Show how increasing n-gram order improves generated text quality.
    Arrow goes vertically DOWN along the right side."""

    def construct(self):
        self.next_section("title")
        title = Text("N-gram Language Models", font_size=38, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        samples = [
            (
                "Order 0 (Uniform Random)",
                "xqj bz fmk ort wc aelp gh",
                RED_C,
            ),
            (
                "Order 1 (Unigram)",
                "e tah oin sr dlcu mfpg ywb",
                ORANGE,
            ),
            (
                "Order 2 (Bigram)",
                "th an in he re ou at on er",
                YELLOW,
            ),
            (
                "Order 3 (Trigram)",
                "the ing and ion tio for ent",
                GREEN_C,
            ),
        ]

        prev_group = title
        groups = []

        ngram_section_names = [
            "order_0_uniform_random",
            "order_1_unigram",
            "order_2_bigram",
            "order_3_trigram",
        ]
        for order_label, sample_text, color in samples:
            idx = samples.index((order_label, sample_text, color))
            self.next_section(ngram_section_names[idx])
            label = Text(order_label, font_size=26, color=color, weight=BOLD)
            label.next_to(prev_group, DOWN, buff=0.55).align_to(LEFT * 5.5, LEFT)

            sample = Text(
                sample_text,
                font_size=22,
                font="Monospace",
                color=WHITE,
            )
            sample.next_to(label, DOWN, buff=0.15).align_to(label, LEFT)

            # quality indicator bar
            order_idx = samples.index((order_label, sample_text, color))
            bar_width = 1.0 + order_idx * 1.5
            bar = Rectangle(
                width=bar_width,
                height=0.15,
                fill_color=color,
                fill_opacity=0.7,
                stroke_width=0,
            )
            quality_label = Text("quality", font_size=14, color=GREY_B)
            bar.next_to(sample, DOWN, buff=0.15).align_to(sample, LEFT)
            quality_label.next_to(bar, RIGHT, buff=0.15)

            group = VGroup(label, sample, bar, quality_label)
            groups.append(group)

            self.play(
                FadeIn(label, shift=RIGHT * 0.3),
                run_time=0.5,
            )
            self.play(
                Write(sample),
                GrowFromEdge(bar, LEFT),
                FadeIn(quality_label),
                run_time=0.8,
            )
            self.wait(0.3)
            prev_group = group

        self.next_section("improvement_arrow")
        # arrow showing improvement: vertical DOWN along right side
        arrow_x = 6.0
        arrow = Arrow(
            np.array([arrow_x, groups[0][0].get_center()[1], 0]),
            np.array([arrow_x, groups[-1][-1].get_bottom()[1], 0]),
            color=GREEN_A,
            stroke_width=4,
        )
        arrow_label = Text("Better", font_size=20, color=GREEN_A)
        arrow_label.next_to(arrow, RIGHT, buff=0.15)
        self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=0.7)

        self.wait(2)


class LanguageAsCompressionScene(Scene):
    """Visualize Shannon entropy and how better models compress text more."""

    def construct(self):
        self.next_section("title")
        title = Text("Language Modeling as Compression", font_size=36, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        self.next_section("sample_text")
        # ---- sample text ----
        raw_text = "the cat sat on the mat"
        text_mob = Text(raw_text, font_size=30, font="Monospace")
        text_mob.next_to(title, DOWN, buff=0.5)
        self.play(Write(text_mob), run_time=0.8)
        self.wait(0.3)

        self.next_section("uniform_encoding")
        # ---- Step 1: uniform encoding ----
        step1_label = Text(
            "Uniform encoding: ~4.75 bits / char (log2 27)",
            font_size=22,
            color=RED_C,
        )
        step1_label.next_to(text_mob, DOWN, buff=0.6)
        self.play(FadeIn(step1_label), run_time=0.5)

        # Build character blocks -- all same width for uniform
        chars = list(raw_text)
        block_width = 0.30
        uniform_blocks = VGroup()
        for ch in chars:
            rect = Rectangle(
                width=block_width,
                height=0.45,
                fill_color=RED_C,
                fill_opacity=0.5,
                stroke_color=RED_C,
                stroke_width=1,
            )
            char_label = Text(ch, font_size=16, font="Monospace")
            char_label.move_to(rect)
            block = VGroup(rect, char_label)
            uniform_blocks.add(block)
        uniform_blocks.arrange(RIGHT, buff=0.02)
        uniform_blocks.next_to(step1_label, DOWN, buff=0.3)
        self.play(
            LaggedStart(*[FadeIn(b) for b in uniform_blocks], lag_ratio=0.04),
            run_time=1.0,
        )
        self.wait(0.5)

        self.next_section("entropy_aware_encoding")
        # ---- Step 2: entropy-aware encoding ----
        step2_label = Text(
            "Entropy coding: common chars need fewer bits",
            font_size=22,
            color=GREEN_C,
        )
        step2_label.next_to(uniform_blocks, DOWN, buff=0.5)
        self.play(FadeIn(step2_label), run_time=0.5)

        # Approximate bits per character based on English frequency
        freq_bits = {
            " ": 2.5,
            "e": 3.0,
            "t": 3.2,
            "a": 3.4,
            "h": 3.8,
            "o": 3.8,
            "n": 3.9,
            "s": 4.0,
            "c": 4.8,
            "m": 4.8,
        }
        max_bits = 4.8
        compressed_blocks = VGroup()
        for ch in chars:
            bits = freq_bits.get(ch, 4.5)
            w = block_width * (bits / max_bits)
            w = max(w, 0.12)
            rect = Rectangle(
                width=w,
                height=0.45,
                fill_color=GREEN_C,
                fill_opacity=0.5,
                stroke_color=GREEN_C,
                stroke_width=1,
            )
            char_label = Text(ch, font_size=16, font="Monospace")
            char_label.move_to(rect)
            block = VGroup(rect, char_label)
            compressed_blocks.add(block)
        compressed_blocks.arrange(RIGHT, buff=0.02)
        compressed_blocks.next_to(step2_label, DOWN, buff=0.3)

        self.play(
            LaggedStart(*[FadeIn(b) for b in compressed_blocks], lag_ratio=0.04),
            run_time=1.0,
        )
        self.wait(0.5)

        self.next_section("total_width_comparison")
        # ---- show total width comparison ----
        brace_u = Brace(uniform_blocks, DOWN, color=RED_C)
        brace_u_label = Text("~104 bits", font_size=18, color=RED_C)
        brace_u_label.next_to(brace_u, DOWN, buff=0.1)

        brace_c = Brace(compressed_blocks, DOWN, color=GREEN_C)
        brace_c_label = Text("~74 bits", font_size=18, color=GREEN_C)
        brace_c_label.next_to(brace_c, DOWN, buff=0.1)

        self.play(
            GrowFromCenter(brace_u), FadeIn(brace_u_label),
            GrowFromCenter(brace_c), FadeIn(brace_c_label),
            run_time=0.7,
        )
        self.wait(0.5)

        self.next_section("cross_entropy_comparison")
        # ---- Step 3: cross-entropy decreasing with model order ----
        ce_title = Text(
            "Cross-entropy decreases with better models:",
            font_size=22,
            color=YELLOW,
        )
        # Move everything up to make room
        everything_above = VGroup(
            text_mob, step1_label, uniform_blocks, brace_u, brace_u_label,
            step2_label, compressed_blocks, brace_c, brace_c_label,
        )
        self.play(
            FadeOut(everything_above),
            run_time=0.6,
        )

        ce_title.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(ce_title), run_time=0.4)

        orders = ["Uniform\n(4.75)", "Unigram\n(4.14)", "Bigram\n(3.56)", "Trigram\n(2.77)"]
        values = [4.75, 4.14, 3.56, 2.77]
        colors_ce = [RED_C, ORANGE, YELLOW, GREEN_C]
        max_val = 5.0
        bar_max_h = 2.8

        bars = VGroup()
        bar_labels = VGroup()
        for i, (label_text, val, col) in enumerate(zip(orders, values, colors_ce)):
            h = (val / max_val) * bar_max_h
            bar = Rectangle(
                width=1.2,
                height=h,
                fill_color=col,
                fill_opacity=0.7,
                stroke_color=col,
                stroke_width=2,
            )
            lbl = Text(label_text, font_size=18, color=col)
            bars.add(bar)
            bar_labels.add(lbl)

        bars.arrange(RIGHT, buff=0.6, aligned_edge=DOWN)
        bars.next_to(ce_title, DOWN, buff=0.6)

        for i in range(len(bars)):
            bar_labels[i].next_to(bars[i], DOWN, buff=0.2)

        self.play(
            LaggedStart(
                *[GrowFromEdge(b, DOWN) for b in bars],
                lag_ratio=0.2,
            ),
            LaggedStart(
                *[FadeIn(l) for l in bar_labels],
                lag_ratio=0.2,
            ),
            run_time=2.0,
        )

        self.next_section("better_compression_arrow")
        # downward arrow across bars
        arrow = Arrow(
            bars[0].get_top() + UP * 0.3,
            bars[-1].get_top() + UP * 0.3,
            color=GREEN_A,
            stroke_width=4,
        )
        arrow_text = Text("Better compression", font_size=20, color=GREEN_A)
        arrow_text.next_to(arrow, UP, buff=0.15)
        self.play(GrowArrow(arrow), FadeIn(arrow_text), run_time=0.7)

        self.wait(2)


class GOFAIDiagramScene(Scene):
    """Diagram contrasting GOFAI (hand-written rules) with neural approaches.
    Left side: rule-based flowchart that fails on ambiguity.
    Right side: neural network black box that learns from data."""

    def construct(self):
        self.next_section("title")
        title = Text("GOFAI vs. Learned Models", font_size=38, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # =====================================================================
        # LEFT SIDE: GOFAI rule-based system
        # =====================================================================
        self.next_section("gofai_input")
        left_title = Text("Rule-Based (GOFAI)", font_size=24, color=BLUE_C, weight=BOLD)
        left_title.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(FadeIn(left_title), run_time=0.4)

        # Input box
        input_box = RoundedRectangle(
            width=2.8, height=0.7, corner_radius=0.15,
            color=WHITE, fill_color=GREY_E, fill_opacity=0.6,
        ).move_to(LEFT * 3.5 + UP * 1.6)
        input_text = Text("Input Text", font_size=18, color=WHITE)
        input_text.move_to(input_box)
        self.play(FadeIn(input_box), FadeIn(input_text), run_time=0.4)

        # Arrow input -> rules
        arrow1 = Arrow(
            input_box.get_bottom(), input_box.get_bottom() + DOWN * 0.8,
            color=GREY_B, stroke_width=3,
        )
        self.play(GrowArrow(arrow1), run_time=0.3)

        self.next_section("gofai_rules")
        # Rules box (larger, with example rules)
        rules_box = RoundedRectangle(
            width=3.2, height=1.8, corner_radius=0.15,
            color=BLUE_C, fill_color=BLUE_E, fill_opacity=0.3,
        ).move_to(LEFT * 3.5 + DOWN * 0.3)
        rules_title = Text("Hand-Written Rules", font_size=16, color=BLUE_C, weight=BOLD)
        rules_title.next_to(rules_box.get_top(), DOWN, buff=0.12)

        rule1 = Text(
            'IF "bank" + "river"',
            font_size=12, font="Monospace", color=GREY_A,
        )
        rule1_r = Text(
            '-> riverbank',
            font_size=12, font="Monospace", color=GREEN_C,
        )
        rule2 = Text(
            'IF "bank" + "money"',
            font_size=12, font="Monospace", color=GREY_A,
        )
        rule2_r = Text(
            '-> financial',
            font_size=12, font="Monospace", color=GREEN_C,
        )

        rule1.next_to(rules_title, DOWN, buff=0.2).align_to(rules_box.get_left() + RIGHT * 0.2, LEFT)
        rule1_r.next_to(rule1, DOWN, buff=0.06).align_to(rule1, LEFT)
        rule2.next_to(rule1_r, DOWN, buff=0.15).align_to(rule1, LEFT)
        rule2_r.next_to(rule2, DOWN, buff=0.06).align_to(rule2, LEFT)

        self.play(
            FadeIn(rules_box), FadeIn(rules_title),
            run_time=0.4,
        )
        self.play(
            FadeIn(rule1), FadeIn(rule1_r),
            FadeIn(rule2), FadeIn(rule2_r),
            run_time=0.6,
        )

        self.next_section("gofai_output")
        # Arrow rules -> output
        arrow2 = Arrow(
            rules_box.get_bottom(), rules_box.get_bottom() + DOWN * 0.8,
            color=GREY_B, stroke_width=3,
        )
        self.play(GrowArrow(arrow2), run_time=0.3)

        # Output box
        output_box = RoundedRectangle(
            width=2.8, height=0.7, corner_radius=0.15,
            color=WHITE, fill_color=GREY_E, fill_opacity=0.6,
        ).move_to(LEFT * 3.5 + DOWN * 2.2)
        output_text = Text("Output", font_size=18, color=WHITE)
        output_text.move_to(output_box)
        self.play(FadeIn(output_box), FadeIn(output_text), run_time=0.4)

        self.wait(0.5)

        self.next_section("gofai_ambiguity_failure")
        # ---- Animate ambiguous input and failure ----
        ambig_input = Text(
            '"I went to the bank"',
            font_size=16, font="Monospace", color=YELLOW,
        )
        ambig_input.move_to(input_box)
        self.play(
            FadeOut(input_text),
            FadeIn(ambig_input),
            run_time=0.5,
        )

        # Show confusion -- question marks in output
        confusion = Text("???", font_size=36, color=RED_C, weight=BOLD)
        confusion.move_to(output_box)
        fail_rect = SurroundingRectangle(output_box, color=RED_C, buff=0.05)

        self.play(
            FadeOut(output_text),
            FadeIn(confusion),
            Create(fail_rect),
            run_time=0.6,
        )

        fail_label = Text("No matching rule!", font_size=16, color=RED_C)
        fail_label.next_to(fail_rect, DOWN, buff=0.15)
        self.play(FadeIn(fail_label), run_time=0.4)
        self.wait(0.8)

        # =====================================================================
        # RIGHT SIDE: Neural / learned model
        # =====================================================================
        self.next_section("learned_model_input")
        right_title = Text("Learned Model", font_size=24, color=GREEN_C, weight=BOLD)
        right_title.move_to(RIGHT * 3.5 + UP * 2.5)
        self.play(FadeIn(right_title), run_time=0.4)

        # Input box
        nn_input_box = RoundedRectangle(
            width=2.8, height=0.7, corner_radius=0.15,
            color=WHITE, fill_color=GREY_E, fill_opacity=0.6,
        ).move_to(RIGHT * 3.5 + UP * 1.6)
        nn_input_text = Text(
            '"I went to the bank"',
            font_size=14, font="Monospace", color=YELLOW,
        )
        nn_input_text.move_to(nn_input_box)
        self.play(FadeIn(nn_input_box), FadeIn(nn_input_text), run_time=0.4)

        # Arrow
        nn_arrow1 = Arrow(
            nn_input_box.get_bottom(), nn_input_box.get_bottom() + DOWN * 0.8,
            color=GREY_B, stroke_width=3,
        )
        self.play(GrowArrow(nn_arrow1), run_time=0.3)

        self.next_section("learned_model_network")
        # Neural network box
        nn_box = RoundedRectangle(
            width=3.2, height=1.8, corner_radius=0.15,
            color=GREEN_C, fill_color=GREEN_E, fill_opacity=0.3,
        ).move_to(RIGHT * 3.5 + DOWN * 0.3)
        nn_label = Text("Neural Network", font_size=18, color=GREEN_C, weight=BOLD)
        nn_label.move_to(nn_box.get_center() + UP * 0.35)
        nn_sub = Text("Learns patterns\nfrom data", font_size=14, color=GREY_A)
        nn_sub.move_to(nn_box.get_center() + DOWN * 0.3)

        self.play(FadeIn(nn_box), FadeIn(nn_label), FadeIn(nn_sub), run_time=0.5)

        # Arrow
        nn_arrow2 = Arrow(
            nn_box.get_bottom(), nn_box.get_bottom() + DOWN * 0.8,
            color=GREY_B, stroke_width=3,
        )
        self.play(GrowArrow(nn_arrow2), run_time=0.3)

        self.next_section("learned_model_success")
        # Output box with correct answer
        nn_output_box = RoundedRectangle(
            width=2.8, height=0.7, corner_radius=0.15,
            color=WHITE, fill_color=GREY_E, fill_opacity=0.6,
        ).move_to(RIGHT * 3.5 + DOWN * 2.2)
        nn_output_text = Text("financial (95%)", font_size=18, color=GREEN_C, weight=BOLD)
        nn_output_text.move_to(nn_output_box)
        success_rect = SurroundingRectangle(nn_output_box, color=GREEN_C, buff=0.05)

        self.play(
            FadeIn(nn_output_box), FadeIn(nn_output_text),
            Create(success_rect),
            run_time=0.6,
        )

        self.next_section("divider")
        # Divider line
        divider = DashedLine(
            UP * 2.8, DOWN * 2.8,
            color=GREY_D, dash_length=0.15,
        )
        self.play(Create(divider), run_time=0.4)

        self.wait(2)
