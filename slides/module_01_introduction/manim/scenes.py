"""
Module 1: Course Introduction - Manim Animations
Only scenes that require spatial/data visualization belong here.
Most content has been moved to reveal.js slide animations.
"""

from manim import *
import numpy as np


class LanguageAsCompressionScene(Scene):
    """Visualize uniform vs variable-length encoding of 'the cat sat on the mat'.

    Three sections (click-through):
      1. Uniform encoding: equal-width blocks, 22 x 5 = 110 bits
      2. Code table: prefix codes sorted by frequency
      3. Variable-length encoding: block widths proportional to bit count, 70 bits
    """

    # Colors matching the slide theme
    BG_COLOR = "#0a0e1a"
    TEXT_COLOR = "#e8eaf0"
    MUTED_COLOR = "#8892a4"
    UNIFORM_COLOR = "#e06c75"
    SHORT_CODE_COLOR = "#50c878"
    MED_CODE_COLOR = "#98c379"
    LONG_CODE_COLOR = "#e5c07b"
    HEADING_COLOR = "#4a9eff"

    # Fonts (system-available)
    HEADING_FONT = "Helvetica Neue"
    BODY_FONT = "Helvetica Neue"
    MONO_FONT = "Courier New"

    STRING = "the cat sat on the mat"
    CODE_TABLE = {
        " ": ("00", 5),
        "t": ("01", 5),
        "a": ("100", 3),
        "e": ("101", 2),
        "h": ("110", 2),
        "n": ("1110", 1),
        "o": ("11110", 1),
        "s": ("111110", 1),
        "c": ("1111110", 1),
        "m": ("1111111", 1),
    }

    def construct(self):
        self.camera.background_color = self.BG_COLOR
        self._section_uniform()
        self._section_code_table()
        self._section_variable()

    # ------------------------------------------------------------------
    # Section 1: Uniform encoding
    # ------------------------------------------------------------------
    def _section_uniform(self):
        self.next_section("uniform", skip_animations=False)

        title = Text(
            "Language as Compression", font=self.HEADING_FONT,
            font_size=40, color=self.TEXT_COLOR, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        subtitle = Text(
            'Encode: "the cat sat on the mat"  (22 characters)',
            font=self.BODY_FONT, font_size=22, color=self.MUTED_COLOR,
        ).next_to(title, DOWN, buff=0.25)

        label = Text(
            "Uniform encoding: every character = 5 bits",
            font=self.BODY_FONT, font_size=22, color=self.UNIFORM_COLOR,
        ).next_to(subtitle, DOWN, buff=0.5).align_to(subtitle, LEFT)

        BLOCK_W = 0.5
        BLOCK_H = 0.45
        blocks = VGroup()
        for ch in self.STRING:
            display_ch = ch if ch != " " else "\u2423"
            rect = Rectangle(
                width=BLOCK_W, height=BLOCK_H,
                fill_color=self.UNIFORM_COLOR, fill_opacity=0.35,
                stroke_color=self.UNIFORM_COLOR, stroke_width=1.5,
            )
            char_label = Text(
                display_ch, font=self.MONO_FONT, font_size=16, color=self.TEXT_COLOR,
            ).move_to(rect)
            block = VGroup(rect, char_label)
            blocks.add(block)

        blocks.arrange(RIGHT, buff=0.04)
        blocks.next_to(label, DOWN, buff=0.4)

        bit_labels = VGroup()
        for block in blocks:
            bl = Text("5", font=self.MONO_FONT, font_size=12, color=self.MUTED_COLOR)
            bl.next_to(block, UP, buff=0.08)
            bit_labels.add(bl)

        total_text = Text(
            "22 chars x 5 bits = 110 bits total",
            font=self.BODY_FONT, font_size=22, color=self.UNIFORM_COLOR, weight=BOLD,
        ).next_to(blocks, DOWN, buff=0.4)

        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)
        self.play(FadeIn(label), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in blocks], lag_ratio=0.03),
            run_time=1.2,
        )
        self.play(FadeIn(bit_labels), run_time=0.5)
        self.play(FadeIn(total_text), run_time=0.5)
        self.wait(0.3)

        self._uniform_group = VGroup(title, subtitle, label, blocks, bit_labels, total_text)

    # ------------------------------------------------------------------
    # Section 2: Code table
    # ------------------------------------------------------------------
    def _section_code_table(self):
        self.next_section("code_table", skip_animations=False)

        title = self._uniform_group[0]
        subtitle = self._uniform_group[1]
        to_remove = VGroup(*self._uniform_group[2:])
        self.play(FadeOut(to_remove), run_time=0.5)

        prompt = Text(
            "Assign shorter codes to common characters",
            font=self.BODY_FONT, font_size=22, color=self.SHORT_CODE_COLOR,
        ).next_to(subtitle, DOWN, buff=0.5).align_to(subtitle, LEFT)

        header_data = ["char", "count", "code", "bits"]
        col_xs = [-3.5, -1.8, 0.0, 2.5]
        header_y = prompt.get_bottom()[1] - 0.6

        headers = VGroup()
        for i, hd in enumerate(header_data):
            h = Text(hd, font=self.BODY_FONT, font_size=18, color=self.MUTED_COLOR, weight=BOLD)
            h.move_to([col_xs[i], header_y, 0])
            headers.add(h)

        header_line = Line(
            start=[-4.5, header_y - 0.2, 0],
            end=[3.5, header_y - 0.2, 0],
            color=self.MUTED_COLOR, stroke_width=0.8,
        )

        rows = VGroup()
        sorted_chars = list(self.CODE_TABLE.keys())
        for idx, ch in enumerate(sorted_chars):
            code, count = self.CODE_TABLE[ch]
            bits = len(code)
            row_y = header_y - 0.45 - idx * 0.35

            if bits <= 2:
                row_color = self.SHORT_CODE_COLOR
            elif bits <= 3:
                row_color = self.MED_CODE_COLOR
            else:
                row_color = self.LONG_CODE_COLOR

            display_ch = "' '" if ch == " " else ch
            char_t = Text(display_ch, font=self.MONO_FONT, font_size=16, color=row_color)
            char_t.move_to([col_xs[0], row_y, 0])

            count_t = Text(str(count), font=self.MONO_FONT, font_size=16, color=self.TEXT_COLOR)
            count_t.move_to([col_xs[1], row_y, 0])

            code_t = Text(code, font=self.MONO_FONT, font_size=16, color=row_color)
            code_t.move_to([col_xs[2], row_y, 0])

            bits_t = Text(str(bits), font=self.MONO_FONT, font_size=16, color=self.TEXT_COLOR)
            bits_t.move_to([col_xs[3], row_y, 0])

            row = VGroup(char_t, count_t, code_t, bits_t)
            rows.add(row)

        insight = Text(
            "Common chars (' ', t) = 2 bits.  Rare chars (c, m) = 7 bits.",
            font=self.BODY_FONT, font_size=20, color=self.TEXT_COLOR,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(prompt), run_time=0.4)
        self.play(FadeIn(headers), Create(header_line), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.06),
            run_time=1.5,
        )
        self.play(FadeIn(insight), run_time=0.5)
        self.wait(0.3)

        self._table_group = VGroup(prompt, headers, header_line, rows, insight)

    # ------------------------------------------------------------------
    # Section 3: Variable-length encoding
    # ------------------------------------------------------------------
    def _section_variable(self):
        self.next_section("variable", skip_animations=False)

        title = self._uniform_group[0]
        subtitle = self._uniform_group[1]

        self.play(FadeOut(self._table_group), run_time=0.5)

        label = Text(
            "Variable-length encoding",
            font=self.BODY_FONT, font_size=22, color=self.SHORT_CODE_COLOR,
        ).next_to(subtitle, DOWN, buff=0.5).align_to(subtitle, LEFT)

        SCALE = 0.14
        BLOCK_H = 0.45
        blocks_row1 = VGroup()
        blocks_row2 = VGroup()

        chars_list = list(self.STRING)
        split_at = 11

        for i, ch in enumerate(chars_list):
            code, _ = self.CODE_TABLE[ch]
            bits = len(code)
            w = bits * SCALE

            if bits <= 2:
                fill_c = self.SHORT_CODE_COLOR
                fill_op = 0.45
            elif bits <= 3:
                fill_c = self.MED_CODE_COLOR
                fill_op = 0.35
            else:
                fill_c = self.LONG_CODE_COLOR
                fill_op = 0.25

            rect = Rectangle(
                width=w, height=BLOCK_H,
                fill_color=fill_c, fill_opacity=fill_op,
                stroke_color=fill_c, stroke_width=1.5,
            )

            display_ch = ch if ch != " " else "\u2423"
            char_label = Text(
                display_ch, font=self.MONO_FONT, font_size=14, color=self.TEXT_COLOR,
            ).move_to(rect)

            code_label = Text(
                code, font=self.MONO_FONT, font_size=9, color=fill_c,
            ).next_to(rect, UP, buff=0.06)

            block = VGroup(code_label, rect, char_label)

            if i < split_at:
                blocks_row1.add(block)
            else:
                blocks_row2.add(block)

        blocks_row1.arrange(RIGHT, buff=0.03)
        blocks_row2.arrange(RIGHT, buff=0.03)
        # Center both rows horizontally, position below label
        both_rows = VGroup(blocks_row1, blocks_row2).arrange(DOWN, buff=0.35, center=True)
        both_rows.next_to(label, DOWN, buff=0.5)
        both_rows.move_to([0, both_rows.get_center()[1], 0])  # center horizontally

        # Comparison boxes (use RoundedRectangle for rounded corners)
        uniform_rect = RoundedRectangle(
            width=3.8, height=0.55, corner_radius=0.08,
            fill_color=self.UNIFORM_COLOR, fill_opacity=0.08,
            stroke_color=self.UNIFORM_COLOR, stroke_width=1,
        )
        uniform_label = Text(
            "Uniform: 110 bits", font=self.BODY_FONT, font_size=18,
            color=self.UNIFORM_COLOR, weight=BOLD,
        ).move_to(uniform_rect)
        uniform_box = VGroup(uniform_rect, uniform_label)

        variable_rect = RoundedRectangle(
            width=3.8, height=0.55, corner_radius=0.08,
            fill_color=self.SHORT_CODE_COLOR, fill_opacity=0.08,
            stroke_color=self.SHORT_CODE_COLOR, stroke_width=1,
        )
        variable_label = Text(
            "Variable: 70 bits", font=self.BODY_FONT, font_size=18,
            color=self.SHORT_CODE_COLOR, weight=BOLD,
        ).move_to(variable_rect)
        variable_box = VGroup(variable_rect, variable_label)

        comparison = VGroup(uniform_box, variable_box).arrange(RIGHT, buff=0.6)
        comparison.next_to(blocks_row2, DOWN, buff=0.5)

        savings = Text(
            "36% smaller. Better prediction = better compression.",
            font=self.BODY_FONT, font_size=20, color=self.TEXT_COLOR,
        ).next_to(comparison, DOWN, buff=0.3)

        self.play(FadeIn(label), run_time=0.4)
        self.play(
            LaggedStart(
                *[FadeIn(b, shift=UP * 0.15) for b in blocks_row1],
                lag_ratio=0.04,
            ),
            run_time=1.0,
        )
        self.play(
            LaggedStart(
                *[FadeIn(b, shift=UP * 0.15) for b in blocks_row2],
                lag_ratio=0.04,
            ),
            run_time=1.0,
        )
        self.play(FadeIn(comparison), run_time=0.5)
        self.play(FadeIn(savings), run_time=0.5)
        self.wait(0.3)
