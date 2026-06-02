"""Module 4 Manim animations."""

from manim import *


BG = "#0a0e1a"
TEXT = "#e8eaf0"
MUTED = "#8892a4"
PRIMARY = "#4a9eff"
SECONDARY = "#f5a623"
GREEN = "#3fb950"
LINE = "#2a3450"

FONT = "Helvetica Neue"


class BPETrainingScene(Scene):
    """Step-through Byte-Pair Encoding on a tiny corpus."""

    def token_box(self, label: str, color: str = PRIMARY) -> VGroup:
        text = Text(label, font=FONT, font_size=24, color=TEXT)
        box = RoundedRectangle(
            width=max(0.62, text.width + 0.28),
            height=0.48,
            corner_radius=0.08,
            stroke_color=color,
            fill_color=color,
            fill_opacity=0.12,
            stroke_width=2,
        )
        text.move_to(box)
        return VGroup(box, text)

    def word_group(self, symbols: list[str]) -> VGroup:
        return VGroup(*[self.token_box(symbol) for symbol in symbols]).arrange(RIGHT, buff=0.06)

    def corpus_group(self, rows: list[list[str]]) -> VGroup:
        return VGroup(*[self.word_group(row) for row in rows]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)

    def label(self, text: str, size: int = 28, color: str = TEXT) -> Text:
        return Text(text, font=FONT, font_size=size, color=color)

    def construct(self):
        self.camera.background_color = BG

        title = self.label("Byte-Pair Encoding", 34, TEXT).to_edge(UP, buff=0.35)
        subtitle = self.label("Repeatedly merge the most frequent adjacent pair", 20, MUTED).next_to(title, DOWN, buff=0.12)

        rows_chars = [
            ["l", "o", "w"],
            ["l", "o", "w"],
            ["l", "o", "w", "e", "r"],
            ["l", "o", "w", "e", "s", "t"],
        ]
        corpus = self.corpus_group(rows_chars).scale(0.95).move_to(LEFT * 3.2 + DOWN * 0.25)

        corpus_label = self.label("Corpus: low low lower lowest", 22, SECONDARY).next_to(corpus, UP, buff=0.35)

        # Section 1: start from characters.
        self.next_section("start", skip_animations=False)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.5)
        self.play(FadeIn(corpus_label), LaggedStart(*[FadeIn(row) for row in corpus], lag_ratio=0.15), run_time=0.9)
        start_note = self.label("Initial vocabulary: characters", 22, MUTED).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(start_note), run_time=0.4)
        self.wait(0.3)

        # Section 2: count adjacent pairs.
        self.next_section("count_pairs", skip_animations=False)
        counts = VGroup(
            self.label("Pair counts", 26, SECONDARY),
            self.label("lo  : 4", 24, TEXT),
            self.label("ow : 4", 24, TEXT),
            self.label("we : 2", 24, TEXT),
            self.label("er  : 1", 24, MUTED),
            self.label("es  : 1", 24, MUTED),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to(RIGHT * 3.1 + DOWN * 0.05)

        highlights = VGroup()
        for row in corpus:
            left = row[0].get_left()
            right = row[1].get_right()
            center = (left + right) / 2
            highlights.add(
                RoundedRectangle(
                    width=row[0].width + row[1].width + 0.12,
                    height=0.58,
                    corner_radius=0.08,
                    stroke_color=SECONDARY,
                    stroke_width=3,
                ).move_to(center)
            )

        self.play(FadeOut(start_note), FadeIn(counts), Create(highlights), run_time=0.9)
        count_note = self.label("Choose one frequent pair to merge first", 22, MUTED).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(count_note), run_time=0.4)
        self.wait(0.3)

        # Section 3: merge lo.
        self.next_section("merge_lo", skip_animations=False)
        rows_lo = [
            ["lo", "w"],
            ["lo", "w"],
            ["lo", "w", "e", "r"],
            ["lo", "w", "e", "s", "t"],
        ]
        corpus_lo = self.corpus_group(rows_lo).scale(0.95).move_to(corpus)
        vocab_lo = VGroup(
            self.label("Vocabulary gains", 24, SECONDARY),
            self.token_box("lo", SECONDARY),
        ).arrange(DOWN, buff=0.2).next_to(counts, DOWN, buff=0.45)

        self.play(FadeOut(highlights), FadeOut(count_note), Transform(corpus, corpus_lo), FadeIn(vocab_lo), run_time=0.9)
        lo_note = self.label("The pair l + o becomes one token: lo", 22, TEXT).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(lo_note), run_time=0.4)
        self.wait(0.3)

        # Section 4: merge low.
        self.next_section("merge_low", skip_animations=False)
        low_highlights = VGroup()
        for row in corpus:
            left = row[0].get_left()
            right = row[1].get_right()
            center = (left + right) / 2
            low_highlights.add(
                RoundedRectangle(
                    width=row[0].width + row[1].width + 0.12,
                    height=0.58,
                    corner_radius=0.08,
                    stroke_color=SECONDARY,
                    stroke_width=3,
                ).move_to(center)
            )

        rows_low = [
            ["low"],
            ["low"],
            ["low", "e", "r"],
            ["low", "e", "s", "t"],
        ]
        corpus_low = self.corpus_group(rows_low).scale(0.95).move_to(corpus)
        vocab_low = VGroup(
            self.label("Vocabulary gains", 24, SECONDARY),
            VGroup(self.token_box("lo", SECONDARY), self.token_box("low", SECONDARY)).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.2).move_to(vocab_lo)

        self.play(FadeOut(lo_note), Create(low_highlights), run_time=0.45)
        self.play(FadeOut(low_highlights), Transform(corpus, corpus_low), Transform(vocab_lo, vocab_low), run_time=0.85)
        low_note = self.label("Now lo + w becomes one token: low", 22, TEXT).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(low_note), run_time=0.4)
        self.wait(0.3)

        # Section 5: reusable pieces.
        self.next_section("result", skip_animations=False)
        result = VGroup(
            VGroup(self.label("lower", 24, TEXT), self.label("=", 24, MUTED), self.token_box("low", SECONDARY), self.token_box("er", GREEN)).arrange(RIGHT, buff=0.16),
            VGroup(self.label("lowest", 24, TEXT), self.label("=", 24, MUTED), self.token_box("low", SECONDARY), self.token_box("est", GREEN)).arrange(RIGHT, buff=0.16),
        ).arrange(DOWN, buff=0.34, aligned_edge=LEFT).move_to(RIGHT * 2.7 + DOWN * 0.2)

        final_note = self.label("Frequent pieces stay whole; rare words split into reusable parts", 22, TEXT).to_edge(DOWN, buff=0.55)
        self.play(FadeOut(counts), FadeOut(vocab_lo), FadeOut(low_note), FadeIn(result), run_time=0.8)
        self.play(FadeIn(final_note), run_time=0.4)
        self.wait(0.4)
