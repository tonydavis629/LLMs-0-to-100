"""Module 4 Manim animations."""

from manim import *

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


BG = "#0a0e1a"
TEXT = "#e8eaf0"
MUTED = "#8892a4"
PRIMARY = "#4a9eff"
SECONDARY = "#f5a623"
GREEN = "#3fb950"
LINE = "#2a3450"

FONT = "Helvetica Neue"


# ---------------------------------------------------------------------------
# Crisp-kerning Text.
#
# Manim renders a Text mobject's glyphs at a pixel size tied to `font_size`; at
# small sizes the inter-glyph advances get coarsely rounded, so text drawn at
# e.g. font_size=23 and then shown near full slide width looks loosely and
# unevenly spaced ("bad kerning"). Rendering at a large reference size and
# scaling the VECTOR result down yields accurate, tight kerning at any display
# size. We shadow Text so every call in this file benefits without touching the
# call sites; layout helpers read `.width`/`.height` after the scale, so they
# keep working unchanged.
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


# ---------------------------------------------------------------------------
# Shared helpers (no LaTeX: every label is a Text mobject so the scenes render
# without a TeX install; no color gradients, matching the course style rules).
# ---------------------------------------------------------------------------

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


def title_bar(text: str, sub: str | None = None) -> VGroup:
    t = label(text, 34, TEXT, weight=BOLD).to_edge(UP, buff=0.35)
    grp = VGroup(t)
    if sub:
        s = label(sub, 20, MUTED).next_to(t, DOWN, buff=0.12)
        grp.add(s)
    return grp


def value_bars(values, x0=-2.0, dx=0.62, width=0.44, unit=0.7,
               color=PRIMARY, baseline=0.0, fill=0.55):
    """A row of vertical bars whose signed heights encode the values, each
    bar resting on (or hanging from) a common baseline at deterministic x."""
    bars = VGroup()
    for i, v in enumerate(values):
        h = max(abs(v) * unit, 0.05)
        bar = Rectangle(width=width, height=h, stroke_width=1.5,
                        stroke_color=color, fill_color=color, fill_opacity=fill)
        y = baseline + (h / 2 if v >= 0 else -h / 2)
        bar.move_to([x0 + i * dx, y, 0])
        bars.add(bar)
    return bars


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


class StepScene(Scene):
    """Base scene: dark background plus a single managed caption at the bottom."""

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


class RecurrenceVsAttentionScene(StepScene):
    """Why the transformer replaced recurrence: path length and parallelism."""

    def construct(self):
        self.setup_bg()
        title = title_bar("Recurrence vs Attention")
        self.add(title)

        words = ["The", "cat", "sat", "on", "mat"]
        boxes = VGroup(*[cell(w, 1.5, 0.78, PRIMARY, size=26) for w in words])
        boxes.arrange(RIGHT, buff=0.6).move_to(UP * 1.4)

        # ---- recurrence ----
        self.next_section("recurrence", skip_animations=False)
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15), run_time=0.8)
        chain = VGroup()
        for i in range(len(boxes) - 1):
            chain.add(Arrow(boxes[i].get_right(), boxes[i + 1].get_left(),
                            buff=0.06, stroke_width=4, color=MUTED, max_tip_length_to_length_ratio=0.25))
        self.play(LaggedStart(*[GrowArrow(a) for a in chain], lag_ratio=0.2), run_time=0.8)
        self.caption("An RNN carries one hidden state along the chain, step by step.")

        dot = Dot(color=SECONDARY, radius=0.13).move_to(boxes[0].get_center())
        self.play(FadeIn(dot, scale=1.5), run_time=0.3)
        for i in range(1, len(boxes)):
            self.play(dot.animate.move_to(boxes[i].get_center()), run_time=0.32, rate_func=linear)
        self.caption("For 'The' to reach 'mat', the signal crosses every step between: path length O(n).")
        self.wait(0.2)

        # ---- attention ----
        self.next_section("attention", skip_animations=False)
        self.play(FadeOut(chain), FadeOut(dot), run_time=0.4)
        arcs = VGroup()
        last = boxes[-1]
        for i in range(len(boxes) - 1):
            arcs.add(CurvedArrow(last.get_bottom() + DOWN * 0.05, boxes[i].get_bottom() + DOWN * 0.05,
                                 angle=-1.0, color=SECONDARY, stroke_width=3, tip_length=0.18))
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.18), run_time=1.0)
        self.caption("Self-attention links 'mat' to every earlier token directly, in one layer.")
        self.play(*[b.animate.set_stroke(SECONDARY) for b in boxes], run_time=0.4)
        self.caption("Every token does this at once: constant path length O(1), fully parallel.")
        self.wait(0.3)


class EmbeddingLookupScene(StepScene):
    """A token id selects one learned row of the embedding matrix."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Embedding Lookup", "word to vector"))

        # ---- word -> id ----
        self.next_section("word", skip_animations=False)
        word = cell("France", 1.9, 0.85, SECONDARY, size=30).move_to(LEFT * 4.6 + UP * 1.1)
        a1 = Arrow(word.get_right(), word.get_right() + RIGHT * 0.9, buff=0.12, color=MUTED, stroke_width=4)
        idb = cell("id 4881", 1.9, 0.85, PRIMARY, size=28).next_to(a1, RIGHT, buff=0.12)
        self.play(FadeIn(word), run_time=0.4)
        self.play(GrowArrow(a1), FadeIn(idb), run_time=0.5)
        self.caption("A token is just an integer index.")

        # ---- lookup in the matrix ----
        self.next_section("lookup", skip_animations=False)
        rows = VGroup()
        ids = ["4879", "4880", "4881", "4882"]
        for rid in ids:
            cells = VGroup(*[Square(0.42, stroke_color=LINE, stroke_width=1.5,
                                    fill_color=PRIMARY, fill_opacity=0.10) for _ in range(6)])
            cells.arrange(RIGHT, buff=0.06)
            rlab = label(rid, 18, MUTED).next_to(cells, LEFT, buff=0.22)
            rows.add(VGroup(rlab, cells))
        rows.arrange(DOWN, buff=0.16, aligned_edge=RIGHT).move_to(RIGHT * 2.2 + UP * 0.9)
        mat_label = label("embedding matrix", 18, MUTED).next_to(rows, UP, buff=0.22)
        a2 = Arrow(idb.get_right(), rows.get_left() + LEFT * 0.1, buff=0.15, color=MUTED, stroke_width=4)
        self.play(GrowArrow(a2), FadeIn(mat_label), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.1), run_time=0.9)

        target = rows[2][1]  # the 4881 cells
        hl = SurroundingRectangle(rows[2], color=SECONDARY, buff=0.08, stroke_width=3)
        self.play(Create(hl), target.animate.set_fill(SECONDARY, opacity=0.30), run_time=0.6)
        self.caption("The id selects one learned row of the matrix.")

        # ---- the row IS the vector ----
        self.next_section("vector", skip_animations=False)
        nums = ["0.12", "-0.4", "0.9", "0.3", "-0.7", "0.5"]
        vec = VGroup(*[cell(n, 0.92, 0.62, SECONDARY, size=20) for n in nums]).arrange(RIGHT, buff=0.08)
        vec.move_to(DOWN * 2.0)
        vlab = label("the token's vector", 20, SECONDARY).next_to(vec, UP, buff=0.22)
        self.play(TransformFromCopy(target, vec), FadeIn(vlab), run_time=0.9)
        self.caption("That row is the token's vector. From here on, the model works only with vectors.")
        self.wait(0.3)


class PositionalEncodingScene(StepScene):
    """Attention is order-blind; adding a position vector fixes that."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Positional Encoding"))

        # ---- order-blind ----
        self.next_section("blind", skip_animations=False)
        rowA = VGroup(cell("cat", 1.5, 0.8, PRIMARY, size=26), cell("sat", 1.5, 0.8, PRIMARY, size=26))
        rowA.arrange(RIGHT, buff=0.4).move_to(UP * 1.9 + LEFT * 0.5)
        rowB = VGroup(cell("sat", 1.5, 0.8, PRIMARY, size=26), cell("cat", 1.5, 0.8, PRIMARY, size=26))
        rowB.arrange(RIGHT, buff=0.4).move_to(UP * 0.6 + LEFT * 0.5)
        eq = label("same set of vectors", 22, MUTED).next_to(VGroup(rowA, rowB), RIGHT, buff=0.6)
        self.play(FadeIn(rowA), FadeIn(rowB), run_time=0.6)
        self.play(FadeIn(eq), run_time=0.4)
        self.caption("Attention sees a set, not a sequence: 'cat sat' and 'sat cat' look identical.")

        # ---- add position ----
        self.next_section("add", skip_animations=False)
        self.play(FadeOut(rowB), FadeOut(eq), rowA.animate.move_to(UP * 1.7 + LEFT * 4.5), run_time=0.6)

        def addrow(tok, pos, res, y):
            t = cell(tok, 1.4, 0.72, PRIMARY, size=24)
            plus = label("+", 30, MUTED)
            p = cell(pos, 1.4, 0.72, MUTED, size=22)
            eqs = label("=", 30, MUTED)
            r = cell(res, 1.7, 0.72, SECONDARY, size=22)
            g = VGroup(t, plus, p, eqs, r).arrange(RIGHT, buff=0.28).move_to(UP * y)
            return g

        r1 = addrow("cat", "pos 0", "cat @ 0", 0.4)
        r2 = addrow("sat", "pos 1", "sat @ 1", -0.9)
        self.play(FadeOut(rowA), run_time=0.3)
        self.play(LaggedStart(FadeIn(r1), FadeIn(r2), lag_ratio=0.3), run_time=0.9)
        self.caption("Adding a distinct position vector to each token makes order part of the representation.")
        self.wait(0.3)


class AttentionFlowScene(StepScene):
    """Query-key-value flow for one query token."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Self-Attention", "one query reads from all tokens"))

        toks = ["The", "cat", "sat"]
        weights = [0.2, 0.3, 0.5]
        colors = [PRIMARY, PRIMARY, SECONDARY]

        # ---- project to Q, K, V ----
        self.next_section("project", skip_animations=False)
        query = cell("query: sat", 2.4, 0.8, SECONDARY, size=24).move_to(UP * 2.2 + LEFT * 3.4)
        rows = VGroup()
        for t, c in zip(toks, colors):
            tk = cell(t, 1.1, 0.66, c, size=22)
            k = cell("K", 0.66, 0.66, PRIMARY, size=22)
            v = cell("V", 0.66, 0.66, GREEN, size=22)
            rows.add(VGroup(tk, k, v).arrange(RIGHT, buff=0.22))
        rows.arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(LEFT * 3.4 + DOWN * 0.3)
        self.play(FadeIn(query), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.15), run_time=0.8)
        self.caption("Each token emits a query; every token exposes a key (K) and a value (V).")

        # ---- scores -> weights ----
        self.next_section("scores", skip_animations=False)
        qdots = VGroup()
        for r in rows:
            qdots.add(Line(query.get_bottom(), r[1].get_top(), color=MUTED, stroke_width=2, stroke_opacity=0.6))
        self.play(LaggedStart(*[Create(l) for l in qdots], lag_ratio=0.1), run_time=0.7)
        wbars = VGroup()
        wlabs = VGroup()
        for r, w, c in zip(rows, weights, colors):
            bar = Rectangle(width=w * 4.2, height=0.46, stroke_width=1.5, stroke_color=c,
                            fill_color=c, fill_opacity=0.5)
            bar.next_to(r, RIGHT, buff=0.7).align_to(r, LEFT).shift(RIGHT * (r.width + 0.7))
            bar.move_to([1.2 + w * 2.1, r.get_y(), 0])
            wl = label(f"{w:.1f}", 20, c).next_to(bar, RIGHT, buff=0.15)
            wbars.add(bar)
            wlabs.add(wl)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in wbars], lag_ratio=0.12),
                  LaggedStart(*[FadeIn(l) for l in wlabs], lag_ratio=0.12), run_time=0.9)
        self.caption("Compare the query to each key, then softmax the scores into attention weights.")

        # ---- weighted sum ----
        self.next_section("weighted", skip_animations=False)
        out = cell("output", 2.0, 0.9, SECONDARY, size=24).move_to(RIGHT * 4.6 + DOWN * 0.3)
        flows = VGroup()
        for r in rows:
            flows.add(Arrow(r[2].get_right(), out.get_left(), buff=0.15, color=GREEN,
                            stroke_width=3, max_tip_length_to_length_ratio=0.12, stroke_opacity=0.7))
        self.play(LaggedStart(*[GrowArrow(a) for a in flows], lag_ratio=0.12), FadeIn(out), run_time=0.9)
        self.caption("The output is the weighted sum of values: 'sat' keeps most of itself, plus context.")
        self.wait(0.3)


class FFNExpandScene(StepScene):
    """The position-wise FFN: project up, activate, project down."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Feed-Forward Network", "applied to each token independently"))

        def col(n, x, color, r=0.10):
            dots = VGroup(*[Dot(radius=r, color=color, fill_opacity=0.9) for _ in range(n)])
            dots.arrange(DOWN, buff=(3.6 / max(n, 1)) - 2 * r)
            dots.move_to([x, 0, 0])
            return dots

        inp = col(4, -4.6, PRIMARY)
        hid = col(12, 0.0, MUTED, r=0.085)
        outp = col(4, 4.6, SECONDARY)

        # ---- token vector ----
        self.next_section("vector", skip_animations=False)
        self.play(FadeIn(inp), run_time=0.5)
        in_lab = label("d = 4\n(token vector)", 20, PRIMARY).next_to(inp, DOWN, buff=0.35)
        self.play(FadeIn(in_lab), run_time=0.3)
        self.caption("Start from one token's vector.")

        # ---- expand ----
        self.next_section("expand", skip_animations=False)
        up = VGroup(*[Line(a.get_center(), b.get_center(), color=PRIMARY, stroke_width=1.4, stroke_opacity=0.18)
                      for a in inp for b in hid]).set_z_index(-1)
        hid_lab = label("4d = 16 (wider hidden layer)", 20, MUTED).next_to(hid, UP, buff=0.35)
        self.play(Create(up), run_time=0.7)
        self.play(FadeIn(hid), FadeIn(hid_lab), run_time=0.6)
        self.caption("Project up into a much wider hidden space.")

        # ---- activate ----
        self.next_section("activate", skip_animations=False)
        act_lab = label("nonlinearity (ReLU)", 20, GREEN).move_to(hid_lab)
        dims = [1, 4, 6, 9, 11]  # units ReLU zeroes out (negative pre-activations)
        self.play(Transform(hid_lab, act_lab),
                  *[hid[i].animate.set_fill(LINE, opacity=0.35) for i in dims], run_time=0.6)
        self.caption("ReLU zeroes every negative unit, keeping some features and switching others off.")

        # ---- contract ----
        self.next_section("contract", skip_animations=False)
        down = VGroup(*[Line(a.get_center(), b.get_center(), color=SECONDARY, stroke_width=1.4, stroke_opacity=0.18)
                        for a in hid for b in outp]).set_z_index(-1)
        out_lab = label("back to d = 4", 20, SECONDARY).next_to(outp, DOWN, buff=0.35)
        self.play(Create(down), run_time=0.7)
        self.play(FadeIn(outp), FadeIn(out_lab), run_time=0.5)
        self.caption("Project back down. The wide hidden layer is where much of the model's knowledge lives.")
        self.wait(0.3)


class ResidualStreamScene(StepScene):
    """The residual stream as a shared highway that each sub-layer reads and adds to."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("The Residual Stream"))

        stream_y = -1.2
        stream = RoundedRectangle(width=9.2, height=0.46, corner_radius=0.23,
                                  stroke_color=PRIMARY, stroke_width=2,
                                  fill_color=PRIMARY, fill_opacity=0.16).move_to([0, stream_y, 0])
        s_lab = label("residual stream", 18, PRIMARY).next_to(stream, DOWN, buff=0.16).set_opacity(0.85)
        embed = cell("embed", 1.3, 0.66, MUTED, size=20).move_to([-5.7, stream_y, 0])
        logits = cell("logits", 1.3, 0.66, SECONDARY, size=20).move_to([5.7, stream_y, 0])
        in_arrow = Arrow(embed.get_right(), stream.get_left(), buff=0.08, color=MUTED,
                         stroke_width=3, max_tip_length_to_length_ratio=0.4)
        out_arrow = Arrow(stream.get_right(), logits.get_left(), buff=0.08, color=SECONDARY,
                          stroke_width=3, max_tip_length_to_length_ratio=0.4)

        # ---- the stream ----
        self.next_section("stream", skip_animations=False)
        self.play(FadeIn(embed), GrowFromEdge(stream, LEFT), run_time=0.7)
        self.play(FadeIn(s_lab), GrowArrow(in_arrow), run_time=0.4)
        self.caption("Picture the block stack as a shared highway flowing left to right.")

        stream_top = stream.get_top()[1]
        box_y, box_w, box_h = 0.95, 1.7, 0.7

        # Each sub-layer reads from the stream (left arrow, straight up) and writes
        # its update back at a + node (right arrow, straight down): both vertical.
        def sublayer(name, cx, color):
            box = cell(name, box_w, box_h, color, size=20).move_to([cx, box_y, 0])
            box_bottom = box.get_bottom()[1]
            read = Arrow([cx - 0.45, stream_top, 0], [cx - 0.45, box_bottom, 0],
                         buff=0.06, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.18)
            plus = VGroup(Circle(radius=0.17, color=SECONDARY, stroke_width=2.5,
                                 fill_color=BG, fill_opacity=1),
                          label("+", 22, SECONDARY))
            plus[1].move_to(plus[0])
            plus.move_to([cx + 0.45, stream_y, 0])
            write = Arrow([cx + 0.45, box_bottom, 0], [cx + 0.45, plus.get_top()[1] + 0.02, 0],
                          buff=0.06, color=color, stroke_width=3, max_tip_length_to_length_ratio=0.18)
            return VGroup(box, read, plus, write)

        def play_block(components):
            for comp in components:
                box, read, plus, write = comp
                self.play(FadeIn(box), GrowArrow(read), run_time=0.35)
                self.play(GrowArrow(write), FadeIn(plus, scale=1.4), run_time=0.35)

        # ---- block 1 ----
        self.next_section("block1", skip_animations=False)
        b1 = [sublayer("Attention", -3.2, PRIMARY), sublayer("MLP", -1.1, GREEN)]
        play_block(b1)
        self.caption("Each sub-layer reads the current state and adds its update back: it never overwrites.")

        # ---- block 2 ----
        self.next_section("block2", skip_animations=False)
        b2 = [sublayer("Attention", 1.1, PRIMARY), sublayer("MLP", 3.2, GREEN)]
        play_block(b2)
        self.caption("Stacking repeats this. Because every write is an addition, early information survives.")

        # ---- readout ----
        self.next_section("readout", skip_animations=False)
        self.play(GrowArrow(out_arrow), FadeIn(logits),
                  Indicate(stream, color=SECONDARY, scale_factor=1.02), run_time=0.7)
        formula = cell("output = embedding + sum of all sub-layer updates", 9.2, 0.66, SECONDARY, size=22).move_to(UP * 2.25)
        self.play(FadeIn(formula), run_time=0.5)
        self.caption("The final state is the embedding plus every layer's contribution.")
        self.wait(0.3)


class NormDemoScene(StepScene):
    """LayerNorm recenters and rescales; RMSNorm only rescales."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("LayerNorm vs RMSNorm"))

        raw = [1.5, -0.5, 2.0, 0.4, -1.0]
        arr = np.array(raw, dtype=float)
        mean = arr.mean()
        std = arr.std()
        rms = np.sqrt((arr ** 2).mean())
        ln = (arr - mean) / std
        rn = arr / rms

        def baseline_line(cx, color=MUTED):
            return DashedLine([cx - 1.8, 0, 0], [cx + 1.8, 0, 0], color=color, stroke_width=2, dash_length=0.12)

        # ---- the vector ----
        self.next_section("vector", skip_animations=False)
        center = value_bars(raw, x0=-1.3, dx=0.62, unit=0.7, color=PRIMARY, baseline=0.0)
        center.move_to([0, 0.2, 0])
        base0 = baseline_line(0).move_to([0, center[0].get_y() - center[0].height / 2 if raw[0] >= 0 else 0, 0])
        base0 = DashedLine([-2.0, 0.2, 0], [2.0, 0.2, 0], color=MUTED, stroke_width=2, dash_length=0.12)
        mean_line = DashedLine([-2.0, 0.2 + mean * 0.7, 0], [2.0, 0.2 + mean * 0.7, 0], color=SECONDARY, stroke_width=2, dash_length=0.12)
        clab = label("a token vector", 20, PRIMARY).next_to(center, UP, buff=1.4)
        self.play(FadeIn(base0), LaggedStart(*[GrowFromEdge(b, DOWN) for b in center], lag_ratio=0.1), run_time=0.9)
        self.play(FadeIn(clab), Create(mean_line), run_time=0.4)
        self.caption("A vector drifts in both offset (mean) and scale as it moves through layers.")

        # ---- layernorm (left) ----
        self.next_section("layernorm", skip_animations=False)
        lnbars = value_bars(ln, x0=-5.6, dx=0.55, unit=0.6, color=GREEN, baseline=0.2)
        lbase = DashedLine([-6.2, 0.2, 0], [-3.0, 0.2, 0], color=MUTED, stroke_width=2, dash_length=0.12)
        llab = label("LayerNorm\n(x - mean) / std", 19, GREEN).next_to(lnbars, UP, buff=0.7)
        self.play(FadeIn(lbase), TransformFromCopy(center, lnbars), FadeIn(llab), run_time=1.0)
        self.caption("LayerNorm recenters to zero mean, then rescales to unit variance.")

        # ---- rmsnorm (right) ----
        self.next_section("rmsnorm", skip_animations=False)
        rnbars = value_bars(rn, x0=3.0, dx=0.55, unit=0.6, color=SECONDARY, baseline=0.2)
        rbase = DashedLine([2.6, 0.2, 0], [5.8, 0.2, 0], color=MUTED, stroke_width=2, dash_length=0.12)
        rmean = rn.mean()
        rmean_line = DashedLine([2.6, 0.2 + rmean * 0.6, 0], [5.8, 0.2 + rmean * 0.6, 0], color=SECONDARY, stroke_width=2, dash_length=0.10)
        rlab = label("RMSNorm\nx / rms  (no recentering)", 19, SECONDARY).next_to(rnbars, UP, buff=0.7)
        self.play(FadeIn(rbase), TransformFromCopy(center, rnbars), FadeIn(rlab), run_time=1.0)
        self.play(Create(rmean_line), run_time=0.3)
        self.caption("RMSNorm keeps only the rescaling. Cheaper, and it works just as well in practice.")
        self.wait(0.3)


class SamplingScene(StepScene):
    """How temperature, top-k, and top-p reshape the next-token distribution."""

    def construct(self):
        self.setup_bg()
        self.add(title_bar("Sampling", "from a distribution to one token"))

        toks = ["Paris", "city", "of", "the", "London", "Lyon", "Nice", "Metz"]
        base = np.array([0.34, 0.18, 0.13, 0.10, 0.09, 0.07, 0.05, 0.04])
        x0, dx, unit = -4.9, 1.32, 6.5

        def make_bars(probs, colors):
            g = VGroup()
            for i, (p, c) in enumerate(zip(probs, colors)):
                h = max(p * unit, 0.05)
                bar = Rectangle(width=0.8, height=h, stroke_width=1.5, stroke_color=c,
                                fill_color=c, fill_opacity=0.6)
                bar.move_to([x0 + i * dx, -1.6 + h / 2, 0])
                g.add(bar)
            return g

        labels = VGroup()
        for i, t in enumerate(toks):
            labels.add(label(t, 16, MUTED).move_to([x0 + i * dx, -2.0, 0]))

        def softmax_T(p, T):
            z = np.log(p + 1e-9) / T
            e = np.exp(z - z.max())
            return e / e.sum()

        # ---- distribution ----
        self.next_section("dist", skip_animations=False)
        cols = [PRIMARY] * len(toks)
        bars = make_bars(base, cols)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.07),
                  FadeIn(labels), run_time=1.0)
        self.caption("The model outputs a probability for every token in the vocabulary.")

        # ---- temperature: sharper (its own clip, so it can be paused on) ----
        self.next_section("temp_sharp", skip_animations=False)
        tlab = label("temperature T = 0.5  (sharper)", 22, SECONDARY).to_edge(UP, buff=1.2)
        sharp = make_bars(softmax_T(base, 0.5), cols)
        self.play(FadeIn(tlab), Transform(bars, sharp), run_time=0.8)
        self.caption("T < 1 sharpens toward the top token.")
        self.wait(0.3)

        # ---- temperature: flatter (separate clip) ----
        self.next_section("temp_flat", skip_animations=False)
        tlab2 = label("temperature T = 1.8  (flatter)", 22, SECONDARY).move_to(tlab)
        flat = make_bars(softmax_T(base, 1.8), cols)
        self.play(Transform(tlab, tlab2), Transform(bars, flat), run_time=0.8)
        self.caption("T > 1 flattens toward uniform: more random, more 'creative'.")

        # ---- top-k ----
        self.next_section("topk", skip_animations=False)
        base_bars = make_bars(base, cols)
        kcols = [SECONDARY, SECONDARY, SECONDARY] + [LINE] * 5
        kbars = make_bars(base, kcols)
        klab = label("top-k = 3", 22, SECONDARY).move_to(tlab)
        self.play(Transform(tlab, klab), Transform(bars, kbars), run_time=0.8)
        self.caption("Top-k keeps only the k most likely tokens, then renormalizes over them.")

        # ---- top-p ----
        self.next_section("topp", skip_animations=False)
        cum = np.cumsum(base)
        keep = cum <= 0.75
        keep[0] = True
        pcols = [SECONDARY if k else LINE for k in keep]
        pbars = make_bars(base, pcols)
        plab = label("top-p = 0.75  (nucleus)", 22, SECONDARY).move_to(tlab)
        self.play(Transform(tlab, plab), Transform(bars, pbars), run_time=0.8)
        self.caption("Top-p keeps the smallest set whose probability sums past p; its size adapts.")
        self.wait(0.3)
