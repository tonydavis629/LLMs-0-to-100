# Module 1: N-gram Language Models

## Overview

Build character-level and word-level n-gram models from scratch and use them to generate text. This recreates Shannon's 1948 experiment on statistical language modeling using *Alice in Wonderland* as the corpus.

## Setup

From the `exercises/` directory:

```
uv sync
```

## Running

```
uv run python module_01_introduction/src/main.py
```

The runner goes through the steps in order. Each step's header line carries a tag, and the step's output follows: the text your code generated, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 3: char_unigram() === CORRECT
 wgm oetgtee ,nei mae  tyah hur esseaca e ...
  CORRECT    returns exactly as many characters as requested
  CORRECT    only produces characters that appear in the training text
  CORRECT    samples in proportion to frequency (9 a's : 1 b gives ~90% a)
```

Run a single step with `--step`:

```
uv run python module_01_introduction/src/main.py --step 4
uv run python module_01_introduction/src/main.py --step ec
```

Step 1 always runs, because every other step needs the corpus it loads. To see the finished output, run the reference answers:

```
uv run python module_01_introduction/src/main.py --solution
```

## What to implement

`exercise.py` at the module root is the only file you edit. Fill in each `raise NotImplementedError(...)` line &mdash; each requires only one line of code.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `load_text()` | Read the corpus and strip the Project Gutenberg header/footer |
| 2 | `char_uniform()` | 0th-order model — uniform random characters |
| 3 | `char_unigram()` | 1st-order model — sample by character frequency |
| 4 | `build_char_ngram_model()` | Count (context &rarr; next char) pairs |
| 5 | `word_unigram()` | Word-frequency baseline |
| 6 | `build_word_ngram_model()` | Word-level n-gram counts |
| EC | `cross_entropy()` | Compute bits-per-character; perplexity falls out of it |

Every function in `exercise.py` has a blank to fill. Everything already written for you lives in `src/`:

- `src/sampling.py` &mdash; the loops that turn a count table into text (`generate_from_char_model()`, `generate_from_word_model()`)
- `src/text.py` &mdash; `tokenize()`
- `src/metrics.py` &mdash; `perplexity()`
- `src/main.py` &mdash; the runner

The tests live in `tests/`, one file per step (`test_step1_load_text.py` through `test_step6_word_ngram_model.py`, plus `test_extra_credit.py`). Each calls your function on small inputs with a known answer, so you can read the test for the step you are on to see exactly what is expected.

You do not need to edit any of them. Because the sampling loop is already provided, finishing `build_char_ngram_model()` immediately prints generated bigram and trigram text.

`solution/exercise.py` is the same file with every blank filled in.

## Data

`data/alice.txt` is a Project Gutenberg copy of *Alice in Wonderland*. `load_text()` is responsible for trimming the Gutenberg header/footer so only the book body is used as training data.

## Extra credit

Implement `cross_entropy()` to measure how well each n-gram model fits held-out text. `perplexity()` is provided and just calls `2 ** cross_entropy(...)`.

The runner's extra-credit section trains n-gram models of order 1 through 5 on the first 90% of the book and prints the bits per character and perplexity of each on the last 10%. Its tests cover three hand-computable cases (a perfect model scores 0 bits, a 50/50 guess costs 1 bit, an unseen character is smoothed to 1e-6) plus the trend on the real corpus.
