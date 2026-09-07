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

By default the runner trains and samples from every model. Pick one with `--model`:

```
uv run python module_01_introduction/src/main.py --model char3
```

The runner gracefully skips any step that still raises `NotImplementedError`, so you can run after each fill-in. To see the finished output, run the reference answers:

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

You do not need to edit any of them. Because the sampling loop is already provided, finishing `build_char_ngram_model()` immediately prints generated bigram and trigram text.

`solution/exercise.py` is the same file with every blank filled in.

## Data

`data/alice.txt` is a Project Gutenberg copy of *Alice in Wonderland*. `load_text()` is responsible for trimming the Gutenberg header/footer so only the book body is used as training data.

## Extra credit

Implement `cross_entropy()` to measure how well each n-gram model fits held-out text. `perplexity()` is provided and just calls `2 ** cross_entropy(...)`.
