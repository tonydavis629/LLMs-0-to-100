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

The runner gracefully skips any step that still raises `NotImplementedError`, so you can run after each fill-in.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `load_text()` | Read the corpus and strip the Project Gutenberg header/footer |
| 2 | `char_uniform()` | 0th-order model — uniform random characters |
| 3 | `char_unigram()` | 1st-order model — sample by character frequency |
| 4 | `build_char_ngram_model()` | Count (context &rarr; next char) pairs |
| 5 | `generate_from_char_model()` | Sample next chars given the trailing context |
| 6 | `word_unigram()` | Word-frequency baseline |
| 7 | `build_word_ngram_model()` | Word-level n-gram counts |
| 8 | `generate_from_word_model()` | Sample next words given the trailing context |
| EC | `cross_entropy()` | Compute bits-per-character; perplexity falls out of it |

## Data

`data/alice.txt` is a Project Gutenberg copy of *Alice in Wonderland*. `load_text()` is responsible for trimming the Gutenberg header/footer so only the book body is used as training data.

## Extra credit

Implement `cross_entropy()` to measure how well each n-gram model fits held-out text. `perplexity()` is provided and just calls `2 ** cross_entropy(...)`.
