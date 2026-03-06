# Module 1 Exercise: N-gram Language Models

Build character-level and word-level n-gram models to generate text,
recreating Shannon's 1948 experiment on statistical language modeling.

## Running the Exercise

From the repository root:

```bash
# Run the exercise (shows output for all implemented functions)
uv run python -m exercises.module_01_introduction.src.main \
    exercises/module_01_introduction/data/alice.txt

# Run a single model
uv run python -m exercises.module_01_introduction.src.main \
    exercises/module_01_introduction/data/alice.txt --model char3
```

Models: `uniform`, `char1`, `char2`, `char3`, `word1`, `word2`, `word3`, `all`

## What to Do

Open `src/ngrams.py` and fill in the lines marked with
`raise NotImplementedError(...)`. Each function needs only ONE line of code.

Run the exercise after each step to see your progress -- functions you
haven't implemented yet are skipped automatically.

## Data

The corpus is _Alice's Adventures in Wonderland_ from Project Gutenberg,
included at `data/alice.txt`.

## Solution

The complete reference implementation is in `solution/src/`.
