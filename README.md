# LLMs 0 to 100

A hands-on course covering Large Language Models from fundamentals to deployment.

## Getting Started

### Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A terminal (macOS Terminal, Linux shell, or Windows WSL)
- A text editor or IDE (VS Code recommended)

### Install uv

If you don't have `uv` installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify it works:

```bash
uv --version
```

### Set Up the Course Environment

Clone the repository and set up the Python environment:

```bash
git clone https://github.com/tonydavis629/LLMs-0-to-100
cd LLMs-0-to-100
uv sync
```


### Verify Your Setup

```bash
uv run python -c "import torch, print('Environment ready.')"
```

### Build the Slide Decks

Build every module's bundled lecture deck from source:

```bash
uv run python build_course.py
```

This builds the presentation for each lecture in the `slides` folder.

## Course Structure

The course has 12 modules, each with a lecture and a coding exercise.

```
.venv/           # the one shared environment (created by `uv sync`)
slides/          # reveal.js presentations
  module_XX.html # built, self-contained lecture deck (open in a browser)
  module_XX/
    source/      # slide source partials, config, per-module styles
    images/      # figures referenced by the deck
    manim/       # source for lecture animations
exercises/       # coding exercises with TODOs for you to complete
  module_XX/
    exercise.py  # the only file you edit
    src/         # runner and helpers (internal plumbing)
    data/        # bundled sample data, including the trained checkpoints
    solution/    # reference implementation (try the exercise first!)
```

## Disable Autocomplete

Since this is an educational course, you should be disabling AI autocomplete so you can learn from the exercises.

Add at the top of VSCode settings.json:

`"chat.disableAIFeatures": true,`

And disable any autocomplete extensions like TabNine.

## Working on an Exercise

Each module's exercise is in `exercises/module_XX_name/`. You only edit `exercise.py` at the module root; `src/` holds the runner and helpers.

```bash
# Run Exercise 1 from the repository root
uv run python exercises/module_01_introduction/src/main.py
```

Fill in the TODOs in the exercise files. Check the solution in `exercises/module_XX/solution/` if you get stuck.

### Viewing Slides

Open the HTML file from `slides/` directly in your browser

## Modules

1. **Course Introduction** &mdash; Information theory, Shannon, n-gram language models
2. **Perceptrons and Optimization** &mdash; Neural networks, backpropagation, gradient descent
3. **Attention Mechanisms** &mdash; Q/K/V, multi-head attention, positional encoding
4. **LLM Architectures** &mdash; Tokenization, transformer block, encoder/decoder/decoder-only, MoE
5. **Pretraining** &mdash; Masked LM, autoregressive training
6. **Finetuning** &mdash; InstructGPT, LoRA, parameter-efficient methods
7. **RL Post-Training** &mdash; RLHF, GRPO, DPO
8. **Multimodal Models** &mdash; Vision, audio, CLIP
9. **Evaluation and Benchmarking** &mdash; Metrics, benchmarks, red teaming
10. **LLM Deployment** &mdash; HBM bottleneck, MoE, vLLM
11. **Practical Applications** &mdash; In-context learning, RAG, agents
12. **The Future of LLMs** &mdash; Scaling laws, SSM/Mamba, diffusion
