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

This creates the course virtual environment at the repository root.

### Set Up the Exercise Environment

The `exercises/` folder has its own separate `uv` environment:

```bash
cd exercises
uv sync
```

This creates `exercises/.venv`. Use it for exercise code only.

### Verify Your Setup

```bash
uv run python -c "print('Environment ready.')"
```

### Build the Slide Decks

Build every module's bundled lecture deck from source:

```bash
uv run python build_course.py
```

This writes `index.bundled.html` in each `slides/module_XX_*` folder that has slide source files.

## Course Structure

The course has 12 modules, each with a lecture (slides) and a coding exercise.

```
slides/          -- reveal.js presentations (open index.html in a browser)
  module_XX/
    manim/       -- source for lecture animations
exercises/       -- coding exercises with TODOs for you to complete
  module_XX/
    solution/    -- reference implementation (try the exercise first!)
```

### Working on an Exercise

Each module's exercise is in `exercises/module_XX_name/`. The source code is in `src/`.

```bash
# Shortest way to run Exercise 1
cd exercises
uv run python module_01_introduction/src/main.py
```

Fill in the TODOs in the exercise files. Check the solution in `exercises/module_XX/solution/` if you get stuck.

### Viewing Slides

Open the HTML file directly in your browser:

```bash
open slides/module_01_introduction/index.bundled.html     # macOS
xdg-open slides/module_01_introduction/index.bundled.html  # Linux
```

Use arrow keys to navigate. Press `S` for speaker notes.

## Modules

1. **Course Introduction** -- Information theory, Shannon, n-gram language models
2. **Perceptrons and Optimization** -- Neural networks, backpropagation, gradient descent
3. **Attention Mechanisms** -- Q/K/V, multi-head attention, positional encoding
4. **LLM Architectures** — Tokenization, transformer block, encoder/decoder/decoder-only, MoE
5. **Pretraining** -- Masked LM, autoregressive training
6. **Finetuning** -- InstructGPT, LoRA, parameter-efficient methods
7. **RL Post-Training** -- RLHF, GRPO, DPO
8. **Multimodal Models** -- Vision, audio, CLIP
9. **LLM Deployment** -- HBM bottleneck, MoE, vLLM
10. **Practical Applications** -- In-context learning, RAG, agents
11. **Evaluation and Benchmarking** -- Metrics, benchmarks, red teaming
12. **The Future of LLMs** -- Scaling laws, SSM/Mamba, diffusion
