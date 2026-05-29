:::divider id="divider-setup" title="Getting Set Up" sub="Python environment with uv"
:::

---

<!-- .slide: id="setup-uv" -->

## Setting Up Your Environment

:::columns cols="2"
### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

uv is a fast Python package manager. It replaces pip, venv, and pyenv. <!-- .element: class="text-muted" style="font-size: 13pt; margin-top: 10px;" -->
+++
### 2. Set up the course

```bash
# Clone and sync
git clone https://github.com/tonydavis629/LLMs-0-to-100
cd LLMs-0-to-100
uv sync

# Verify
uv run python -c "print('Ready.')"
```

One environment for all modules. Run `uv sync` once. <!-- .element: class="text-muted" style="font-size: 13pt; margin-top: 10px;" -->
:::
