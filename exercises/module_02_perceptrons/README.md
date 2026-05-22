# Module 2: Perceptrons and Optimization

## Overview

Build a classifier with PyTorch. You will implement a single-neuron classifier (perceptron) with hand-written gradient math, train it on 2D data, and visualize its decision boundary. You will then watch it fail on non-linearly-separable (XOR-like) data and fix the problem by building a multi-layer perceptron (`MLP`, an `nn.Module`) with a ReLU hidden layer, trained with PyTorch autograd.

We use PyTorch tensors throughout, but you still write your own activation functions, the single neuron's gradients, and (for the extra credit) your own optimizer, so nothing important is hidden behind the framework.

## Setup

From the `exercises/` directory:

```
uv sync
```

PyTorch installs as the CPU build from `https://download.pytorch.org/whl/cpu` (pinned in `pyproject.toml`).

## Running

```
uv run python module_02_perceptrons/src/main.py
```

Output plots are saved to `module_02_perceptrons/output/`. The runner gracefully skips any step that still raises `NotImplementedError`, so you can run after each fill-in.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code (or two at most).

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `forward()` | Single-neuron output: sigmoid of a weighted sum |
| 2 | `binary_cross_entropy()` | Loss function connecting back to Shannon entropy |
| 3 | `compute_gradients()` | Gradient of loss w.r.t. weights and bias (by hand) |
| 4 | `update_parameters()` | One gradient-descent step |
| 5 | (provided) | See the single neuron fail on the XOR-like dataset |
| 6 | `relu()` | The ReLU activation: max(0, z) (hidden-layer nonlinearity) |
| 7 | `MLP.forward()` | Two-layer MLP: ReLU hidden layer, sigmoid output |
| EC | `SGD.step()` | Your own optimizer; train the MLP with it |

The single neuron does its gradients by hand (steps 3 and 4). The `MLP` is an `nn.Module`: you write only its `forward`, and PyTorch's autograd computes the gradients during training. `src/main.py` is the runner and `src/visualization.py` holds the plotting helpers &mdash; both are provided. You should only need to edit `exercise.py`.

## Data

Two pre-computed 2D datasets are provided in `data/`:

- `linear_separable.csv` &mdash; two clusters a single line can separate
- `non_linear_separable.csv` &mdash; XOR-like pattern requiring a nonlinear boundary

## Extra credit

Implement `SGD.step()` &mdash; your own optimizer. When you call `loss.backward()`, autograd fills in each parameter's `.grad`; an optimizer is the piece that then steps every parameter downhill (`p -= lr * p.grad`). The runner trains the MLP a second time using your optimizer and checks it matches `torch.optim.SGD`. The full implementation lives in `solution/exercise.py`.
