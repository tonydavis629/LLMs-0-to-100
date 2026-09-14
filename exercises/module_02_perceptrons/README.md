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

The runner goes through the steps in order. Each step's header line carries a tag, and the step's output follows: any training progress, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 4: update_parameters() === CORRECT
  CORRECT    moves against the gradient: w=[1,1], dw=[1,-1], lr=0.5 gives [0.5, 1.5]
  CORRECT    updates the bias the same way: b=0.5, db=0.2, lr=0.5 gives 0.4
  CORRECT    the learning rate scales the step size
  CORRECT    a zero gradient leaves the parameters unchanged
```

Run a single step with `--step` (1 to 7, or `ec`):

```
uv run python module_02_perceptrons/src/main.py --step 3
```

Plots are saved to `module_02_perceptrons/output/` by steps 5 and 7. `exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`; `sigmoid()` is provided in `src/activations.py` and you write `relu()` yourself. Run the finished answers with `--solution`:

```
uv run python module_02_perceptrons/src/main.py --solution
```

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code (or two at most).

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `forward()` | Single-neuron output: sigmoid of a weighted sum |
| 2 | `binary_cross_entropy()` | Loss function connecting back to Shannon entropy |
| 3 | `compute_gradients()` | Gradient of loss w.r.t. weights and bias (by hand) |
| 4 | `update_parameters()` | One gradient-descent step |
| 5 | (no new code) | Train the neuron from steps 1&ndash;4; watch it learn the linear data and fail on XOR |
| 6 | `relu()` | The ReLU activation: max(0, z) (hidden-layer nonlinearity) |
| 7 | `MLP.forward()` | Two-layer MLP: ReLU hidden layer, sigmoid output |
| EC | `SGD.step()` | Your own optimizer; train the MLP with it |

The single neuron does its gradients by hand (steps 3 and 4). The `MLP` is an `nn.Module`: you write only its `forward`, and PyTorch's autograd computes the gradients during training. `src/main.py` is the runner and `src/visualization.py` holds the plotting helpers &mdash; both are provided. The tests live in `tests/`, one file per step (`test_step1_forward.py` through `test_step7_mlp.py`, plus `test_extra_credit.py`). Each calls your function on small tensors with a known answer, so you can read the test for the step you are on to see exactly what is expected. You should only need to edit `exercise.py`.

## Data

Two pre-computed 2D datasets are provided in `data/`:

- `linear_separable.csv` &mdash; two clusters a single line can separate
- `non_linear_separable.csv` &mdash; XOR-like pattern requiring a nonlinear boundary

## Extra credit

Implement `SGD.step()` &mdash; your own optimizer. When you call `loss.backward()`, autograd fills in each parameter's `.grad`; an optimizer is the piece that then steps every parameter downhill (`p -= lr * p.grad`). The runner trains the MLP a second time using your optimizer. Its tests check one step against a hand-computed example, confirm the update happens in place, and compare a step against `torch.optim.SGD` on the same layer with the same gradients. The full implementation lives in `solution/exercise.py`.
