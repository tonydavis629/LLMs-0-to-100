# Module 2: Perceptrons and Optimization

## Overview

Build a classifier from scratch using only numpy. You will implement a single-neuron classifier (perceptron), train it on 2D data, and visualize its decision boundary. Then you will see it fail on non-linearly-separable data and fix the problem by adding a ReLU hidden layer (MLP).

## Setup

From the `exercises/` directory:

```
uv sync
```

## Running

```
uv run python module_02_perceptrons/src/main.py
```

Output plots are saved to `module_02_perceptrons/output/`. The runner gracefully skips any step that still raises `NotImplementedError`, so you can run after each fill-in.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `forward()` | Single neuron output: sigmoid(dot(x, w) + b) |
| 2 | `binary_cross_entropy()` | Loss function connecting back to Shannon entropy |
| 3 | `compute_gradients()` | Gradient of loss w.r.t. weights and bias |
| 4 | `update_parameters()` | One gradient descent step |
| 5 | (provided) | See the single neuron fail on the XOR-like dataset |
| 6 | `relu()` | The ReLU activation: max(0, z) (hidden-layer nonlinearity) |
| 7 | `mlp_forward()` | Two-layer MLP: ReLU hidden layer, sigmoid output |
| EC | `mlp_gradients()` | Backpropagation through the ReLU MLP |

`src/main.py` is the runner and `src/visualization.py` holds the plotting helpers &mdash; both are provided. You should only need to edit `exercise.py`.

## Data

Two pre-computed 2D datasets are provided in `data/`:

- `linear_separable.csv` &mdash; two clusters a single line can separate
- `non_linear_separable.csv` &mdash; XOR-like pattern requiring a nonlinear boundary

## Extra credit

Implement `mlp_gradients()` to compute backpropagation through the ReLU hidden layer. The only twist versus a sigmoid layer is the activation derivative: ReLU's derivative is 1 where the pre-activation is positive and 0 otherwise. The full derivation lives in `solution/exercise.py`.
