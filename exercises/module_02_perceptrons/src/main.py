"""
Module 2 Exercise runner: Perceptrons and Neural Networks (PyTorch)

Run with:
    uv run python module_02_perceptrons/src/main.py

Every step is tagged on its header line, then its output follows: any
training progress your code produced and the result of each test in tests/.
The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-7, or "ec" for extra credit).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `import exercise`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also ensure src/ is on the path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

import matplotlib
from exercise import (
    MLP,
    SGD,
    binary_cross_entropy,
    compute_gradients,
    forward,
    relu,
    update_parameters,
)
from src.activations import sigmoid

# One test file per step lives in tests/
from tests.test_extra_credit import check_sgd_step, check_sgd_training
from tests.test_step1_forward import check_forward
from tests.test_step2_bce import check_binary_cross_entropy
from tests.test_step3_gradients import check_compute_gradients
from tests.test_step4_update import check_update_parameters
from tests.test_step5_single_neuron import check_single_neuron
from tests.test_step6_relu import check_relu
from tests.test_step7_mlp import check_mlp_forward, check_mlp_training
from visualization import (
    load_csv,
    plot_decision_boundary,
    plot_loss_curve,
    save_comparison,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths to the bundled datasets. Walk up from this file until we find data/
# (works from both src/main.py and solution/src/main.py).
_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
if not (_MODULE_DIR / "data").exists():
    _MODULE_DIR = _MODULE_DIR.parent
DATA_DIR = _MODULE_DIR / "data"
LINEAR_DATA = DATA_DIR / "linear_separable.csv"
NONLINEAR_DATA = DATA_DIR / "non_linear_separable.csv"
OUTPUT_DIR = _MODULE_DIR / "output"

def load_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CSV and return (X, y) as float32 tensors.

    Args:
        path: Path to a bundled CSV dataset.

    Returns:
        (X, y): feature tensor of shape (n_samples, 2) and label tensor of shape (n_samples,).
    """
    X_np, y_np = load_csv(str(path))
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    return X, y

# ---------------------------------------------------------------------------
# Single neuron: hand-written forward / loss / gradients / update
# ---------------------------------------------------------------------------

def train_perceptron(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.5,
    epochs: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Train a single neuron with full-batch gradient descent (no autograd).

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        learning_rate: Step size for each gradient-descent update.
        epochs: Number of full passes through the dataset.

    Returns:
        (weights, bias, losses): trained parameters and the loss history.
    """
    torch.manual_seed(42)
    weights = torch.randn(2) * 0.1
    bias = torch.zeros(())  # 0-dim tensor
    losses: list[float] = []

    for epoch in range(epochs):
        y_pred = forward(X, weights, bias)
        loss = binary_cross_entropy(y, y_pred).mean()
        losses.append(float(loss))

        dw, db = compute_gradients(X, y, y_pred)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={loss:.4f}")

    return weights, bias, losses

def neuron_accuracy(X: torch.Tensor, y: torch.Tensor, weights, bias) -> tuple[int, int]:
    """Count correctly classified samples for the single neuron.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        weights: Learned weight vector, shape (2,).
        bias: Learned scalar bias.

    Returns:
        (correct, total): number correct and total number of samples.
    """
    with torch.no_grad():
        preds = (sigmoid(X @ weights + bias) >= 0.5).float()
    correct = int((preds == y).sum())
    return correct, X.shape[0]

def neuron_predict(weights, bias):
    """Return a vectorized predict_fn(grid) -> probabilities for plotting.

    Args:
        weights: Learned weight vector, shape (2,).
        bias: Learned scalar bias.

    Returns:
        A function that maps a NumPy grid of points to predicted probabilities.
    """
    def predict(grid):
        g = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            return sigmoid(g @ weights + bias).numpy()
    return predict

# ---------------------------------------------------------------------------
# MLP: autograd + an optimizer (torch's, or the student's for extra credit)
# ---------------------------------------------------------------------------

def train_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden_size: int = 8,
    learning_rate: float = 1.0,
    epochs: int = 500,
    use_custom_optimizer: bool = False,
    verbose: bool = True,
) -> tuple[MLP, list[float]]:
    """Train the MLP with autograd.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        hidden_size: Number of neurons in the hidden layer.
        learning_rate: Step size for the optimizer.
        epochs: Number of training passes.
        use_custom_optimizer: Whether to use the exercise's SGD class.
        verbose: Whether to print progress every 100 epochs.

    Returns:
        (model, losses): trained model and per-epoch loss history.
    """
    torch.manual_seed(42)
    model = MLP(input_size=2, hidden_size=hidden_size)
    if use_custom_optimizer:
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    y_col = y.unsqueeze(1)  # shape (N, 1) to match the model output
    losses: list[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)                                  # forward (student code)
        loss = binary_cross_entropy(y_col, output).mean()  # average over the batch
        loss.backward()                                    # autograd: every gradient
        optimizer.step()                                   # nudge parameters downhill

        losses.append(loss.item())
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={loss.item():.4f}")

    return model, losses

def mlp_accuracy(X: torch.Tensor, y: torch.Tensor, model: MLP) -> tuple[int, int]:
    """Count correctly classified samples for the MLP.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        model: Trained MLP model.

    Returns:
        (correct, total): number correct and total number of samples.
    """
    with torch.no_grad():
        preds = (model(X).squeeze(1) >= 0.5).float()
    correct = int((preds == y).sum())
    return correct, X.shape[0]

def mlp_predict(model: MLP):
    """Return a vectorized predict_fn(grid) -> probabilities for plotting.

    Args:
        model: Trained MLP model.

    Returns:
        A function that maps a NumPy grid of points to predicted probabilities.
    """
    def predict(grid):
        g = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            return model(g).squeeze(1).numpy()
    return predict

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1() -> str:
    return run_step("Step 1: forward()", lambda: None, lambda: check_forward(forward))


def step_2() -> str:
    return run_step("Step 2: binary_cross_entropy()", lambda: None,
                    lambda: check_binary_cross_entropy(binary_cross_entropy))


def step_3() -> str:
    return run_step("Step 3: compute_gradients()", lambda: None,
                    lambda: check_compute_gradients(compute_gradients))


def step_4() -> str:
    return run_step("Step 4: update_parameters()", lambda: None,
                    lambda: check_update_parameters(update_parameters))


def step_5(X_lin, y_lin, X_nl, y_nl, results: dict) -> str:
    """No new code: train the neuron from Steps 1-4 on both datasets."""

    def show():
        # Part A: linearly separable data, where a single neuron should succeed
        print(f"Linear data: {len(y_lin)} samples from linear_separable.csv")
        weights, bias, losses = train_perceptron(X_lin, y_lin, learning_rate=0.5, epochs=100)
        print(f"  Final weights: [{weights[0]:.4f}, {weights[1]:.4f}], bias: {float(bias):.4f}")
        correct, total = neuron_accuracy(X_lin, y_lin, weights, bias)
        print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        results["lin_losses"], results["lin_acc"] = losses, (correct, total)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(neuron_predict(weights, bias), X_lin.numpy(), y_lin.numpy(),
                               title="Perceptron: Linear Data", ax=ax1)
        plot_loss_curve(losses, title="Training Loss", ax=ax2)
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "step5_linear_perceptron.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved plot to output/step5_linear_perceptron.png")

        # Part B: the same neuron on XOR-like data, where no line can separate the classes
        print(f"XOR data: {len(y_nl)} samples from non_linear_separable.csv")
        weights_nl, bias_nl, losses_nl = train_perceptron(X_nl, y_nl, learning_rate=0.5, epochs=100)
        print(f"  Final weights: [{weights_nl[0]:.4f}, {weights_nl[1]:.4f}], bias: {float(bias_nl):.4f}")
        correct, total = neuron_accuracy(X_nl, y_nl, weights_nl, bias_nl)
        print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        results["nl_losses"], results["nl_acc"] = losses_nl, (correct, total)
        results["neuron_nl_params"] = (weights_nl, bias_nl)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(neuron_predict(weights_nl, bias_nl), X_nl.numpy(), y_nl.numpy(),
                               title="Perceptron: XOR Data (Fails)", ax=ax1)
        plot_loss_curve(losses_nl, title="Training Loss (Plateaus)", ax=ax2)
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "step5_nonlinear_perceptron.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved plot to output/step5_nonlinear_perceptron.png")

    return run_step("Step 5: train the single neuron (Steps 1-4 together)", show,
                    lambda: check_single_neuron(results))


def step_6() -> str:
    return run_step("Step 6: relu()", lambda: None, lambda: check_relu(relu))


def step_7(X_nl, y_nl, results: dict) -> str:
    """Train the MLP on the XOR data with autograd and torch.optim.SGD."""

    def show():
        model, losses = train_mlp(X_nl, y_nl, hidden_size=8, learning_rate=1.0, epochs=500)
        correct, total = mlp_accuracy(X_nl, y_nl, model)
        print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        results["mlp_acc"] = (correct, total)

        # Side-by-side decision boundaries, if the single neuron from Step 5 is available
        if "neuron_nl_params" in results:
            save_comparison(
                neuron_predict(*results["neuron_nl_params"]),
                mlp_predict(model),
                X_nl.numpy(), y_nl.numpy(),
                filepath=str(OUTPUT_DIR / "step7_comparison.png"),
            )
            print("  Saved comparison plot to output/step7_comparison.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_loss_curve(losses, title="MLP Training Loss", ax=ax)
        fig.savefig(str(OUTPUT_DIR / "step7_mlp_loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved MLP loss plot to output/step7_mlp_loss.png")

    return run_step("Step 7: MLP.forward()", show,
                    lambda: check_mlp_forward(MLP) + check_mlp_training(results))


def step_extra(X_nl, y_nl, results: dict) -> str:
    """Train the same MLP again, with the student's optimizer in place of torch's."""

    def show():
        # Probe the optimizer first so an unfinished step() reports its own TODO
        probe = torch.zeros(1, requires_grad=True)
        probe.grad = torch.zeros(1)
        SGD([probe], lr=0.1).step()

        try:
            model, _ = train_mlp(X_nl, y_nl, hidden_size=8, learning_rate=1.0, epochs=500,
                                 use_custom_optimizer=True)
        except NotImplementedError:
            raise NotImplementedError("needs Step 7 (MLP.forward) to train the MLP with your optimizer")
        correct, total = mlp_accuracy(X_nl, y_nl, model)
        print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        results["ec_acc"] = (correct, total)

    return run_step("Extra Credit: SGD.step()", show,
                    lambda: check_sgd_step(SGD) + check_sgd_training(results))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_CHOICES = ["1", "2", "3", "4", "5", "6", "7", "ec"]


def main():
    parser = argparse.ArgumentParser(description="Perceptrons and neural networks")
    parser.add_argument("--step", choices=[*STEP_CHOICES, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = STEP_CHOICES if args.step == "all" else [args.step]

    OUTPUT_DIR.mkdir(exist_ok=True)
    X_lin, y_lin = load_tensors(LINEAR_DATA)
    X_nl, y_nl = load_tensors(NONLINEAR_DATA)
    results: dict = {}  # training outcomes shared between steps

    for step in steps:
        if step == "1":
            step_1()
        elif step == "2":
            step_2()
        elif step == "3":
            step_3()
        elif step == "4":
            step_4()
        elif step == "5":
            step_5(X_lin, y_lin, X_nl, y_nl, results)
        elif step == "6":
            step_6()
        elif step == "7":
            step_7(X_nl, y_nl, results)
        elif step == "ec":
            step_extra(X_nl, y_nl, results)

if __name__ == "__main__":
    main()
