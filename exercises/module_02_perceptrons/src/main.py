"""
Module 2 Exercise runner: Perceptrons and Neural Networks (PyTorch)

Run with:
    uv run python module_02_perceptrons/src/main.py

Each step below runs one piece of exercise.py, then tests it. Read top to
bottom, the steps build a classifier twice:

    1-4. one neuron, by hand: forward pass, loss, gradients, update
    5.   train that neuron: it learns a straight line, so it fails on XOR
    6-7. a small network with a ReLU hidden layer, trained with autograd,
         which solves XOR
    ec.  extra credit: your own SGD optimizer in place of PyTorch's

Add --step N to run one step (1-7, or "ec" for extra credit).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    MLP,
    SGD,
    binary_cross_entropy,
    compute_gradients,
    forward,
    relu,
    update_parameters,
)
from src.evaluation import load_dataset, mlp_accuracy, mlp_predict, neuron_accuracy, neuron_predict
from src.reporting import run_step
from src.visualization import save_boundary_and_loss, save_comparison, save_loss_curve

# One test file per step lives in tests/
from tests.test_extra_credit import check_sgd_step, check_sgd_training
from tests.test_step1_forward import check_forward
from tests.test_step2_bce import check_binary_cross_entropy
from tests.test_step3_gradients import check_compute_gradients
from tests.test_step4_update import check_update_parameters
from tests.test_step5_single_neuron import check_single_neuron
from tests.test_step6_relu import check_relu
from tests.test_step7_mlp import check_mlp_forward, check_mlp_training

MODULE_DIR = Path(__file__).resolve().parent.parent
LINEAR_DATA = MODULE_DIR / "data" / "linear_separable.csv"         # two blobs a line can split
NONLINEAR_DATA = MODULE_DIR / "data" / "non_linear_separable.csv"  # XOR: no line can split it
OUTPUT_DIR = MODULE_DIR / "output"

# Step 5 keeps the neuron it trained on XOR here, so Step 7 can plot it next to the MLP
xor_neuron = None


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_perceptron(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.5,
    epochs: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Train a single neuron with full-batch gradient descent (no autograd).

    Every line of the loop calls a function you write in Steps 1-4.

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
        y_pred = forward(X, weights, bias)                     # Step 1: predict
        loss = binary_cross_entropy(y, y_pred).mean()          # Step 2: how wrong, on average
        losses.append(float(loss))

        dw, db = compute_gradients(X, y, y_pred)               # Step 3: which way is downhill
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)  # Step 4: step

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={loss:.4f}")

    return weights, bias, losses


def train_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden_size: int = 8,
    learning_rate: float = 1.0,
    epochs: int = 500,
    use_custom_optimizer: bool = False,
) -> tuple[MLP, list[float]]:
    """Train the MLP with autograd.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        hidden_size: Number of neurons in the hidden layer.
        learning_rate: Step size for the optimizer.
        epochs: Number of training passes.
        use_custom_optimizer: Whether to use your SGD class (extra credit) instead of torch's.

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
        output = model(X)                                  # forward (Step 7)
        loss = binary_cross_entropy(y_col, output).mean()  # average over the batch
        loss.backward()                                    # autograd: every gradient
        optimizer.step()                                   # nudge parameters downhill

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={loss.item():.4f}")

    return model, losses


# ---------------------------------------------------------------------------
# Steps 1-4: one neuron, by hand
# ---------------------------------------------------------------------------
# These steps only run the tests; Step 5 puts the four pieces to work.


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


# ---------------------------------------------------------------------------
# Step 5: train the single neuron
# ---------------------------------------------------------------------------


def train_and_plot_neuron(data_path: Path, boundary_title: str, loss_title: str, plot_name: str):
    """Train the neuron on one dataset, print how it did, and save its plot.

    Returns (weights, bias, losses, correct, total).
    """
    X, y = load_dataset(data_path)
    weights, bias, losses = train_perceptron(X, y, learning_rate=0.5, epochs=100)
    print(f"  Final weights: [{weights[0]:.4f}, {weights[1]:.4f}], bias: {float(bias):.4f}")
    correct, total = neuron_accuracy(X, y, weights, bias)
    print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    save_boundary_and_loss(neuron_predict(weights, bias), X.numpy(), y.numpy(), losses,
                           boundary_title, loss_title, filepath=str(OUTPUT_DIR / plot_name))
    print(f"  Saved plot to output/{plot_name}")
    return weights, bias, losses, correct, total


def step_5() -> str:
    """No new code: train the neuron from Steps 1-4 on both datasets."""
    runs = {}  # dataset -> (losses, correct, total), filled in by show() for the tests

    def show():
        global xor_neuron

        # Part A: linearly separable data, where a single neuron should succeed
        print(f"Linear data: {len(load_dataset(LINEAR_DATA)[1])} samples from linear_separable.csv")
        _, _, losses, correct, total = train_and_plot_neuron(
            LINEAR_DATA, "Perceptron: Linear Data", "Training Loss", "step5_linear_perceptron.png")
        runs["linear"] = (losses, correct, total)

        # Part B: the same neuron on XOR-like data, where no line can separate the classes
        print(f"XOR data: {len(load_dataset(NONLINEAR_DATA)[1])} samples from non_linear_separable.csv")
        weights, bias, losses, correct, total = train_and_plot_neuron(
            NONLINEAR_DATA, "Perceptron: XOR Data (Fails)", "Training Loss (Plateaus)",
            "step5_nonlinear_perceptron.png")
        runs["xor"] = (losses, correct, total)
        xor_neuron = (weights, bias)

    return run_step("Step 5: train the single neuron (Steps 1-4 together)", show,
                    lambda: check_single_neuron(runs["linear"], runs["xor"]))


# ---------------------------------------------------------------------------
# Steps 6-7: a hidden layer solves XOR
# ---------------------------------------------------------------------------


def step_6() -> str:
    return run_step("Step 6: relu()", lambda: None, lambda: check_relu(relu))


def step_7() -> str:
    """Train the MLP on the XOR data with autograd and torch.optim.SGD."""
    accuracy = []  # (correct, total), filled in by show() for the tests

    def show():
        X, y = load_dataset(NONLINEAR_DATA)
        model, losses = train_mlp(X, y, hidden_size=8, learning_rate=1.0, epochs=500)
        correct, total = mlp_accuracy(X, y, model)
        print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        accuracy[:] = [correct, total]

        # Side-by-side decision boundaries, if Step 5 trained the single neuron this run
        if xor_neuron is not None:
            save_comparison(neuron_predict(*xor_neuron), mlp_predict(model), X.numpy(), y.numpy(),
                            filepath=str(OUTPUT_DIR / "step7_comparison.png"))
            print("  Saved comparison plot to output/step7_comparison.png")
        save_loss_curve(losses, "MLP Training Loss", filepath=str(OUTPUT_DIR / "step7_mlp_loss.png"))
        print("  Saved MLP loss plot to output/step7_mlp_loss.png")

    return run_step("Step 7: MLP.forward()", show,
                    lambda: check_mlp_forward(MLP) + check_mlp_training(*accuracy))


# ---------------------------------------------------------------------------
# Extra credit: your own optimizer
# ---------------------------------------------------------------------------


def step_extra() -> str:
    """Train the same MLP again, with your optimizer in place of torch's."""
    accuracy = []  # (correct, total), filled in by show() for the tests

    def show():
        # Try your optimizer on one tiny parameter first, so an unfinished step() reports its own TODO
        probe = torch.zeros(1, requires_grad=True)
        probe.grad = torch.zeros(1)
        SGD([probe], lr=0.1).step()

        X, y = load_dataset(NONLINEAR_DATA)
        try:
            model, _ = train_mlp(X, y, hidden_size=8, learning_rate=1.0, epochs=500,
                                 use_custom_optimizer=True)
        except NotImplementedError:
            raise NotImplementedError("needs Step 7 (MLP.forward) to train the MLP with your optimizer")
        correct, total = mlp_accuracy(X, y, model)
        print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        accuracy[:] = [correct, total]

    return run_step("Extra Credit: SGD.step()", show,
                    lambda: check_sgd_step(SGD) + check_sgd_training(*accuracy))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4, "5": step_5,
         "6": step_6, "7": step_7, "ec": step_extra}


def main():
    parser = argparse.ArgumentParser(description="Perceptrons and neural networks")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, step in STEPS.items():
        if args.step in ("all", name):
            step()


if __name__ == "__main__":
    main()
