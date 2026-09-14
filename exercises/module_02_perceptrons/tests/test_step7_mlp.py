"""Step 7: MLP.forward()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_mlp_forward(MLP) -> list[Check]:
    """Two layers: ReLU hidden layer, sigmoid output, checked against the layers' own math."""
    checks = []
    torch.manual_seed(0)
    model = MLP(input_size=2, hidden_size=4)
    x = torch.randn(5, 2)
    out = model(x)

    checks.append(
        ok("returns one probability per sample, shape (n_samples, 1)")
        if tuple(out.shape) == (5, 1)
        else bad("returns one probability per sample, shape (n_samples, 1)",
                 f"expected (5, 1), got {tuple(out.shape)}")
    )
    checks.append(
        ok("every output is strictly between 0 and 1")
        if bool(((out > 0) & (out < 1)).all())
        else bad("every output is strictly between 0 and 1",
                 f"min {float(out.min()):.4f}, max {float(out.max()):.4f} (did you apply sigmoid last?)")
    )

    # Recompute with the model's own layers and torch's relu/sigmoid
    with torch.no_grad():
        expected = torch.sigmoid(model.output(torch.relu(model.hidden(x))))
    checks.append(
        ok("equals sigmoid(output(relu(hidden(x)))) using the model's own layers")
        if out.shape == expected.shape and torch.allclose(out.detach(), expected, atol=1e-5)
        else bad("equals sigmoid(output(relu(hidden(x)))) using the model's own layers",
                 f"expected {expected.flatten().tolist()[:3]}..., got {out.detach().flatten().tolist()[:3]}...")
    )
    return checks


def check_mlp_training(results: dict) -> list[Check]:
    """After training on the XOR data, the hidden layer should make it solvable."""
    correct, total = results["mlp_acc"]
    pct = 100 * correct / total
    return [
        ok("XOR data: the MLP reaches at least 95% accuracy where the single neuron got 50%")
        if pct >= 95
        else bad("XOR data: the MLP reaches at least 95% accuracy where the single neuron got 50%",
                 f"got {correct}/{total} ({pct:.1f}%)")
    ]
