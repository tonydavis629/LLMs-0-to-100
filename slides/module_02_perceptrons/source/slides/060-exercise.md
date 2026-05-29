:::divider id="divider-exercise" title="Exercise" sub="Text Classifier with Decision Boundary Visualization"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_02_perceptrons/exercise.py` and fill in the `NotImplementedError` lines. Run after each step &mdash; unfinished functions are skipped automatically. <!-- .element: class="text-lg" -->

```bash
# Run the exercise (skips any not yet implemented)
cd exercises
uv run python module_02_perceptrons/src/main.py
```

The exercise trains a single-neuron classifier and an MLP on 2D data, visualizing how they learn to separate classes. Check the `output/` directory for plots after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Perceptrons and Neural Networks

Build it with PyTorch tensors: train a single neuron on linearly separable data, watch it fail on XOR-like data, then fix it with a ReLU hidden layer. <!-- .element: class="text-lg" -->

Each function is mostly written for you &mdash; you fill in **one key line**. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-data-viz" -->

## The Datasets

<div style="display:flex; align-items:center; justify-content:center; flex:1;">
  <img src="images/datasets.png" alt="Linearly separable and XOR-like exercise datasets" style="max-width: 94%; max-height: 500px; border: 1px solid var(--line-color); border-radius: 8px;">
</div>

---

:::step id="exercise-step1-code" title="Step 1: forward()"
```python
def forward(X: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute the output of a single neuron for a whole batch.

    Args:
        X: Input tensor, shape (n_samples, 2).
        weights: Weight vector, shape (2,).
        bias: Scalar bias tensor.

    Returns:
        A probability tensor of shape (n_samples,) with values in (0, 1).
    """
    # TODO: Compute the neuron's output in one line
    raise NotImplementedError("TODO: implement the forward pass")
```
+++
**Hint:** Use the batch matrix-vector product `X @ weights`, add `bias`, then pass the result to `sigmoid()`.
+++
**Answer:**

```python
return sigmoid(X @ weights + bias)
```
:::

---

:::step id="exercise-step2-code" title="Step 2: binary_cross_entropy()"
```python
def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the binary cross-entropy loss: -[y log(p) + (1 - y) log(1 - p)].

    Args:
        y_true: True label(s), 0 or 1 (scalar or tensor).
        y_pred: Predicted probability/probabilities in (0, 1).

    Returns:
        The per-sample loss (same shape as the inputs); lower is better.
    """
    # Clip predictions away from 0 and 1 to avoid log(0) = -infinity
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)

    # TODO: Compute and return the BCE loss using the formula in the docstring
    raise NotImplementedError("TODO: implement binary cross-entropy")
```
+++
**Hint:** Use `torch.log(y_pred)` and `torch.log(1 - y_pred)`, matching the formula in the docstring.
+++
**Answer:**

```python
return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
```
:::

---

:::step id="exercise-step3-code" title="Step 3: compute_gradients()"
```python
def compute_gradients(
    X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For sigmoid + BCE the chain rule simplifies to:
        error = y_pred - y_true
        dw    = X.T @ error / n
        db    = error.mean()
    """
    # TODO: Compute the averaged gradients using the formulas in the docstring
    raise NotImplementedError("TODO: implement gradient computation")
```
+++
**Hint:** The error is `y_pred - y_true`. The weight gradient is `X.T @ error` averaged over the batch; the bias gradient is the mean error.
+++
**Answer:**

```python
error = y_pred - y_true
n = X.shape[0]
dw = X.T @ error / n
db = error.mean()
return (dw, db)
```
:::

---

:::step id="exercise-step4-code" title="Step 4: update_parameters()"
```python
def update_parameters(
    weights: torch.Tensor,
    bias: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the weights and bias with one gradient-descent step.

    Args:
        weights: Current weight vector, shape (2,).
        bias: Current bias (0-dim tensor).
        dw: Gradient w.r.t. weights, shape (2,).
        db: Gradient w.r.t. bias (0-dim tensor).
        learning_rate: Step size, e.g. 0.5.

    Returns:
        (new_weights, new_bias): the updated parameters.
    """
    # TODO: Apply the gradient-descent update rule
    raise NotImplementedError("TODO: implement parameter update")
```
+++
**Hint:** Subtract `learning_rate * dw` from weights, `learning_rate * db` from bias.
+++
**Answer:**

```python
return (weights - learning_rate * dw, bias - learning_rate * db)
```
:::

---

:::terminal id="exercise-step5-output" title="Steps 1&ndash;4: Single Neuron Output" cmd="uv run python module_02_perceptrons/src/main.py" caption="99.3% on linear data, but only 50% on XOR &mdash; a single neuron is a linear classifier."
<span class="header">============================================================
PART 1: Single Neuron on Linearly Separable Data
============================================================</span>
<span class="success">Loaded 150 samples from linear_separable.csv</span>
  Epoch  20/100  loss=0.0753
  Epoch  40/100  loss=0.0536
  Epoch  60/100  loss=0.0457
  Epoch  80/100  loss=0.0416
  Epoch 100/100  loss=0.0391

Final weights: [2.4276, 1.7352], bias: 0.1047
<span class="success">Accuracy: 149/150 (99.3%)</span>
Saved plot to output/step5_linear_perceptron.png

<span class="header">============================================================
PART 2: Single Neuron on Non-Linearly Separable Data (XOR)
============================================================</span>
<span class="success">Loaded 160 samples from non_linear_separable.csv</span>
  Epoch  20/100  loss=0.6926
  Epoch  40/100  loss=0.6926
  Epoch  60/100  loss=0.6926
  Epoch  80/100  loss=0.6926
  Epoch 100/100  loss=0.6926

Final weights: [0.0070, -0.0399], bias: -0.0016
<span class="t-fail">Accuracy: 80/160 (50.0%)</span>
(A single neuron cannot solve this non-linearly-separable problem.)
Saved plot to output/step6_nonlinear_perceptron.png

<span class="header">============================================================
PART 3: MLP on Non-Linearly Separable Data
============================================================</span>
<span class="skipped">  [skipped: TODO: implement the MLP forward pass]</span>
:::

---

<!-- .slide: id="exercise-step6-context" -->

## Step 5: Why the Single Neuron Fails

PART 2 ran the **same** trained neuron on the XOR-like dataset. No new code &mdash; the runner does it automatically. <!-- .element: class="text-lg" -->

:::note
**50% accuracy is random guessing.** The loss flat-lines at **0.693**, which is exactly $\ln 2$ &mdash; the cross-entropy of a fair coin. <!-- .element: class="text-lg" style="margin:0;" -->
:::

No straight line separates XOR, so the best a single linear neuron can do is shrug and predict 0.5 for everything. The fix is a **hidden layer with a nonlinearity**. <!-- .element: class="text-lg" style="margin-top: 18px;" -->

---

:::step id="exercise-step6-code" title="Step 6: relu()"
```python
def relu(z: torch.Tensor) -> torch.Tensor:
    """The ReLU activation: max(0, z).

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with negatives replaced by 0.
    """
    # TODO: Return the elementwise max of 0 and z in one line
    raise NotImplementedError("TODO: implement the ReLU activation")
```
+++
**Hint:** `torch.clamp(z, min=0.0)` keeps positive entries and clamps negative entries to zero.
+++
**Answer:**

```python
return torch.clamp(z, min=0.0)
```
:::

---

:::step id="exercise-step7-code" title="Step 7: MLP.forward()"
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the forward pass through both layers.

    Args:
        x: Input batch, shape (n_samples, 2).

    Returns:
        Output probabilities, shape (n_samples, 1), each in (0, 1).
    """
    # TODO: Compute the two-layer forward pass and return the output probabilities
    raise NotImplementedError("TODO: implement MLP forward pass")
```
+++
**Hint:** Apply `relu()` to `self.hidden(x)`, then feed that hidden representation into `self.output(...)`, then apply `sigmoid()`.
+++
**Answer:**

```python
h = relu(self.hidden(x))
return sigmoid(self.output(h))
```
:::

---

:::terminal id="exercise-step7-output" title="Step 7: MLP Output" cmd="uv run python module_02_perceptrons/src/main.py" caption="100% &mdash; the ReLU hidden layer lets the MLP learn a non-linear boundary the single neuron could not."
<span class="header">============================================================
PART 1: Single Neuron on Linearly Separable Data
============================================================</span>
<span class="success">Accuracy: 149/150 (99.3%)</span>

<span class="header">============================================================
PART 2: Single Neuron on Non-Linearly Separable Data (XOR)
============================================================</span>
<span class="t-fail">Accuracy: 80/160 (50.0%)</span>

<span class="header">============================================================
PART 3: MLP on Non-Linearly Separable Data
============================================================</span>
  Epoch 100/500  loss=0.0217
  Epoch 200/500  loss=0.0107
  Epoch 300/500  loss=0.0071
  Epoch 400/500  loss=0.0053
  Epoch 500/500  loss=0.0042

<span class="success">Accuracy: 160/160 (100.0%)</span>
Saved comparison plot to output/step7_comparison.png
Saved MLP loss plot to output/step7_mlp_loss.png

<span class="header">============================================================
EXTRA CREDIT: MLP trained with your own SGD optimizer
============================================================</span>
<span class="success">Accuracy: 160/160 (100.0%)</span>
(Your optimizer matches torch.optim.SGD.)

<span class="header">============================================================
Done! Check the output/ directory for plots.
============================================================</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit: Your Own Optimizer

Implement `SGD.step()` &mdash; the optimizer step that applies gradients PyTorch already computed with autograd. <!-- .element: class="text-lg" -->

```python
def step(self) -> None:
    """Apply one gradient-descent update to every parameter, in place.

    Args:
        None.

    Returns:
        None. Each parameter tensor is modified in place.
    """
    # TODO: update each parameter in place using its gradient
    raise NotImplementedError("Extra credit: implement the optimizer step")
```

<div class="fragment hint-box">

**Hint:** Inside `with torch.no_grad():`, loop over `self.params` and update each one in place with `p -= self.lr * p.grad`.

</div>

The runner trains the same MLP once with `torch.optim.SGD` and once with this small optimizer class. Both use the same gradients; only the parameter update object changes. <!-- .element: class="text-lg" style="margin-top: 15px;" -->
