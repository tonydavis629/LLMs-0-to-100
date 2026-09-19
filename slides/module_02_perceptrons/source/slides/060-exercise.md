:::divider id="divider-exercise" title="Exercise" sub="Text Classifier with Decision Boundary Visualization"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_02_perceptrons/exercise.py` &mdash; the only file you edit &mdash; and fill in the `NotImplementedError` lines. Run after each step. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_02_perceptrons/src/main.py

# Run a single step (1-7, or ec for the extra credit)
uv run python module_02_perceptrons/src/main.py --step 3
```

Trains a single neuron and an MLP on 2D data. Plots land in `output/` after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Perceptrons and Neural Networks

Built with PyTorch tensors:

- Train a single neuron on linearly separable data
- Watch it fail on XOR-like data
- Fix it with a ReLU hidden layer

Each function is mostly written. You fill in **one key line**, and every step has its own tests in `tests/`. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

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

    This is Shannon's entropy used as a loss: it measures how surprised we are
    by the prediction given the true label. It works elementwise, so it accepts
    a single sample or a whole batch.

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
    """Compute gradients of the BCE loss w.r.t. the neuron's weights and bias.

    For ONE sample, sigmoid + BCE collapse to dL/dz = y_pred - y_true, so
        dL/dw_j = (y_pred - y_true) * x_j     (error times that weight's input)
        dL/db   = (y_pred - y_true)

    Args:
        X: Input batch, shape (n_samples, 2).
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted probabilities, shape (n_samples,).

    Returns:
        (dw, db): gradient w.r.t. weights (shape (2,)) and bias (scalar tensor).
    """
    error = y_pred - y_true  # How far each prediction is from its label
    n = X.shape[0]  # Number of samples in the batch
    db = error.mean()  # Bias gradient: the average error (provided for you)
    dw = None
    # TODO: Compute dw, the weight gradient, from X and error averaged over the batch
    raise NotImplementedError("TODO: compute dw")
    return (dw, db)
```
+++
**Hint:** Transpose `X` so a matrix product with `error` sums over the samples, then divide by `n`.
+++
**Answer:**

```python
dw = X.T @ error / n
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

    Gradient descent moves each parameter a small step in the negative
    gradient direction (downhill on the loss surface).

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

:::terminal id="exercise-step5-output" title="Steps 1&ndash;4: Single Neuron Output" cmd="uv run python module_02_perceptrons/src/main.py" maxw="920px" caption="99.3% on linear data, but only 50% on XOR. Step 5 tests for that failure: a correct single neuron <em>must</em> get stuck at ln 2 here."
<span class="header">=== Step 1: forward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: binary_cross_entropy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: compute_gradients() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: update_parameters() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 5: train the single neuron (Steps 1-4 together) ===</span> <span class="success">CORRECT</span>
Linear data: 150 samples from linear_separable.csv
  Epoch  20/100  loss=0.0753
  <span class="skipped">...</span>
  Epoch 100/100  loss=0.0391
  Final weights: [2.4276, 1.7352], bias: 0.1047
  <span class="success">Accuracy: 149/150 (99.3%)</span>
XOR data: 160 samples from non_linear_separable.csv
  Epoch  20/100  loss=0.6926
  <span class="skipped">...</span>
  Epoch 100/100  loss=0.6926
  Final weights: [0.0070, -0.0399], bias: -0.0016
  <span class="t-fail">Accuracy: 80/160 (50.0%)</span>
  <span class="success">CORRECT</span>    linear data: accuracy is at least 95%
  <span class="success">CORRECT</span>    linear data: the loss falls to under a quarter of its starting value
  <span class="success">CORRECT</span>    XOR data: the loss is stuck at ln 2 = 0.693, the cost of a coin flip
  <span class="success">CORRECT</span>    XOR data: accuracy is no better than chance (a single neuron is a linear classifier)
<span class="header">=== Step 6: relu() ===</span> <span class="skipped">INCOMPLETE</span>
:::

---

<!-- .slide: id="exercise-step6-context" -->

## Step 5: Why the Single Neuron Fails

PART 2 ran the **same** trained neuron on the XOR-like dataset. No new code needed. <!-- .element: class="text-lg" -->

:::note
**50% accuracy is random guessing.** The loss flat-lines at **0.693** = $\ln 2$: the cross-entropy of a fair coin. <!-- .element: class="text-lg" style="margin:0;" -->
:::

No line separates XOR, so the neuron predicts 0.5 for everything. The fix: a **hidden layer with a nonlinearity**. <!-- .element: class="text-lg" style="margin-top: 18px;" -->

---

:::step id="exercise-step6-code" title="Step 6: relu()"
```python
def relu(z: torch.Tensor) -> torch.Tensor:
    """The ReLU (Rectified Linear Unit) activation: max(0, z).

    Keeps positive values unchanged and clamps negatives to zero. It is cheap
    and avoids the vanishing-gradient problem, so it dominates the HIDDEN
    layers of modern networks.

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with negatives replaced by 0.
    """
    # TODO: Return the elementwise maximum of 0 and z in one line
    raise NotImplementedError("TODO: implement the ReLU activation")
```
+++
**Hint:** Use `torch.clamp`.
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
    raise NotImplementedError("TODO: implement the MLP forward pass")
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

:::terminal id="exercise-step7-output" title="Step 7: MLP Output" cmd="uv run python module_02_perceptrons/src/main.py" maxw="920px" caption="100% &mdash; the ReLU hidden layer lets the MLP learn a non-linear boundary the single neuron could not. The forward-pass test recomputes the output from the model's own layers."
<span class="header t-green">=== Step 5: train the single neuron (Steps 1-4 together) ===</span> <span class="success">CORRECT</span>
  <span class="t-gray">Accuracy: 149/150 (99.3%)</span>
  <span class="t-gray">Accuracy: 80/160 (50.0%)</span>
<span class="header t-yellow">=== Step 6: relu() ===</span> <span class="success">CORRECT</span>
  <span class="success">CORRECT</span>    negatives become 0 and positives pass through: [-2,-0.5,0,0.5,2] -> [0,0,0,0.5,2]
  <span class="success">CORRECT</span>    works elementwise on a matrix and keeps its shape
  <span class="success">CORRECT</span>    lets gradients through where z > 0 and blocks them where z < 0

<span class="header t-cyan">=== Step 7: MLP.forward() ===</span> <span class="success">CORRECT</span>
  Epoch 100/500  loss=0.0217
  <span class="skipped">...</span>
  Epoch 500/500  loss=0.0042
  <span class="success">Accuracy: 160/160 (100.0%)</span>
  Saved comparison plot to output/step7_comparison.png
  Saved MLP loss plot to output/step7_mlp_loss.png
  <span class="success">CORRECT</span>    returns one probability per sample, shape (n_samples, 1)
  <span class="success">CORRECT</span>    every output is strictly between 0 and 1
  <span class="success">CORRECT</span>    equals sigmoid(output(relu(hidden(x)))) using the model's own layers
  <span class="success">CORRECT</span>    XOR data: the MLP reaches at least 95% accuracy where the single neuron got 50%

<span class="header">=== Extra Credit: SGD.step() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">Extra credit: implement the optimizer step</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit: Your Own Optimizer

Steps 3 and 4 again, for every parameter of the MLP at once, with autograd doing Step 3 for you. <!-- .element: class="text-lg" -->

<div style="display:flex; gap:24px; align-items:stretch; margin-top:4px;">
<div style="flex:1;">

**Step 3 (single neuron, by hand)**

```python
dw, db = compute_gradients(X, y, y_pred)
```

Returns the gradients. `dw` has the shape of `weights`.

</div>
<div style="flex:1;">

**Autograd (MLP, four parameter tensors)**

```python
loss.backward()
```

Computes every gradient, then stores each one on its own tensor as `p.grad`. Same shape as `p`. This is `dw`, for every `p`.

</div>
</div>

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (w * w).sum()      # L = w1^2 + w2^2
loss.backward()
w.grad                    # tensor([2., 4.]) == dL/dw == 2w
```

---

<!-- .slide: id="exercise-extra-credit-loop" -->

## Where step() Runs

The runner trains the same MLP with `torch.optim.SGD`, then again with your class. Same gradients; only the update object changes. <!-- .element: class="text-lg" -->

```python
for epoch in range(epochs):
    optimizer.zero_grad()      # clear last step's .grad on every parameter
    loss = binary_cross_entropy(y_col, model(X)).mean()
    loss.backward()            # autograd: fill every p.grad   (Step 3)
    optimizer.step()           # your code: p -= lr * p.grad   (Step 4)
```

- `step()` is `update_parameters` applied to every tensor in `self.params`, in place, so the model keeps pointing at the same tensors
- `.grad` accumulates: `backward()` adds to whatever is there, so `zero_grad()` (provided) wipes it first
- The update is not part of the loss, so it goes inside `with torch.no_grad():` to keep it off autograd's tape

---

:::step id="exercise-extra-code" title="Extra Credit: SGD.step()"
```python
def step(self) -> None:
    """Apply one gradient-descent update to every parameter, in place.

    Args:
        None.

    Returns:
        None. Each parameter tensor is modified in place.
    """
    # TODO: update each parameter in place using its gradient p.grad (Step 4, for every tensor)
    raise NotImplementedError("Extra credit: implement the optimizer step")
```
+++
**Hint:** Inside `with torch.no_grad():`, loop over `self.params` and update each one in place with `p -= self.lr * p.grad`.
+++
**Answer:**

```python
with torch.no_grad():
    for p in self.params:
        p -= self.lr * p.grad
```
:::

---

:::terminal id="exercise-extra-output" title="Extra Credit: Output" cmd="uv run python module_02_perceptrons/src/main.py --step ec" maxw="920px" caption="Identical loss curve to Step 7: your three lines do exactly what torch.optim.SGD does."
<span class="header t-blue">=== Extra Credit: SGD.step() ===</span> <span class="success">CORRECT</span>
  Epoch 100/500  loss=0.0217
  Epoch 200/500  loss=0.0107
  Epoch 300/500  loss=0.0071
  Epoch 400/500  loss=0.0053
  Epoch 500/500  loss=0.0042
  <span class="success">Accuracy: 160/160 (100.0%)</span>
  <span class="success">CORRECT</span>    one step: p=[1,2], grad=[0.5,-1], lr=0.1 gives [0.95, 2.1]
  <span class="success">CORRECT</span>    updates the parameter tensor in place (the model keeps pointing at it)
  <span class="success">CORRECT</span>    matches torch.optim.SGD after one step on an nn.Linear layer
  <span class="success">CORRECT</span>    the MLP trained with your optimizer reaches at least 95% on XOR
:::
