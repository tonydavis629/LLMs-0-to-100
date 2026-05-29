:::divider id="divider-gradient-descent" title="Gradient Descent" sub="Walking downhill on the loss surface"
:::

---

<!-- .slide: id="gradient-descent-rule" -->
## The Update Rule

Compute the gradient of the loss with respect to all weights, then take a step in the negative gradient direction:

$$
w_{\text{new}} = w_{\text{old}} + \left(-\eta \frac{\partial L}{\partial w}\right)
$$

:::columns cols="3" gap="25px"
<div style="text-align: center;">

**$\eta$ too large**

Overshoots the minimum and may diverge entirely

</div>
+++
<div style="text-align: center;">

**$\eta$ too small**

Converges very slowly, may get stuck in local minima

</div>
+++
<div style="text-align: center;">

**$\eta$ just right**

Steady convergence toward a good minimum

</div>
:::

---

<!-- .slide: id="sgd" -->
## Stochastic Gradient Descent

Instead of computing the gradient over the entire dataset (batch gradient descent), **SGD** computes it over a random subset (mini-batch).

:::columns cols="2" gap="30px"
**Small batch (e.g., 32)**

- Noisier gradient estimates
- Faster per step
- Noise helps escape local minima
+++
**Large batch (e.g., 256)**

- Smoother gradient estimates
- More memory required
- May converge to sharper minima
:::

**Typical batch sizes:** 32, 64, 128, 256. The trade-off between noise and stability is a practical tuning decision.

---

<!-- .slide: id="loss-landscape" -->
## Loss Landscape

The loss function defines a surface in weight space. For a network with $n$ weights, this is a surface in $(n + 1)$-dimensional space.

:::columns cols="3" gap="25px"
**Sharp Minima**

Small weight perturbations cause large loss changes. Tend to **generalize poorly**.
+++
**Flat Minima**

Robust to small perturbations. Tend to **generalize better** to new data.
+++
**Saddle Points**

Gradient is zero but not a minimum. In high dimensions, **far more common** than local minima.
:::

The learning rate affects which kind of minimum the optimizer finds. SGD noise helps escape sharp minima and saddle points.

<p class="footnote">Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets"</p>

---

:::interactive id="anim-optimizer" widget="lossLandscape" title="Loss Landscape: Two Weights"
:::

---

<!-- .slide: id="overfitting" -->
## Overfitting and Generalization

A network with enough parameters can **memorize** any training set &mdash; achieving zero training loss but failing on new data.

:::columns cols="2" gap="40px"
**Overfitting**

- Training loss goes to zero
- Test loss stays high or increases
- The model has memorized the training data, not learned the pattern
+++
**Generalization**

- Low loss on both training and test data
- The model has learned the underlying structure
- Performs well on data it has never seen
:::

**Regularization techniques** (named here, covered in later modules): **dropout** (randomly zero out neurons during training) and **weight decay** (penalize large weights with $\lambda \|\mathbf{w}\|^2$).

---

:::manim id="anim-overfitting" scene="overfit-viz"
:::

---

<!-- .slide: id="adam" -->
## Adam Optimizer

**Adam** (Adaptive Moment Estimation) maintains per-parameter learning rates based on the history of gradients.

- Tracks the first moment (mean) and second moment (variance) of each parameter's gradient
- Parameters with consistently large gradients get smaller learning rates
- Parameters with small or noisy gradients get larger learning rates

Adam is the **default optimizer** in practice for most deep learning tasks. When in doubt, start with Adam.

---

:::interactive id="adam-landscape" widget="adamLandscape" title="Adam on the Loss Landscape"
:::

---

<!-- .slide: id="computation-graphs" -->

## Computation Graphs

Frameworks like PyTorch build a **dynamic computation graph** as the forward pass runs. <!-- .element: class="text-lg" -->

- Each operation records its inputs and which function produced the output
- The backward pass walks the graph in reverse, applying the chain rule at every node
- You write only the forward pass; the gradients come for free
<!-- .element: class="text-lg" style="margin-top: 15px;" -->

:::note
This is **automatic differentiation** (autograd) &mdash; why you can prototype a new architecture in a few lines without deriving a single gradient by hand. <!-- .element: class="text-lg" style="margin: 0;" -->
:::
