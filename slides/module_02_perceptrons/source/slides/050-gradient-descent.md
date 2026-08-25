:::divider id="divider-gradient-descent" title="Gradient Descent" sub="Walking downhill on the loss surface"
:::

---

<!-- .slide: id="gradient-descent-rule" -->
## The Update Rule

Compute the gradient of the loss, then step in the opposite direction:

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

**SGD** computes the gradient on a random mini-batch, not the whole dataset.

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

**Typical batch sizes:** 32, 64, 128, 256. Noise vs. stability is a tuning decision.

---

<!-- .slide: id="loss-landscape" -->
## Loss Landscape

The loss is a surface in weight space: $n$ weights, $(n+1)$-dimensional surface.

:::columns cols="3" gap="25px"
**Sharp Minima**

Small weight changes, large loss changes. **Generalize poorly**.
+++
**Flat Minima**

Robust to small changes. **Generalize better**.
+++
**Saddle Points**

Zero gradient, not a minimum. In high dimensions, **far more common** than local minima.
:::

SGD noise helps escape sharp minima and saddle points.

<p class="footnote">Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets"</p>

---

:::interactive id="anim-optimizer" widget="lossLandscape" title="Loss Landscape: Two Weights"
:::

---

<!-- .slide: id="overfitting" -->
## Overfitting and Generalization

Enough parameters can **memorize** any training set: zero training loss, useless on new data.

:::columns cols="2" gap="40px"
**Overfitting**

- Training loss goes to zero
- Test loss stays high
- Memorized, not learned
+++
**Generalization**

- Low loss on training AND test data
- Learned the underlying structure
:::

**Regularization** (covered in later modules):
- **Dropout**: randomly zero out neurons during training
- **Weight decay**: penalize large weights with $\lambda \|\mathbf{w}\|^2$

---

:::manim id="anim-overfitting" scene="overfit-viz"
:::

---

<!-- .slide: id="adam" -->
## Adam Optimizer

**Adam** (Adaptive Moment Estimation) keeps a per-parameter learning rate.

- Tracks each gradient's mean (first moment) and variance (second moment)
- Consistently large gradients: smaller learning rate
- Small or noisy gradients: larger learning rate

The **default optimizer** in practice. When in doubt, start with Adam.

---

:::interactive id="adam-landscape" widget="adamLandscape" title="Adam on the Loss Landscape"
:::

---

<!-- .slide: id="computation-graphs" -->

## Computation Graphs

The forward pass writes a tape. The backward pass replays it. <!-- .element: class="text-lg" -->

<div style="text-align:center; margin-top:4px;">
<svg viewBox="0 0 780 300" width="96%" style="max-height:330px;">
  <defs>
    <marker id="cgf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#4a9eff"/></marker>
    <marker id="cgb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#f5a623"/></marker>
  </defs>
  <g font-size="17" text-anchor="middle">
    <rect x="20"  y="40" width="80" height="46" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="60"  y="69" fill="#e8eaf0">x</text>
    <rect x="160" y="40" width="96" height="46" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="208" y="69" fill="#e8eaf0">z = wx</text>
    <rect x="316" y="40" width="96" height="46" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="364" y="69" fill="#e8eaf0">z + b</text>
    <rect x="472" y="40" width="96" height="46" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="520" y="69" fill="#e8eaf0">σ(z)</text>
    <rect x="628" y="40" width="80" height="46" rx="8" fill="#0d1225" stroke="#e74c3c" stroke-width="2"/><text x="668" y="69" fill="#e8eaf0">L</text>
  </g>
  <g stroke="#4a9eff" stroke-width="2.2" fill="none">
    <line x1="102" y1="63" x2="156" y2="63" marker-end="url(#cgf)"/>
    <line x1="258" y1="63" x2="312" y2="63" marker-end="url(#cgf)"/>
    <line x1="414" y1="63" x2="468" y2="63" marker-end="url(#cgf)"/>
    <line x1="570" y1="63" x2="624" y2="63" marker-end="url(#cgf)"/>
  </g>
  <g stroke="#8892a4" stroke-width="1.4" stroke-dasharray="4 4">
    <line x1="208" y1="88" x2="208" y2="132"/>
    <line x1="364" y1="88" x2="364" y2="132"/>
    <line x1="520" y1="88" x2="520" y2="132"/>
  </g>
  <text x="60" y="152" fill="#8892a4" font-size="15" text-anchor="middle">tape</text>
  <g font-size="15" text-anchor="middle">
    <rect x="150" y="132" width="116" height="56" rx="7" fill="#151a2e" stroke="#8892a4" stroke-width="1.4" stroke-dasharray="5 4"/><text x="208" y="153" fill="#8892a4">saved x</text><text x="208" y="175" fill="#e8eaf0">∂z/∂w = x</text>
    <rect x="306" y="132" width="116" height="56" rx="7" fill="#151a2e" stroke="#8892a4" stroke-width="1.4" stroke-dasharray="5 4"/><text x="364" y="153" fill="#8892a4">saved z</text><text x="364" y="175" fill="#e8eaf0">∂/∂b = 1</text>
    <rect x="462" y="132" width="116" height="56" rx="7" fill="#151a2e" stroke="#8892a4" stroke-width="1.4" stroke-dasharray="5 4"/><text x="520" y="153" fill="#8892a4">saved σ(z)</text><text x="520" y="175" fill="#e8eaf0">σ'(z)</text>
  </g>
  <g stroke="#f5a623" stroke-width="2.4" fill="none">
    <line x1="628" y1="230" x2="472" y2="230" marker-end="url(#cgb)"/>
    <line x1="472" y1="230" x2="316" y2="230" marker-end="url(#cgb)"/>
    <line x1="316" y1="230" x2="160" y2="230" marker-end="url(#cgb)"/>
  </g>
  <text x="394" y="266" fill="#f5a623" font-size="16" text-anchor="middle">multiply the saved local derivatives, right to left</text>
</svg>
</div>

The tape is written by running the code, so a new architecture never needs a new derivation. <!-- .element: class="text-lg" style="text-align:center; color: var(--muted-color); margin-top:6px;" -->
