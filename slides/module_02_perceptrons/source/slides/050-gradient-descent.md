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

<!-- .slide: id="local-minima" -->

## Local Minima and Saddle Points

:::columns grid="1fr 1.25fr" gap="30px" valign="center"
<div style="text-align:center;">
<p class="text-lg" style="color: var(--primary-color); font-weight:600; margin:0 0 4px 0;">One weight</p>
<svg viewBox="0 0 320 220" width="100%" style="max-height:250px; display:block; margin:0 auto;">
<line x1="24" y1="204" x2="305" y2="204" stroke="#8892a4" stroke-width="1.5"/>
<line x1="24" y1="204" x2="24" y2="20" stroke="#8892a4" stroke-width="1.5"/>
<text x="300" y="218" fill="#8892a4" font-size="12" text-anchor="end">w</text>
<text x="12" y="18" fill="#8892a4" font-size="12">L</text>
<polyline points="24.0,42.0 26.3,42.2 28.6,42.5 30.9,42.9 33.2,43.5 35.5,44.2 37.8,45.1 40.1,46.1 42.4,47.4 44.7,49.0 47.0,50.9 49.3,53.1 51.6,55.6 53.9,58.5 56.2,61.7 58.5,65.3 60.8,69.3 63.1,73.6 65.4,78.2 67.7,83.1 70.0,88.2 72.3,93.4 74.6,98.6 76.9,103.7 79.2,108.7 81.5,113.5 83.8,117.8 86.1,121.7 88.4,125.0 90.7,127.7 93.0,129.6 95.3,130.7 97.6,131.0 99.9,130.5 102.2,129.2 104.5,127.1 106.8,124.3 109.1,120.8 111.4,116.7 113.7,112.2 116.0,107.3 118.3,102.1 120.6,96.7 122.9,91.3 125.2,85.9 127.5,80.7 129.8,75.6 132.1,70.8 134.4,66.3 136.7,62.2 139.0,58.4 141.3,55.0 143.6,51.9 145.9,49.2 148.2,46.9 150.5,44.8 152.8,43.1 155.1,41.7 157.4,40.5 159.7,39.5 162.0,38.8 164.3,38.2 166.6,37.9 168.9,37.7 171.2,37.8 173.5,38.0 175.8,38.4 178.1,39.0 180.4,39.9 182.7,41.0 185.0,42.4 187.3,44.0 189.6,46.0 191.9,48.2 194.2,50.7 196.5,53.4 198.8,56.4 201.1,59.6 203.4,62.9 205.7,66.3 208.0,69.6 210.3,72.9 212.6,76.0 214.9,78.7 217.2,81.1 219.5,83.1 221.8,84.5 224.1,85.3 226.4,85.5 228.7,85.1 231.0,84.1 233.3,82.5 235.6,80.4 237.9,77.8 240.2,74.8 242.5,71.6 244.8,68.2 247.1,64.6 249.4,61.0 251.7,57.5 254.0,54.2 256.3,51.0 258.6,48.0 260.9,45.3 263.2,42.9 265.5,40.8 267.8,38.9 270.1,37.3 272.4,35.9 274.7,34.7 277.0,33.8 279.3,33.0 281.6,32.3 283.9,31.8 286.2,31.4 288.5,31.0 290.8,30.7 293.1,30.5 295.4,30.3 297.7,30.1 300.0,30.0" fill="none" stroke="#e8eaf0" stroke-width="2.5"/>
<circle cx="226.4" cy="77.5" r="8" fill="#f5a623"/>
<text x="226.4" y="109.5" fill="#f5a623" font-size="13" text-anchor="middle">local minimum</text>
<text x="226.4" y="124.5" fill="#8892a4" font-size="11" text-anchor="middle">gradient is zero, stuck</text>
<circle cx="97.6" cy="129.0" r="5" fill="#3fb950"/>
<text x="97.6" y="153.0" fill="#3fb950" font-size="13" text-anchor="middle">global minimum</text>
</svg>
</div>
+++
<div style="text-align:center;">
<p class="text-lg" style="color: var(--primary-color); font-weight:600; margin:0 0 4px 0;">Two weights</p>
<div class="interactive-host" data-widget="saddle3d" style="height:420px; display:flex;"></div>
</div>
:::

A hump that traps one weight is just a hill once a second weight can walk around it. A true minimum needs **every** direction to curve up: with $n$ weights, one chance in $2^n$. <!-- .element: class="text-lg" style="margin-top:14px;" -->

---

<!-- .slide: id="adam" -->
## Adam Optimizer

**Adaptive Moment Estimation.** Every weight gets its own step size. With gradient $g_t$ at step $t$: <!-- .element: class="text-lg" style="margin-bottom:0;" -->

<div class="adam-list">

- **Smooth the gradient** (momentum): $\textcolor{#4a9eff}{m_t} = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
  - Running average of the gradient. Mini-batch noise cancels, the consistent direction survives.
  - The walk keeps rolling across flat stretches and saddles.
- **Track the gradient's size**: $\textcolor{#f5a623}{v_t} = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
  - Running average of the squared gradient, one value per weight.
  - Tells us how big this weight's gradient usually is.
- **Undo the zero start**: $\hat m_t = m_t / (1 - \beta_1^t)$, $\hat v_t = v_t / (1 - \beta_2^t)$
  - Both averages begin at 0, so $m_1 = 0.1 g_1$ is a tenth of the real gradient.
  - The correction fades as $t$ grows.
- **Big gradient, small step. Small gradient, big step.**
  - Dividing by $\sqrt{\hat v_t}$ cancels the gradient's size, so every weight moves about $\eta$ per step.
  - $\epsilon$ keeps the division finite when $v_t$ is 0.
- **Defaults**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. When in doubt, start with Adam.
  - $\beta_1$ remembers about 10 steps, $\beta_2$ about 1000.

</div>

$$
\mathbf w_{t+1} = \mathbf w_t - \eta \frac{\textcolor{#4a9eff}{\hat m_t}}{\sqrt{\textcolor{#f5a623}{\hat v_t}} + \epsilon}
$$

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
