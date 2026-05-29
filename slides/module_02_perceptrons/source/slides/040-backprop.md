:::divider id="divider-backprop" title="Backpropagation" sub="The chain rule, applied systematically"
:::

---

<!-- .slide: id="loss-functions" -->

## Loss Functions

A **loss function** measures how wrong a prediction is. Training minimizes it. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="40px"
**Mean Squared Error (Regression)** <!-- .element: class="text-lg" style="color: var(--secondary-color); font-weight: 600; margin-bottom: 10px;" -->

$$L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

Penalizes large errors quadratically. Standard for predicting continuous values.
+++
**Binary Cross-Entropy (Classification)** <!-- .element: class="text-lg" style="color: var(--secondary-color); font-weight: 600; margin-bottom: 10px;" -->

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

Measures surprise under the model's predictions. High loss when confident and wrong.
:::

:::note
**Module 1 connection:** binary cross-entropy is Shannon's entropy used as a training objective &mdash; the same formula that scored n-gram models now drives learning. <!-- .element: class="text-lg" style="margin: 0;" -->
:::

---

<!-- .slide: id="backprop-without" -->

## Why We Need Backprop

The naive way to get a gradient: nudge one weight by a tiny $\epsilon$, recompute the loss, and measure the change. <!-- .element: class="text-lg" -->

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w)}{\epsilon}$$

That is **one forward pass per weight**. Modern networks have **billions** of weights, so this is hopeless. <!-- .element: class="text-lg" -->

Backpropagation gets *every* gradient in a **single backward pass**, about the cost of one forward pass. That is why training large networks is possible at all. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="backprop-overview" -->

## The Backpropagation Algorithm

Run the network forward, then walk **backward** applying the **chain rule** to turn the loss into a weight update.

<div style="text-align:center; margin-top:6px;">
<svg viewBox="0 0 760 230" width="96%" style="max-height:300px;">
  <defs>
    <marker id="bpf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#4a9eff"/></marker>
    <marker id="bpb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#f5a623"/></marker>
  </defs>
  <g font-size="16" text-anchor="middle">
    <rect x="22"  y="86" width="92" height="52" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2.2"/><text x="68"  y="118" fill="#e8eaf0">input x</text>
    <rect x="186" y="86" width="110" height="52" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2.2"/><text x="241" y="112" fill="#e8eaf0">weight W</text><text x="241" y="130" fill="#8892a4" font-size="13">z = Wx + b</text>
    <rect x="368" y="86" width="100" height="52" rx="8" fill="#0d1225" stroke="#4a9eff" stroke-width="2.2"/><text x="418" y="112" fill="#e8eaf0">output ŷ</text><text x="418" y="130" fill="#8892a4" font-size="13">ŷ = σ(z)</text>
    <rect x="540" y="86" width="92" height="52" rx="8" fill="#0d1225" stroke="#e74c3c" stroke-width="2.2"/><text x="586" y="118" fill="#e8eaf0">loss L</text>
    <rect x="664" y="86" width="84" height="52" rx="8" fill="#0d1225" stroke="#3fb950" stroke-width="2.2"/><text x="706" y="112" fill="#e8eaf0">update</text><text x="706" y="130" fill="#8892a4" font-size="12">new W</text>
  </g>
  <g stroke="#4a9eff" stroke-width="2.4" fill="none">
    <line x1="116" y1="100" x2="184" y2="100" marker-end="url(#bpf)"/>
    <line x1="298" y1="100" x2="366" y2="100" marker-end="url(#bpf)"/>
    <line x1="470" y1="100" x2="538" y2="100" marker-end="url(#bpf)"/>
  </g>
  <text x="385" y="64" fill="#4a9eff" font-size="14" text-anchor="middle">forward pass</text>
  <g stroke="#f5a623" stroke-width="2.4" fill="none">
    <line x1="540" y1="158" x2="470" y2="158" marker-end="url(#bpb)"/>
    <line x1="368" y1="158" x2="298" y2="158" marker-end="url(#bpb)"/>
    <path d="M586,140 C594,190 696,190 706,140" marker-end="url(#bpb)"/>
  </g>
  <text x="385" y="208" fill="#f5a623" font-size="14" text-anchor="middle">backward pass</text>
</svg>
</div>

The chain rule multiplies one local derivative per stage along the path **loss &rarr; output &rarr; weight**, giving $\partial L/\partial w$. The update rule then adds the negative-gradient step. <!-- .element: class="text-lg" style="text-align:center; color: var(--muted-color); margin-top:10px;" -->

---

<!-- .slide: id="chain-rule" -->

## The Chain Rule in the Backward Pass

To find how a weight deep in the network affects the loss, the chain rule multiplies local derivatives along the path from output back to that weight.

<div style="text-align:center; margin-top:6px;">
<svg viewBox="0 0 700 240" width="94%" style="max-height:300px;">
  <defs>
    <marker id="crf" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#4a9eff"/></marker>
    <marker id="crb" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f5a623"/></marker>
  </defs>
  <g font-size="15" text-anchor="middle">
    <rect x="16" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="60" y="119" fill="#e8eaf0">w</text>
    <rect x="160" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="204" y="119" fill="#e8eaf0">z</text>
    <rect x="310" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="354" y="119" fill="#e8eaf0">ŷ</text>
    <rect x="460" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#e74c3c" stroke-width="2"/><text x="504" y="119" fill="#e8eaf0">L</text>
    <rect x="590" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#3fb950" stroke-width="2"/><text x="634" y="112" fill="#e8eaf0">update</text><text x="634" y="130" fill="#8892a4" font-size="12">w new</text>
  </g>
  <g stroke="#4a9eff" stroke-width="2" fill="none">
    <line x1="106" y1="100" x2="158" y2="100" marker-end="url(#crf)"/>
    <line x1="250" y1="100" x2="308" y2="100" marker-end="url(#crf)"/>
    <line x1="400" y1="100" x2="458" y2="100" marker-end="url(#crf)"/>
  </g>
  <g stroke="#f5a623" stroke-width="2.4" fill="none">
    <line x1="460" y1="152" x2="400" y2="152" marker-end="url(#crb)"/>
    <line x1="310" y1="152" x2="250" y2="152" marker-end="url(#crb)"/>
    <line x1="160" y1="152" x2="106" y2="152" marker-end="url(#crb)"/>
    <path d="M504,140 C512,192 626,192 634,140" marker-end="url(#crb)"/>
  </g>
  <text x="135" y="184" fill="#f5a623" font-size="14" text-anchor="middle">∂z/∂w</text>
  <text x="285" y="184" fill="#f5a623" font-size="14" text-anchor="middle">∂ŷ/∂z</text>
  <text x="435" y="184" fill="#f5a623" font-size="14" text-anchor="middle">∂L/∂ŷ</text>
  <text x="350" y="215" fill="#e8eaf0" font-size="15" text-anchor="middle" font-weight="600">
    ∂L/∂w = (∂L/∂ŷ) &times; (∂ŷ/∂z) &times; (∂z/∂w)
  </text>
</svg>
</div>

Each layer only needs its own **local** derivative; the chain rule stitches them together. This is why autograd works: no matter how deep the network, each node just multiplies its gradient by the derivative of the operation it performed.

---

<!-- .slide: id="gradient-step-calculation" -->

## Gradient Descent as a Calculation

At a point on the loss surface, the gradient points uphill. The update adds a step in the opposite direction.

:::columns cols="2" gap="36px"
**Current point**

$$
\mathbf w_{\text{old}} =
\begin{bmatrix}
w_1 \\
w_2
\end{bmatrix}
$$

$$
\nabla L(\mathbf w_{\text{old}}) =
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2}
\end{bmatrix}
$$
+++
**Update**

$$
\mathbf w_{\text{new}} =
\mathbf w_{\text{old}} + \left(-\eta \nabla L(\mathbf w_{\text{old}})\right)
$$

The coordinates change because the weights change; the height changes because the loss at the new coordinates is different.
:::
