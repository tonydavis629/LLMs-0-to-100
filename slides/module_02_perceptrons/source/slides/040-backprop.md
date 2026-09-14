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

<!-- .slide: id="derivatives-review" -->

## Review: Derivatives

The derivative of $f$ at $x$ is the slope of the curve there.

:::columns grid="1.1fr 1fr" gap="30px" valign="center"
<div>
<svg viewBox="0 0 430 300" width="100%" style="max-height:330px; display:block; margin:0 auto;">
<defs><marker id="dxarrow" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"/></marker></defs>
<line x1="40" y1="270" x2="410" y2="270" stroke="#8892a4" stroke-width="1.5" marker-end="url(#dxarrow)"/>
<line x1="40" y1="270" x2="40" y2="12" stroke="#8892a4" stroke-width="1.5" marker-end="url(#dxarrow)"/>
<text x="416" y="286" fill="#8892a4" font-size="13">x</text>
<text x="20" y="16" fill="#8892a4" font-size="13">f(x)</text>
<polyline points="62.5,42.8 68.1,52.7 73.8,62.4 79.4,71.9 85.0,81.0 90.6,89.9 96.2,98.4 101.9,106.7 107.5,114.8 113.1,122.5 118.8,129.9 124.4,137.1 130.0,144.0 135.6,150.6 141.2,156.9 146.9,163.0 152.5,168.8 158.1,174.2 163.8,179.4 169.4,184.4 175.0,189.0 180.6,193.4 186.2,197.4 191.9,201.2 197.5,204.8 203.1,208.0 208.8,210.9 214.4,213.6 220.0,216.0 225.6,218.1 231.2,219.9 236.9,221.5 242.5,222.8 248.1,223.7 253.8,224.4 259.4,224.9 265.0,225.0 270.6,224.9 276.2,224.4 281.9,223.7 287.5,222.8 293.1,221.5 298.8,219.9 304.4,218.1 310.0,216.0 315.6,213.6 321.2,210.9 326.9,208.0 332.5,204.8 338.1,201.2 343.8,197.4 349.4,193.4 355.0,189.0 360.6,184.4 366.2,179.4 371.9,174.2 377.5,168.8 383.1,163.0 388.8,156.9 394.4,150.6 400.0,144.0" fill="none" stroke="#e8eaf0" stroke-width="2.5"/>
<line x1="58.0" y1="57.6" x2="202.0" y2="230.4" stroke="#f5a623" stroke-width="2.5"/>
<line x1="220.0" y1="225.0" x2="310.0" y2="225.0" stroke="#3fb950" stroke-width="2.5"/>
<line x1="319.0" y1="217.8" x2="391.0" y2="160.2" stroke="#4a9eff" stroke-width="2.5"/>
<line x1="130.0" y1="144.0" x2="197.5" y2="144.0" stroke="#8892a4" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="197.5" y1="144.0" x2="197.5" y2="204.8" stroke="#8892a4" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="130.0" y1="144.0" x2="197.5" y2="204.8" stroke="#8892a4" stroke-width="1.5"/>
<circle cx="130.0" cy="144.0" r="5" fill="#f5a623"/>
<circle cx="197.5" cy="204.8" r="4" fill="#8892a4"/>
<circle cx="265" cy="225.0" r="5" fill="#3fb950"/>
<circle cx="355" cy="189.0" r="5" fill="#4a9eff"/>
<text x="120.0" y="149.0" fill="#f5a623" font-size="13" text-anchor="end">(x, f(x))</text>
<text x="163.8" y="138.0" fill="#8892a4" font-size="12" text-anchor="middle">h</text>
<text x="203.5" y="178.4" fill="#8892a4" font-size="12">f(x+h) − f(x)</text>
<text x="162" y="204" fill="#f5a623" font-size="12" text-anchor="end">tangent, slope f′(x) &lt; 0</text>
<text x="265" y="247.0" fill="#3fb950" font-size="12" text-anchor="middle">f′(x) = 0</text>
<text x="365" y="211.0" fill="#4a9eff" font-size="12">f′(x) &gt; 0</text>
</svg>
</div>
+++
<div>

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

- $\dfrac{d}{dx}$: an instruction, differentiate with respect to $x$
- $f'(x)$: a number, the slope of the tangent line at $x$
- $f'(x) < 0$: $f$ is falling. $f'(x) = 0$: flat, a minimum here. $f'(x) > 0$: rising
- $\dfrac{\partial f}{\partial w}$: same idea with several inputs, vary only $w$
</div>
:::

---

<!-- .slide: id="backprop-without" -->

## Why We Need Backprop

The naive way: nudge one weight by $\epsilon$, recompute the loss. <!-- .element: class="text-lg" -->

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w)}{\epsilon}$$

- Costs **one forward pass per weight**
- Modern networks: **billions** of weights
- Backprop: *every* gradient in **one backward pass**

---

<!-- .slide: id="chain-rule" -->

## The Chain Rule in the Backward Pass

**The chain rule.** If $y$ depends on $u$ and $u$ depends on $x$, the rates of change multiply: <!-- .element: class="text-lg" -->

$$
y = f(u), \quad u = g(x)
\qquad \Longrightarrow \qquad
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

$$
y = f(g(x))
\qquad \Longrightarrow \qquad
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

A network is a chain of such dependencies: $w \to z \to \hat{y} \to L$. So multiply local derivatives along the path back to the weight. <!-- .element: class="text-lg" -->

<div style="text-align:center; margin-top:6px;">
<svg viewBox="-70 82 700 148" width="94%" style="max-height:150px;">
  <defs>
    <marker id="crf" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#4a9eff"/></marker>
    <marker id="crb" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f5a623"/></marker>
  </defs>
  <g font-size="15" text-anchor="middle">
    <rect x="16" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="60" y="119" fill="#e8eaf0">w</text>
    <rect x="160" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="204" y="119" fill="#e8eaf0">z</text>
    <rect x="310" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/><text x="354" y="119" fill="#e8eaf0">ŷ</text>
    <rect x="460" y="90" width="88" height="48" rx="7" fill="#0d1225" stroke="#e74c3c" stroke-width="2"/><text x="504" y="119" fill="#e8eaf0">L</text>
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
  </g>
  <text x="135" y="184" fill="#f5a623" font-size="14" text-anchor="middle">∂z/∂w</text>
  <text x="285" y="184" fill="#f5a623" font-size="14" text-anchor="middle">∂ŷ/∂z</text>
  <text x="435" y="184" fill="#f5a623" font-size="14" text-anchor="middle">∂L/∂ŷ</text>
  <text x="350" y="215" fill="#e8eaf0" font-size="15" text-anchor="middle" font-weight="600">
    ∂L/∂w = (∂L/∂ŷ) &times; (∂ŷ/∂z) &times; (∂z/∂w)
  </text>
</svg>
</div>

<!-- .element: class="text-lg" style="margin-top:4px;" -->

---

<!-- .slide: id="backprop-overview" -->

## The Backpropagation Algorithm

Run the network forward, then walk **backward** applying the chain rule from the previous slide.


$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}
$$

$$
W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial L}{\partial W}
$$

<div style="display:flex; justify-content:center; align-items:stretch; gap:14px; margin-top:22px; color: var(--text-color);">
<div style="flex:1; max-width:330px; border:1.5px solid #4a9eff; border-radius:8px; padding:10px 12px; text-align:center;"><div style="display:inline-block; width:36px; height:36px; line-height:36px; font-size:20px; border-radius:50%; background:#4a9eff; color:#0a0e1a; font-weight:700;">1</div><div style="font-weight:600; margin-top:6px;">Forward</div><div style="color: var(--muted-color); font-size:0.85em;">compute ŷ, then L</div></div>
<div style="align-self:center; color: var(--muted-color); font-size:1.4em;">&rarr;</div>
<div style="flex:1; max-width:330px; border:1.5px solid #f5a623; border-radius:8px; padding:10px 12px; text-align:center;"><div style="display:inline-block; width:36px; height:36px; line-height:36px; font-size:20px; border-radius:50%; background:#f5a623; color:#0a0e1a; font-weight:700;">2</div><div style="font-weight:600; margin-top:6px;">Backward</div><div style="color: var(--muted-color); font-size:0.85em;">multiply local derivatives</div></div>
<div style="align-self:center; color: var(--muted-color); font-size:1.4em;">&rarr;</div>
<div style="flex:1; max-width:330px; border:1.5px solid #e74c3c; border-radius:8px; padding:10px 12px; text-align:center;"><div style="display:inline-block; width:36px; height:36px; line-height:36px; font-size:20px; border-radius:50%; background:#e74c3c; color:#0a0e1a; font-weight:700;">3</div><div style="font-weight:600; margin-top:6px;">Gradient</div><div style="color: var(--muted-color); font-size:0.85em;">&part;L/&part;W for every weight</div></div>
<div style="align-self:center; color: var(--muted-color); font-size:1.4em;">&rarr;</div>
<div style="flex:1; max-width:330px; border:1.5px solid #3fb950; border-radius:8px; padding:10px 12px; text-align:center;"><div style="display:inline-block; width:36px; height:36px; line-height:36px; font-size:20px; border-radius:50%; background:#3fb950; color:#0a0e1a; font-weight:700;">4</div><div style="font-weight:600; margin-top:6px;">Update</div><div style="color: var(--muted-color); font-size:0.85em;">W &larr; W &minus; &eta; &part;L/&part;W</div></div>
</div>

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

- New weights = new coordinates
- New coordinates = new loss height
:::
