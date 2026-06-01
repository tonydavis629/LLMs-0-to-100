:::divider id="divider-softmax" title="Softmax" sub="Turning scores into probabilities"
:::

---

<!-- .slide: id="softmax-purpose" -->

## From Logits to a Distribution

Softmax is the function that has been quietly doing the normalization step in every attention layer. It takes a vector of raw scores (**logits**) and turns them into a probability distribution:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 820 230" width="100%" style="max-height: 220px;">
  <g font-size="12" text-anchor="middle" font-weight="600">
    <text x="145" y="22" fill="#f5a623">raw scores z</text>
    <text x="410" y="22" fill="#f5a623">positive scores e^z</text>
    <text x="675" y="22" fill="#4a9eff">probabilities</text>
  </g>
  <g font-size="12" text-anchor="middle">
    <text x="54" y="67" fill="#e8eaf0">cat</text>
    <text x="54" y="117" fill="#e8eaf0">sat</text>
    <text x="54" y="167" fill="#e8eaf0">mat</text>
    <rect x="90" y="48" width="150" height="24" rx="4" fill="rgba(245,166,35,0.62)"/><text x="258" y="65" fill="#e8eaf0">2.0</text>
    <rect x="90" y="98" width="90" height="24" rx="4" fill="rgba(245,166,35,0.42)"/><text x="198" y="115" fill="#e8eaf0">1.0</text>
    <rect x="90" y="148" width="50" height="24" rx="4" fill="rgba(245,166,35,0.25)"/><text x="158" y="165" fill="#e8eaf0">0.0</text>
    <text x="303" y="117" fill="#8892a4" font-size="24">&rarr;</text>
    <rect x="355" y="48" width="170" height="24" rx="4" fill="rgba(245,166,35,0.72)"/><text x="544" y="65" fill="#e8eaf0">7.39</text>
    <rect x="355" y="98" width="63" height="24" rx="4" fill="rgba(245,166,35,0.38)"/><text x="438" y="115" fill="#e8eaf0">2.72</text>
    <rect x="355" y="148" width="24" height="24" rx="4" fill="rgba(245,166,35,0.20)"/><text x="398" y="165" fill="#e8eaf0">1.00</text>
    <text x="568" y="117" fill="#8892a4" font-size="24">&rarr;</text>
    <rect x="620" y="48" width="120" height="24" rx="4" fill="rgba(74,158,255,0.72)"/><text x="759" y="65" fill="#e8eaf0">0.67</text>
    <rect x="620" y="98" width="44" height="24" rx="4" fill="rgba(74,158,255,0.38)"/><text x="682" y="115" fill="#e8eaf0">0.24</text>
    <rect x="620" y="148" width="16" height="24" rx="4" fill="rgba(74,158,255,0.22)"/><text x="655" y="165" fill="#e8eaf0">0.09</text>
  </g>
  <text x="410" y="210" fill="#8892a4" font-size="13" text-anchor="middle">Exponentiate to make scores positive, then divide by the total so the row sums to 1.</text>
</svg>
</div>

Inside attention, softmax is applied across the **keys** for each query. At the output, it is applied across the vocabulary to produce next-token probabilities.

---

<!-- .slide: id="softmax-interactive" -->

:::interactive id="softmax-explorer" widget="softmaxExplorer" title="Logits to Probabilities (and Temperature)"
:::

---

<!-- .slide: id="softmax-temperature" -->

## Temperature

Dividing the logits by a **temperature** $T$ before softmax controls how peaked the distribution is:

$$p_i = \text{softmax}(z / T)_i$$

:::columns cols="3" gap="20px"
**$T < 1$**

Sharper. The model commits to its top choice. More deterministic text.
+++
**$T = 1$**

The distribution exactly as the model produced it.
+++
**$T > 1$**

Flatter. More of the probability mass spreads to other tokens. More varied, riskier text.
:::

:::note
The same scaling idea appeared in attention: dividing $QK^T$ by $\sqrt{d_k}$ is a fixed temperature that keeps the softmax from saturating as the dimension grows.
:::
