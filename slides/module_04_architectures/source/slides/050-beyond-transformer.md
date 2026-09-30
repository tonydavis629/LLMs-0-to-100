:::divider id="divider-beyond" title="Beyond the Vanilla Transformer" sub="Modern blocks, MoE, and sub-quadratic alternatives"
:::

---

<!-- .slide: id="modern-block" -->

## The Modern Block: 2017 vs Llama-Style

:::columns cols="2" gap="30px"
**Original Transformer (2017)**

- LayerNorm
- ReLU activation in FFN
- Sinusoidal positional embeddings
- Post-norm (normalize after sub-layer)
- Bias terms in all linear layers
+++
**Modern Llama-Style**

- RMSNorm (simpler, faster)
- SwiGLU activation (smoother, better gradients)
- RoPE (rotary positional embeddings, callback to Module 3)
- Pre-norm (normalize before sub-layer)
- **Dropped bias terms** in linear layers (saves parameters)
:::

Mostly independent changes; together they make training more stable and efficient.

---

<!-- .slide: id="moe" -->

## Mixture of Experts

<div class="moe-svg">
<svg viewBox="-20 0 1020 400" role="img" aria-label="One transformer block with mixture of experts: dense masked self-attention used by every token, then an MoE layer in place of the feed-forward network, where a router sends each token to 2 of 8 experts and mixes their outputs.">
<defs><marker id="mea" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#f5a623"></path></marker><marker id="meb" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#8892a4"></path></marker></defs>
<rect x="28.0" y="70" width="344" height="282" rx="10" fill="rgba(74,158,255,0.03)" stroke="rgba(74,158,255,0.5)" stroke-width="1.2" stroke-dasharray="5 4"></rect>
<text x="24" y="215" text-anchor="start" font-size="17" fill="#f5a623">N&#215;</text>
<text x="200" y="392" text-anchor="middle" font-size="13" fill="#8892a4">token vectors from the previous block</text>
<line x1="200" y1="375" x2="200" y2="338" stroke="#f5a623" stroke-width="2" marker-end="url(#mea)"></line>
<rect x="50.0" y="290" width="300" height="46" rx="8" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="200.0" y="310.0" text-anchor="middle" font-size="14" fill="#e8eaf0">Masked Multi-Head Self-Attention</text>
<text x="200.0" y="328.0" text-anchor="middle" font-size="12" fill="#8892a4">dense: every token uses all of it</text>
<line x1="200" y1="290" x2="200" y2="262" stroke="#f5a623" stroke-width="2" marker-end="url(#mea)"></line>
<rect x="50.0" y="226" width="300" height="34" rx="8" fill="rgba(136,146,164,0.08)" stroke="#3a4460" stroke-width="1.6"></rect>
<text x="200.0" y="248.0" text-anchor="middle" font-size="14" fill="#e8eaf0">Add &amp; Norm</text>
<line x1="200" y1="226" x2="200" y2="200" stroke="#f5a623" stroke-width="2" marker-end="url(#mea)"></line>
<rect x="50.0" y="148" width="300" height="50" rx="8" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.6"></rect>
<text x="200.0" y="170.0" text-anchor="middle" font-size="15" fill="#e8eaf0">MoE layer</text>
<text x="200.0" y="188.0" text-anchor="middle" font-size="12" fill="#8892a4">replaces the feed-forward network</text>
<line x1="200" y1="148" x2="200" y2="122" stroke="#f5a623" stroke-width="2" marker-end="url(#mea)"></line>
<rect x="50.0" y="86" width="300" height="34" rx="8" fill="rgba(136,146,164,0.08)" stroke="#3a4460" stroke-width="1.6"></rect>
<text x="200.0" y="108.0" text-anchor="middle" font-size="14" fill="#e8eaf0">Add &amp; Norm</text>
<line x1="200" y1="86" x2="200" y2="40" stroke="#f5a623" stroke-width="2" marker-end="url(#mea)"></line>
<text x="200" y="30" text-anchor="middle" font-size="13" fill="#8892a4">to the next block</text>
<rect x="440" y="18" width="550" height="364" rx="10" fill="rgba(245,166,35,0.03)" stroke="rgba(245,166,35,0.5)" stroke-width="1.2" stroke-dasharray="5 4"></rect>
<line x1="350.0" y1="148" x2="440" y2="18" stroke="rgba(245,166,35,0.45)" stroke-width="1.2" stroke-dasharray="4 4"></line>
<line x1="350.0" y1="198" x2="440" y2="382" stroke="rgba(245,166,35,0.45)" stroke-width="1.2" stroke-dasharray="4 4"></line>
<text x="715.0" y="40" text-anchor="middle" font-size="14" fill="#f5a623">inside the MoE layer, for one token</text>
<rect x="460" y="168" width="110" height="64" rx="8" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="515.0" y="197.0" text-anchor="middle" font-size="15" fill="#e8eaf0">router</text>
<text x="515.0" y="215.0" text-anchor="middle" font-size="12" fill="#8892a4">top 2 of 8</text>
<line x1="570" y1="200" x2="638" y2="61" stroke="#2a3450" stroke-width="1.3" stroke-dasharray="4 4"></line>
<rect x="640" y="48" width="150" height="26" rx="6" fill="rgba(136,146,164,0.05)" stroke="#3a4460" stroke-width="1.2"></rect>
<text x="715.0" y="66" text-anchor="middle" font-size="13" fill="#5d6679">expert 1</text>
<line x1="570" y1="200" x2="638" y2="101" stroke="#2a3450" stroke-width="1.3" stroke-dasharray="4 4"></line>
<rect x="640" y="88" width="150" height="26" rx="6" fill="rgba(136,146,164,0.05)" stroke="#3a4460" stroke-width="1.2"></rect>
<text x="715.0" y="106" text-anchor="middle" font-size="13" fill="#5d6679">expert 2</text>
<line x1="570" y1="200" x2="636" y2="141" stroke="#f5a623" stroke-width="2.2" marker-end="url(#mea)"></line>
<rect x="640" y="128" width="150" height="26" rx="6" fill="rgba(245,166,35,0.16)" stroke="#f5a623" stroke-width="1.6"></rect>
<text x="715.0" y="146" text-anchor="middle" font-size="13" fill="#e8eaf0">expert 3 (FFN)</text>
<line x1="792" y1="141" x2="862" y2="190" stroke="#f5a623" stroke-width="2.2" marker-end="url(#mea)"></line>
<text x="835" y="135" text-anchor="middle" font-size="13" fill="#f5a623">&#215; 0.62</text>
<line x1="570" y1="200" x2="638" y2="181" stroke="#2a3450" stroke-width="1.3" stroke-dasharray="4 4"></line>
<rect x="640" y="168" width="150" height="26" rx="6" fill="rgba(136,146,164,0.05)" stroke="#3a4460" stroke-width="1.2"></rect>
<text x="715.0" y="186" text-anchor="middle" font-size="13" fill="#5d6679">expert 4</text>
<line x1="570" y1="200" x2="638" y2="221" stroke="#2a3450" stroke-width="1.3" stroke-dasharray="4 4"></line>
<rect x="640" y="208" width="150" height="26" rx="6" fill="rgba(136,146,164,0.05)" stroke="#3a4460" stroke-width="1.2"></rect>
<text x="715.0" y="226" text-anchor="middle" font-size="13" fill="#5d6679">expert 5</text>
<line x1="570" y1="200" x2="636" y2="261" stroke="#f5a623" stroke-width="2.2" marker-end="url(#mea)"></line>
<rect x="640" y="248" width="150" height="26" rx="6" fill="rgba(245,166,35,0.16)" stroke="#f5a623" stroke-width="1.6"></rect>
<text x="715.0" y="266" text-anchor="middle" font-size="13" fill="#e8eaf0">expert 6 (FFN)</text>
<line x1="792" y1="261" x2="862" y2="210" stroke="#f5a623" stroke-width="2.2" marker-end="url(#mea)"></line>
<text x="835" y="281" text-anchor="middle" font-size="13" fill="#f5a623">&#215; 0.38</text>
<line x1="570" y1="200" x2="638" y2="301" stroke="#2a3450" stroke-width="1.3" stroke-dasharray="4 4"></line>
<rect x="640" y="288" width="150" height="26" rx="6" fill="rgba(136,146,164,0.05)" stroke="#3a4460" stroke-width="1.2"></rect>
<text x="715.0" y="306" text-anchor="middle" font-size="13" fill="#5d6679">expert 7</text>
<line x1="570" y1="200" x2="638" y2="341" stroke="#2a3450" stroke-width="1.3" stroke-dasharray="4 4"></line>
<rect x="640" y="328" width="150" height="26" rx="6" fill="rgba(136,146,164,0.05)" stroke="#3a4460" stroke-width="1.2"></rect>
<text x="715.0" y="346" text-anchor="middle" font-size="13" fill="#5d6679">expert 8</text>
<circle cx="885" cy="200" r="20" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.8"></circle>
<text x="885" y="207" text-anchor="middle" font-size="20" fill="#f5a623">+</text>
<line x1="905" y1="200" x2="950" y2="200" stroke="#8892a4" stroke-width="2" marker-end="url(#meb)"></line>
<text x="968" y="205" text-anchor="middle" font-size="13" fill="#e8eaf0">out</text>
<text x="715" y="372" text-anchor="middle" font-size="12" fill="#8892a4">6 of 8 experts sit idle for this token</text>
</svg>
</div>

- Output $= \sum_i g_i \cdot \text{FFN}_i(\mathbf x)$, where the router weights $g_i$ are a softmax over the top-$k$ expert scores (here 2 of 8) and 0 for the rest
- Mixtral 8x7B stores about 47B parameters but uses about 13B per token (Jiang et al., 2024)
- Serving cost: every expert must stay reachable in memory, though each token runs only two

---

<!-- .slide: id="sub-quadratic" -->

## Sub-Quadratic Alternatives

Attention is $O(n^2)$ in sequence length; the bottleneck for long contexts. Alternatives under active research:

- **State-space models** (Mamba): recurrent update with input-dependent gates; linear in length, fixed-size state
- **Linear attention:** kernelize the softmax; no $n \times n$ matrix
- **RWKV:** recurrent update with linear attention-like weights
- **Hybrids** (Jamba): attention layers for short-range precision, Mamba layers for long-range compression

The transformer still dominates; the $O(n^2)$ cost keeps the search alive.
