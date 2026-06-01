:::divider id="divider-multi-head" title="Multi-Head Attention" sub="Several lookup patterns running in parallel"
:::

---

<!-- .slide: id="why-multi-head" -->

## Why Multiple Heads?

A single attention head produces one weighted average per token. Multiple heads split the feature dimension, so several lookup patterns can run in parallel.
<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 0 880 230" width="100%" style="max-height: 215px;">
  <text x="440" y="20" fill="#8892a4" font-size="12" text-anchor="middle">d_model = 512, split into H = 8 heads with d_k = 64 features each</text>
  <g font-size="10" text-anchor="middle" font-weight="600">
    <rect x="82" y="44" width="576" height="30" fill="none" stroke="#e8eaf0" stroke-width="1.4"/>
    <rect x="82" y="44" width="72" height="30" fill="rgba(74,158,255,0.30)" stroke="#0a0e1a"/><text x="118" y="64" fill="#e8eaf0">64</text>
    <rect x="154" y="44" width="72" height="30" fill="rgba(245,166,35,0.30)" stroke="#0a0e1a"/><text x="190" y="64" fill="#e8eaf0">64</text>
    <rect x="226" y="44" width="72" height="30" fill="rgba(80,200,120,0.30)" stroke="#0a0e1a"/><text x="262" y="64" fill="#e8eaf0">64</text>
    <rect x="298" y="44" width="72" height="30" fill="rgba(199,146,234,0.30)" stroke="#0a0e1a"/><text x="334" y="64" fill="#e8eaf0">64</text>
    <rect x="370" y="44" width="72" height="30" fill="rgba(74,158,255,0.30)" stroke="#0a0e1a"/><text x="406" y="64" fill="#e8eaf0">64</text>
    <rect x="442" y="44" width="72" height="30" fill="rgba(245,166,35,0.30)" stroke="#0a0e1a"/><text x="478" y="64" fill="#e8eaf0">64</text>
    <rect x="514" y="44" width="72" height="30" fill="rgba(80,200,120,0.30)" stroke="#0a0e1a"/><text x="550" y="64" fill="#e8eaf0">64</text>
    <rect x="586" y="44" width="72" height="30" fill="rgba(199,146,234,0.30)" stroke="#0a0e1a"/><text x="622" y="64" fill="#e8eaf0">64</text>
  </g>
  <g stroke="#e8eaf0" stroke-width="1.2">
    <line x1="154" y1="38" x2="154" y2="82"/><line x1="226" y1="38" x2="226" y2="82"/>
    <line x1="298" y1="38" x2="298" y2="82"/><line x1="370" y1="38" x2="370" y2="82"/>
    <line x1="442" y1="38" x2="442" y2="82"/><line x1="514" y1="38" x2="514" y2="82"/>
    <line x1="586" y1="38" x2="586" y2="82"/>
  </g>
  <text x="710" y="64" fill="#e8eaf0" font-size="12">Q, K, and V are grouped by feature slice</text>
  <g stroke="#8892a4" stroke-width="1.1" marker-end="url(#arrmhfirst)">
    <line x1="118" y1="74" x2="118" y2="108"/><line x1="190" y1="74" x2="190" y2="108"/>
    <line x1="262" y1="74" x2="262" y2="108"/><line x1="334" y1="74" x2="334" y2="108"/>
    <line x1="406" y1="74" x2="406" y2="108"/><line x1="478" y1="74" x2="478" y2="108"/>
    <line x1="550" y1="74" x2="550" y2="108"/><line x1="622" y1="74" x2="622" y2="108"/>
  </g>
  <g font-size="11" text-anchor="middle" font-weight="600">
    <rect x="86" y="110" width="64" height="40" rx="5" fill="rgba(74,158,255,0.12)" stroke="#4a9eff"/><text x="118" y="134" fill="#4a9eff">head 1</text>
    <rect x="158" y="110" width="64" height="40" rx="5" fill="rgba(245,166,35,0.12)" stroke="#f5a623"/><text x="190" y="134" fill="#f5a623">head 2</text>
    <rect x="230" y="110" width="64" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878"/><text x="262" y="134" fill="#50c878">head 3</text>
    <rect x="302" y="110" width="64" height="40" rx="5" fill="rgba(199,146,234,0.12)" stroke="#c792ea"/><text x="334" y="134" fill="#c792ea">head 4</text>
    <rect x="374" y="110" width="64" height="40" rx="5" fill="rgba(74,158,255,0.12)" stroke="#4a9eff"/><text x="406" y="134" fill="#4a9eff">head 5</text>
    <rect x="446" y="110" width="64" height="40" rx="5" fill="rgba(245,166,35,0.12)" stroke="#f5a623"/><text x="478" y="134" fill="#f5a623">head 6</text>
    <rect x="518" y="110" width="64" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878"/><text x="550" y="134" fill="#50c878">head 7</text>
    <rect x="590" y="110" width="64" height="40" rx="5" fill="rgba(199,146,234,0.12)" stroke="#c792ea"/><text x="622" y="134" fill="#c792ea">head 8</text>
  </g>
  <text x="440" y="187" fill="#8892a4" font-size="13" text-anchor="middle">The outputs are concatenated back to 512 dimensions, then a final projection mixes information across heads.</text>
  <defs><marker id="arrmhfirst" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker></defs>
</svg>
</div>

:::columns cols="2" gap="30px"
**Different comparisons**

One head may track subject-verb links, while another tracks coreference or nearby tokens.
+++
**Same compute budget**

The total width stays 512 here; it is divided across heads instead of copied for each head.
:::

---

<!-- .slide: id="multi-head-mechanics" -->

## Multi-Head Mechanics

Each head $h$ has its own projections and computes attention independently:

$$\text{head}_h = \text{Attention}(XW_Q^h, XW_K^h, XW_V^h)$$

The heads are concatenated and projected back through $W_O$:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

:::note
**Compute budget:** the total projection size across all heads is kept equal to $d_{\text{model}}$. With 8 heads and $d_{\text{model}} = 512$, each head uses $d_k = 64$. Multi-head attention is not more expensive than single-head &mdash; it is the same compute, split into independent patterns.
:::
