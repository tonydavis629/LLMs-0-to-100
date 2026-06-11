:::divider id="divider-pipeline" title="The Pretraining Pipeline" sub="From a pile of text to a stream of training batches"
:::

---

<!-- .slide: id="pipeline-overview" -->

## The Pipeline End to End

<div class="pipeline-flow">
<svg viewBox="0 0 940 130" role="img" aria-label="Pretraining pipeline: collect, filter, tokenize, pack, split, train and evaluate"><defs><marker id="pf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f5a623"></path></marker></defs><rect x="8" y="44" width="128" height="50" rx="9" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.5)" stroke-width="1.4"></rect><text x="72" y="66" text-anchor="middle" font-size="15" fill="#e8eaf0">collect</text><text x="72" y="83" text-anchor="middle" font-size="11" fill="#8892a4">web, books, code</text><line x1="140" y1="69" x2="166" y2="69" stroke="#f5a623" stroke-width="2.5" marker-end="url(#pf)"></line><rect x="170" y="44" width="128" height="50" rx="9" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.5)" stroke-width="1.4"></rect><text x="234" y="66" text-anchor="middle" font-size="15" fill="#e8eaf0">filter</text><text x="234" y="83" text-anchor="middle" font-size="11" fill="#8892a4">clean, dedup</text><line x1="302" y1="69" x2="328" y2="69" stroke="#f5a623" stroke-width="2.5" marker-end="url(#pf)"></line><rect x="332" y="44" width="128" height="50" rx="9" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.5)" stroke-width="1.4"></rect><text x="396" y="66" text-anchor="middle" font-size="15" fill="#e8eaf0">tokenize</text><text x="396" y="83" text-anchor="middle" font-size="11" fill="#8892a4">text to IDs</text><line x1="464" y1="69" x2="490" y2="69" stroke="#f5a623" stroke-width="2.5" marker-end="url(#pf)"></line><rect x="494" y="44" width="128" height="50" rx="9" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.5)" stroke-width="1.4"></rect><text x="558" y="66" text-anchor="middle" font-size="15" fill="#e8eaf0">pack</text><text x="558" y="83" text-anchor="middle" font-size="11" fill="#8892a4">fixed blocks</text><line x1="626" y1="69" x2="652" y2="69" stroke="#f5a623" stroke-width="2.5" marker-end="url(#pf)"></line><rect x="656" y="44" width="128" height="50" rx="9" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.5)" stroke-width="1.4"></rect><text x="720" y="66" text-anchor="middle" font-size="15" fill="#e8eaf0">split</text><text x="720" y="83" text-anchor="middle" font-size="11" fill="#8892a4">train / val</text><line x1="788" y1="69" x2="814" y2="69" stroke="#f5a623" stroke-width="2.5" marker-end="url(#pf)"></line><rect x="818" y="44" width="116" height="50" rx="9" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.6"></rect><text x="876" y="66" text-anchor="middle" font-size="15" fill="#e8eaf0">train</text><text x="876" y="83" text-anchor="middle" font-size="11" fill="#8892a4">and evaluate</text></svg>
</div>

Most of the work in a real pretraining run happens **before** the optimizer ever runs. Data decisions shape the model as much as the architecture does.

---

<!-- .slide: id="collect-filter" -->

## Collect, Then Filter

:::columns cols="2" gap="34px"
**Collect**

Web pages, books, code, papers, forums, and documentation &mdash; the exact mixture depends on the model's goals and on legal and ethical constraints.
+++
**Filter**

Raw text is mostly junk. Filtering removes:

- Boilerplate and low-quality pages
- Near-duplicate documents (deduplication)
- Unsafe content categories and private information
- Obvious benchmark leakage, where it can be found
:::

Quality and deduplication matter more than raw volume &mdash; a point we return to in the data-quality section.

---

<!-- .slide: id="tokenize-pack" -->

## Tokenize and Pack

The tokenizer from Module 4 converts strings into token IDs. Then those IDs are **packed** into fixed-length blocks so every batch is a regular tensor shape.

- Documents have all different lengths, but the GPU wants uniform `(batch, block)` tensors
- Concatenate documents into one long stream, then chop it into equal blocks
- Insert an **end-of-text** token at each document boundary so the model can see where one document ends and the next begins

The animation on the next slide shows this packing step in motion.

---

:::manim id="packing-anim" scene="sequence-packing"
:::

---

<!-- .slide: id="split-train-eval" -->

## Split, Train, Evaluate

:::columns cols="2" gap="34px"
**Hold out a validation set**

Split the token stream into train and validation **before** reporting any metrics, so validation loss measures generalization rather than memorized batches.
+++
**Train in batches**

Each step is the same four moves:

1. Forward pass on a batch
2. Shifted-target cross-entropy loss
3. Backpropagation
4. Optimizer update
:::

Periodically measure validation loss on held-out blocks to check whether the model is learning patterns that generalize beyond the exact batches it just saw.
