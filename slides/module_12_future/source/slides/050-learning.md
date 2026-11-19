:::divider id="divider-learning" title="Models That Keep Learning" sub="Assumption: learning stops at deployment"
:::

---

<!-- .slide: id="learning-assumption" -->

## The Weights Are Frozen

- Everything a deployed model appears to learn lives in the **context window**
- The session ends and it is lost
- Debug a problem Monday, the model retains nothing Tuesday

:::columns cols="2" gap="34px"
**Workaround: retrieval**

- Put the past in the prompt
- Costs tokens on every call
- The model still has not learned anything
+++
**Workaround: finetuning**

- Bake the past into the weights
- Slow, offline cycle
- Nothing like learning from experience as you go
:::

Neither gives the model accumulated experience. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="learning-forgetting" -->

## Catastrophic Forgetting

**Catastrophic forgetting** (McCloskey and Cohen, 1989):

- Train on new data and old capabilities degrade
- Gradient descent on the new task walks the weights away from the old one
- Nothing in the objective says not to

<div class="card-grid cols-3">
<div class="card"><h4>Replay</h4><p>Mix old data in with the new. Effective, and it means you never actually stop paying for the old data.</p></div>
<div class="card"><h4>Constrain</h4><p>Penalize movement in weights that mattered before (elastic weight consolidation). Needs to know which ones mattered.</p></div>
<div class="card"><h4>Isolate</h4><p>Put new learning in added parameters. This is LoRA, reused as a <strong>memory mechanism</strong> rather than an efficiency trick.</p></div>
</div>

None of the three is a solved recipe at this scale. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="learning-ttt" -->

## Test-Time Training Fine-Tunes on Each Input

<div class="svg-figure">
<svg viewBox="0 0 1000 240" role="img" aria-label="Two pipelines. Standard inference: the input goes through frozen weights to the answer. Test-time training: from the input, build a small training task; take a few gradient steps on a copy of the weights; answer with the updated copy; then throw the update away.">
  <text x="20" y="66" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">standard</text>
  <rect x="170" y="32" width="110" height="56" rx="6" fill="none" stroke="#e8eaf0" stroke-width="2" /><text x="225" y="66" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">input</text><path d="M 280 60 L 322 60" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 330 60 l -10 -5 l 0 10 z" fill="#8892a4" /><rect x="330" y="32" width="200" height="56" rx="6" fill="none" stroke="#4a9eff" stroke-width="2" /><text x="430" y="66" fill="#4a9eff" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">frozen weights</text><path d="M 530 60 L 572 60" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 580 60 l -10 -5 l 0 10 z" fill="#8892a4" /><rect x="580" y="32" width="120" height="56" rx="6" fill="none" stroke="#e8eaf0" stroke-width="2" /><text x="640" y="66" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">answer</text>
  <text x="20" y="171" fill="#50c878" font-size="18" font-family="Inter, sans-serif">test-time</text><text x="20" y="193" fill="#50c878" font-size="18" font-family="Inter, sans-serif">training</text>
  <rect x="170" y="133" width="110" height="64" rx="6" fill="none" stroke="#e8eaf0" stroke-width="2" /><text x="225" y="171" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">input</text><path d="M 280 165 L 302 165" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 310 165 l -10 -5 l 0 10 z" fill="#8892a4" /><rect x="310" y="133" width="190" height="64" rx="6" fill="none" stroke="#f5a623" stroke-width="2" /><text x="405" y="162" fill="#f5a623" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">make a task</text><text x="405" y="183" fill="#8892a4" font-size="14" font-family="Inter, sans-serif" text-anchor="middle">from the input</text><path d="M 500 165 L 522 165" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 530 165 l -10 -5 l 0 10 z" fill="#8892a4" /><rect x="530" y="133" width="200" height="64" rx="6" fill="none" stroke="#50c878" stroke-width="3" /><text x="630" y="162" fill="#50c878" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">gradient steps</text><text x="630" y="183" fill="#8892a4" font-size="14" font-family="Inter, sans-serif" text-anchor="middle">on a copy of the weights</text><path d="M 730 165 L 752 165" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 760 165 l -10 -5 l 0 10 z" fill="#8892a4" /><rect x="760" y="133" width="100" height="64" rx="6" fill="none" stroke="#e8eaf0" stroke-width="2" /><text x="810" y="171" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">answer</text><path d="M 860 165 L 882 165" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 890 165 l -10 -5 l 0 10 z" fill="#8892a4" /><rect x="890" y="133" width="95" height="64" rx="6" fill="none" stroke="#8892a4" stroke-width="2" /><text x="937" y="162" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">discard</text><text x="937" y="183" fill="#8892a4" font-size="14" font-family="Inter, sans-serif" text-anchor="middle">the update</text>
</svg>
</div>

- **Sun et al. (2020)**: an image classifier adapts to each test image by predicting how the image was rotated, a label it can make for itself
- **Akyürek et al. (2024)**: before answering an ARC puzzle, an 8B language model fine-tunes on that puzzle's worked examples. Accuracy rose up to 6 times over fine-tuned baselines
- The update is thrown away after the answer, so the model learns nothing that lasts

---

<!-- .slide: id="learning-diagram" -->

## Test-Time Training Nests Two Loops

<div class="svg-figure">
<svg viewBox="0 0 1000 360" role="img" aria-label="An outer loop of pretraining runs once over a corpus and produces weights. At inference, an inner loop takes gradient steps on each input, then discards them.">
  <rect x="20" y="24" width="960" height="310" rx="10" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="44" y="58" fill="#4a9eff" font-size="22" font-family="Inter, sans-serif">Outer loop: pretraining</text>
  <text x="44" y="88" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">runs once, over a corpus, for months</text>
  <text x="44" y="118" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">learns weights that make the inner loop useful</text>
  <path d="M 300 152 L 300 190" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 300 190 l -5 -10 l 10 0 z" fill="#4a9eff" />
  <text x="316" y="178" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">produces the deployed model</text>
  <rect x="60" y="200" width="880" height="112" rx="8" fill="none" stroke="#50c878" stroke-width="2" />
  <text x="84" y="236" fill="#50c878" font-size="22" font-family="Inter, sans-serif">Inner loop: on each test input</text>
  <text x="84" y="266" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">a few gradient steps on the input, or one per token in a TTT layer</text>
  <text x="84" y="296" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">adapts to this input, then the update is thrown away</text>
  <path d="M 890 226 a 30 30 0 1 1 -22 -29" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 868 197 l 2 11 l 9 -6 z" fill="#50c878" />
  <text x="862" y="292" fill="#50c878" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">per input</text>
</svg>
</div>

This is the test-time training picture: a training phase, then a fast loop at inference that forgets each input. Nested learning removes the boundary between the two. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="learning-ttt-layers" -->

## TTT Layers Train a Small Model at Every Token

Sun et al. (2024) put the same idea inside a layer. The layer's memory is the **weights $W$ of a small model** $f_W$.

:::columns cols="2" gap="34px"
**Each token**

- Training example: input $\mathbf k_t$, target $\mathbf v_t$, both projections of the token
- Update: one gradient step on $\lVert f_W(\mathbf k_t) - \mathbf v_t \rVert^2$
- Output: $f_W(\mathbf q_t)$, the small model applied to the query
+++
**The choice of small model**

- **Linear** $f_W$: the update becomes the delta rule, a cousin of linear attention
- **Two-layer MLP** $f_W$: more capacity than any fixed-size matrix memory
- Ancestor: Schmidhuber's fast weights (1992), one network writing another's weights
:::

---

<!-- .slide: id="learning-titans" -->

## Titans Remembers What Surprised It

**Titans** pushes the inner loop from a per-sequence scratchpad toward persistent memory, with a rule for what gets written:

<div class="metric-box">

Store what the model predicted **badly**.

</div>

- A correctly predicted token carries no new information; a surprising one does
- The loss doubles as the rule for **what to write** to memory

**Nested learning** (Behrouz et al., NeurIPS 2025) goes further. It drops the split between training and inference, and treats the optimizer as a memory too.

---

<!-- .slide: id="learning-sgd-memory" -->

## Gradient Descent Writes Like Linear Attention

Train one linear layer $\mathbf y = W\mathbf x$ with gradient descent. Each step has the same shape as a linear-attention write.

$$\text{linear attention:} \quad \mathbf S_t = \mathbf S_{t-1} + \mathbf k_t \textcolor{#50c878}{\mathbf v_t^\top}$$

$$\text{gradient descent:} \quad W_{t+1} = W_t - \eta \textcolor{#50c878}{\nabla_{\mathbf y} L} \mathbf x_t^\top$$

:::columns cols="2" gap="34px"
**Read the second line as a memory write**

- The key is the input $\mathbf x_t$
- The value is the **error signal** at the output
- The weights store which inputs led to which mistakes
+++
**What follows**

- Training is the model memorizing its own surprise, one example at a time
- Pretraining and in-context memory are **the same kind of process**
- They differ in how often they update
:::

---

<!-- .slide: id="learning-no-phases" -->

## Nested Learning Replaces Phases With Speeds

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="Left, today's models: the weights are trained once during pretraining and then frozen, and each conversation's context is erased afterward, with nothing passing from context into weights. Right, nested learning: one ongoing process in which the attention context updates every token, a fast memory every 4 tokens, a slow memory every 12, and the slowest weights rarely but without ever stopping.">
  <text x="20" y="30" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif">Today: two phases</text>
  <text x="20" y="100" fill="#f5a623" font-size="17" font-family="Inter, sans-serif">weights</text>
  <rect x="120" y="82" width="140" height="28" fill="#f5a623" fill-opacity="0.25" stroke="#f5a623" stroke-width="2" />
  <text x="190" y="102" fill="#f5a623" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">pretraining</text>
  <path d="M 260 96 L 470 96" stroke="#8892a4" stroke-width="2" stroke-dasharray="6 5" fill="none" />
  <text x="365" y="132" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">frozen after release</text>
  <text x="20" y="200" fill="#a06bd4" font-size="17" font-family="Inter, sans-serif">context</text>
  <g fill="#a06bd4" fill-opacity="0.25" stroke="#a06bd4" stroke-width="2"><rect x="280" y="182" width="50" height="28" /><rect x="345" y="182" width="50" height="28" /><rect x="410" y="182" width="50" height="28" /></g>
  <text x="370" y="234" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">one chat each, then erased</text>
  <text x="245" y="282" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">Nothing in a chat reaches the weights.</text>
  <line x1="497" y1="15" x2="497" y2="290" stroke="#2a3450" stroke-width="2" />
  <text x="525" y="30" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif">Nested learning: one process, many speeds</text>
  <text x="525" y="80" fill="#a06bd4" font-size="17" font-family="Inter, sans-serif">attention context</text>
  <text x="525" y="99" fill="#8892a4" font-size="14" font-family="Inter, sans-serif">every token</text>
  <line x1="720" y1="88" x2="975" y2="88" stroke="#2a3450" stroke-width="2" />
  <g stroke="#a06bd4" stroke-width="2"><line x1="730" y1="64" x2="730" y2="88" /> <line x1="739" y1="64" x2="739" y2="88" /> <line x1="748" y1="64" x2="748" y2="88" /> <line x1="757" y1="64" x2="757" y2="88" /> <line x1="766" y1="64" x2="766" y2="88" /> <line x1="775" y1="64" x2="775" y2="88" /> <line x1="784" y1="64" x2="784" y2="88" /> <line x1="793" y1="64" x2="793" y2="88" /> <line x1="802" y1="64" x2="802" y2="88" /> <line x1="811" y1="64" x2="811" y2="88" /> <line x1="820" y1="64" x2="820" y2="88" /> <line x1="829" y1="64" x2="829" y2="88" /> <line x1="838" y1="64" x2="838" y2="88" /> <line x1="847" y1="64" x2="847" y2="88" /> <line x1="856" y1="64" x2="856" y2="88" /> <line x1="865" y1="64" x2="865" y2="88" /> <line x1="874" y1="64" x2="874" y2="88" /> <line x1="883" y1="64" x2="883" y2="88" /> <line x1="892" y1="64" x2="892" y2="88" /> <line x1="901" y1="64" x2="901" y2="88" /> <line x1="910" y1="64" x2="910" y2="88" /> <line x1="919" y1="64" x2="919" y2="88" /> <line x1="928" y1="64" x2="928" y2="88" /> <line x1="937" y1="64" x2="937" y2="88" /> <line x1="946" y1="64" x2="946" y2="88" /> <line x1="955" y1="64" x2="955" y2="88" /> <line x1="964" y1="64" x2="964" y2="88" /></g>
  <text x="525" y="135" fill="#50c878" font-size="17" font-family="Inter, sans-serif">fast memory</text>
  <text x="525" y="154" fill="#8892a4" font-size="14" font-family="Inter, sans-serif">every 4 tokens</text>
  <line x1="720" y1="143" x2="975" y2="143" stroke="#2a3450" stroke-width="2" />
  <g stroke="#50c878" stroke-width="3"><line x1="757" y1="119" x2="757" y2="143" /> <line x1="793" y1="119" x2="793" y2="143" /> <line x1="829" y1="119" x2="829" y2="143" /> <line x1="865" y1="119" x2="865" y2="143" /> <line x1="901" y1="119" x2="901" y2="143" /> <line x1="937" y1="119" x2="937" y2="143" /></g>
  <text x="525" y="190" fill="#4a9eff" font-size="17" font-family="Inter, sans-serif">slow memory</text>
  <text x="525" y="209" fill="#8892a4" font-size="14" font-family="Inter, sans-serif">every 12 tokens</text>
  <line x1="720" y1="198" x2="975" y2="198" stroke="#2a3450" stroke-width="2" />
  <g stroke="#4a9eff" stroke-width="3"><line x1="829" y1="174" x2="829" y2="198" /> <line x1="937" y1="174" x2="937" y2="198" /></g>
  <text x="525" y="245" fill="#f5a623" font-size="17" font-family="Inter, sans-serif">slowest weights</text>
  <text x="525" y="264" fill="#8892a4" font-size="14" font-family="Inter, sans-serif">rarely, but never stops</text>
  <line x1="720" y1="253" x2="975" y2="253" stroke="#2a3450" stroke-width="2" />
  <g stroke="#f5a623" stroke-width="3"><line x1="964" y1="229" x2="964" y2="253" /></g>
  <text x="750" y="290" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">Slow levels keep what fast levels learn for longer.</text>
</svg>
</div>

- Every part of the model is a memory that learns from its own stream of inputs. The parts differ only in **how often they update**
- Pretraining is the slowest level, and its input stream is the whole corpus. The paper calls it in-context learning with a very long context
- The paper compares today's LLMs to anterograde amnesia: the slow memory stopped at release, so nothing new becomes long-term

---

<!-- .slide: id="learning-momentum" -->

## Momentum Is a Short Memory of Gradients

$$\mathbf m_{t+1} = \beta \mathbf m_t + \nabla L(W_t) \qquad W_{t+1} = W_t - \eta \mathbf m_{t+1}$$

<div class="svg-figure">
<svg viewBox="0 0 1000 290" role="img" aria-label="Bar chart of how much each past gradient contributes to the current momentum when beta is 0.9. The contribution shrinks by a factor of 0.9 per step. The most recent 6 gradients make up half of the total and the most recent 43 make up 99 percent. Everything older is nearly forgotten.">
  <rect x="950" y="50.0" width="13" height="180.0" fill="#50c878" fill-opacity="0.7" />
  <rect x="933" y="68.0" width="13" height="162.0" fill="#50c878" fill-opacity="0.7" />
  <rect x="916" y="84.19999999999999" width="13" height="145.8" fill="#50c878" fill-opacity="0.7" />
  <rect x="899" y="98.80000000000001" width="13" height="131.2" fill="#50c878" fill-opacity="0.7" />
  <rect x="882" y="111.9" width="13" height="118.1" fill="#50c878" fill-opacity="0.7" />
  <rect x="865" y="123.7" width="13" height="106.3" fill="#50c878" fill-opacity="0.7" />
  <rect x="848" y="134.3" width="13" height="95.7" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="831" y="143.9" width="13" height="86.1" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="814" y="152.5" width="13" height="77.5" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="797" y="160.3" width="13" height="69.7" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="780" y="167.2" width="13" height="62.8" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="763" y="173.5" width="13" height="56.5" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="746" y="179.2" width="13" height="50.8" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="729" y="184.2" width="13" height="45.8" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="712" y="188.8" width="13" height="41.2" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="695" y="192.9" width="13" height="37.1" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="678" y="196.6" width="13" height="33.4" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="661" y="200.0" width="13" height="30.0" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="644" y="203.0" width="13" height="27.0" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="627" y="205.7" width="13" height="24.3" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="610" y="208.1" width="13" height="21.9" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="593" y="210.3" width="13" height="19.7" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="576" y="212.3" width="13" height="17.7" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="559" y="214.0" width="13" height="16.0" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="542" y="215.6" width="13" height="14.4" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="525" y="217.1" width="13" height="12.9" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="508" y="218.4" width="13" height="11.6" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="491" y="219.5" width="13" height="10.5" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="474" y="220.6" width="13" height="9.4" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="457" y="221.5" width="13" height="8.5" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="440" y="222.4" width="13" height="7.6" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="423" y="223.1" width="13" height="6.9" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="406" y="223.8" width="13" height="6.2" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="389" y="224.4" width="13" height="5.6" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="372" y="225.0" width="13" height="5.0" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="355" y="225.5" width="13" height="4.5" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="338" y="225.9" width="13" height="4.1" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="321" y="226.4" width="13" height="3.6" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="304" y="226.7" width="13" height="3.3" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="287" y="227.0" width="13" height="3.0" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="270" y="227.3" width="13" height="2.7" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="253" y="227.6" width="13" height="2.4" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="236" y="227.8" width="13" height="2.2" fill="#4a9eff" fill-opacity="0.7" />
  <rect x="219" y="228.1" width="13" height="1.9" fill="#8892a4" fill-opacity="0.7" />
  <rect x="202" y="228.3" width="13" height="1.7" fill="#8892a4" fill-opacity="0.7" />
  <rect x="185" y="228.4" width="13" height="1.6" fill="#8892a4" fill-opacity="0.7" />
  <rect x="168" y="228.6" width="13" height="1.4" fill="#8892a4" fill-opacity="0.7" />
  <rect x="151" y="228.7" width="13" height="1.3" fill="#8892a4" fill-opacity="0.7" />
  <rect x="134" y="228.9" width="13" height="1.1" fill="#8892a4" fill-opacity="0.7" />
  <rect x="117" y="229.0" width="13" height="1.0" fill="#8892a4" fill-opacity="0.7" />
  <line x1="40" y1="230" x2="970" y2="230" stroke="#8892a4" stroke-width="1.5" />
  <text x="956" y="256" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">now</text>
  <text x="117" y="256" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">50 steps ago</text>
  <text x="560" y="40" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">Weight of each past gradient in today's momentum, &#946; = 0.9</text>
  <text x="900" y="80" fill="#50c878" font-size="18" font-family="Inter, sans-serif" text-anchor="end">last 6 steps: 50%</text>
  <text x="560" y="180" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">last 43 steps: 99%</text>
  <text x="180" y="200" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">older: under 1%</text>
  <text x="500" y="284" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">Momentum compresses the gradient stream into one vector, and it keeps only the recent past.</text>
</svg>
</div>

Momentum is a second, faster learner inside the optimizer. Gradient descent with momentum is therefore a **two-level** optimization, and the inner level forgets fast. That is one reason training on a new task erases the old one. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="learning-deep-momentum" -->

## Deep Optimizers Are Neural Networks

<div class="svg-figure">
<svg viewBox="0 0 1000 250" role="img" aria-label="Two optimizers side by side. Top, standard momentum: the gradient goes into a fixed rule that keeps one running-average vector, and that vector updates the weights. Bottom, deep momentum: the gradient goes into a small neural network with its own weights, whose output updates the model's weights. A loop shows the small network taking its own gradient step every time, learning to store the gradients it has seen.">
  <text x="20" y="56" fill="#4a9eff" font-size="19" font-family="Inter, sans-serif">momentum</text>
  <text x="20" y="78" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">a fixed rule</text>
  <text x="20" y="156" fill="#50c878" font-size="19" font-family="Inter, sans-serif">deep momentum</text>
  <text x="20" y="178" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">DMGD</text>
  <g fill="none" stroke="#8892a4" stroke-width="2"><rect x="180" y="40" width="130" height="50" rx="6" /><rect x="180" y="140" width="130" height="50" rx="6" /><rect x="720" y="40" width="160" height="50" rx="6" /><rect x="720" y="140" width="160" height="50" rx="6" /></g>
  <rect x="370" y="34" width="290" height="62" rx="6" fill="none" stroke="#4a9eff" stroke-width="2" />
  <rect x="370" y="134" width="290" height="62" rx="6" fill="none" stroke="#50c878" stroke-width="3" />
  <g font-family="Inter, sans-serif" text-anchor="middle">
    <text x="245" y="71" fill="#e8eaf0" font-size="18">gradient</text><text x="245" y="171" fill="#e8eaf0" font-size="18">gradient</text>
    <text x="800" y="71" fill="#e8eaf0" font-size="18">update weights</text><text x="800" y="171" fill="#e8eaf0" font-size="18">update weights</text>
    <text x="515" y="62" fill="#4a9eff" font-size="18">m ← βm + gradient</text><text x="515" y="85" fill="#8892a4" font-size="15">one vector, a running average</text>
    <text x="515" y="162" fill="#50c878" font-size="18">small neural network</text><text x="515" y="185" fill="#8892a4" font-size="15">has its own weights</text>
  </g>
  <g stroke="#8892a4" stroke-width="2" fill="none"><path d="M 310 65 L 362 65" /><path d="M 660 65 L 712 65" /><path d="M 310 165 L 362 165" /><path d="M 660 165 L 712 165" /></g>
  <g fill="#8892a4"><path d="M 370 65 l -10 -5 l 0 10 z" /><path d="M 720 65 l -10 -5 l 0 10 z" /><path d="M 370 165 l -10 -5 l 0 10 z" /><path d="M 720 165 l -10 -5 l 0 10 z" /></g>
  <path d="M 600 196 L 600 222 L 430 222 L 430 204" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 430 196 l -6 10 l 12 0 z" fill="#50c878" />
  <text x="515" y="244" fill="#50c878" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">each step, its weights take their own gradient step to store the new gradient</text>
</svg>
</div>

- Momentum stores past gradients in one vector, updated by a fixed rule. Deep momentum (DMGD) stores them in a small **neural network**
- Every optimizer step, that network trains on the new gradient with its own loss. The model's weights then move by the network's output
- A classic learned optimizer is trained once and frozen. This one keeps learning while it optimizes, which is the point for continual learning

---

<!-- .slide: id="learning-deep-optimizers" -->

## Nested Learning Upgrades Optimizer Memory

If momentum is a memory, the upgrades that improved linear attention apply to it too.

<div class="bench-table dense">
<table>
<thead><tr><th>Optimizer memory</th><th>Upgrade</th><th>Linear-attention analogue</th></tr></thead>
<tbody>
<tr><td><strong>Standard momentum</strong></td><td>A fading running sum of gradients</td><td>Linear attention with decay</td></tr>
<tr><td><strong>Adam</strong></td><td>Also stores each gradient's running variance</td><td>A second running sum, used as a normalizer</td></tr>
<tr><td><strong>Delta momentum</strong></td><td>Writes with the delta rule, so it can overwrite stale gradients</td><td>DeltaNet</td></tr>
<tr><td><strong>Deep momentum</strong> (DMGD)</td><td>Replaces the momentum vector with a small MLP</td><td>Test-time training with a nonlinear inner model</td></tr>
<tr><td><strong>M3</strong></td><td>A fast momentum updated every step and a slow one updated every $C$ steps</td><td>Memory at two timescales</td></tr>
</tbody>
</table>
</div>

The authors present M3 as a proof of concept. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="learning-frequency" -->

## Memories Ordered by Update Speed

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="Thirty-two tokens of a sequence, with a tick each time a component updates. The attention context changes on every token. A fast MLP block updates every 4 tokens and a slow one every 16. The pretrained weights never update after training.">
  <text x="630" y="30" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">32 tokens of input, one tick per update</text>
  <text x="20" y="74" fill="#a06bd4" font-size="19" font-family="Inter, sans-serif">Attention context</text>
  <text x="20" y="96" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">every token</text>
  <line x1="300" y1="90" x2="960" y2="90" stroke="#2a3450" stroke-width="2" />
  <line x1="320" y1="60" x2="320" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="340" y1="60" x2="340" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="360" y1="60" x2="360" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="380" y1="60" x2="380" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="400" y1="60" x2="400" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="420" y1="60" x2="420" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="440" y1="60" x2="440" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="460" y1="60" x2="460" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="480" y1="60" x2="480" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="500" y1="60" x2="500" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="520" y1="60" x2="520" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="540" y1="60" x2="540" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="560" y1="60" x2="560" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="580" y1="60" x2="580" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="600" y1="60" x2="600" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="620" y1="60" x2="620" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="640" y1="60" x2="640" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="660" y1="60" x2="660" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="680" y1="60" x2="680" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="700" y1="60" x2="700" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="720" y1="60" x2="720" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="740" y1="60" x2="740" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="760" y1="60" x2="760" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="780" y1="60" x2="780" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="800" y1="60" x2="800" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="820" y1="60" x2="820" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="840" y1="60" x2="840" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="860" y1="60" x2="860" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="880" y1="60" x2="880" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="900" y1="60" x2="900" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="920" y1="60" x2="920" y2="90" stroke="#a06bd4" stroke-width="2" />
  <line x1="940" y1="60" x2="940" y2="90" stroke="#a06bd4" stroke-width="2" />
  <text x="20" y="130" fill="#50c878" font-size="19" font-family="Inter, sans-serif">Fast MLP block</text>
  <text x="20" y="152" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">every 4 tokens</text>
  <line x1="300" y1="146" x2="960" y2="146" stroke="#2a3450" stroke-width="2" />
  <line x1="380" y1="116" x2="380" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="460" y1="116" x2="460" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="540" y1="116" x2="540" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="620" y1="116" x2="620" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="700" y1="116" x2="700" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="780" y1="116" x2="780" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="860" y1="116" x2="860" y2="146" stroke="#50c878" stroke-width="3" />
  <line x1="940" y1="116" x2="940" y2="146" stroke="#50c878" stroke-width="3" />
  <text x="20" y="186" fill="#4a9eff" font-size="19" font-family="Inter, sans-serif">Slow MLP block</text>
  <text x="20" y="208" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">every 16 tokens</text>
  <line x1="300" y1="202" x2="960" y2="202" stroke="#2a3450" stroke-width="2" />
  <line x1="620" y1="172" x2="620" y2="202" stroke="#4a9eff" stroke-width="3" />
  <line x1="940" y1="172" x2="940" y2="202" stroke="#4a9eff" stroke-width="3" />
  <text x="20" y="242" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">Standard MLP</text>
  <text x="20" y="264" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">frozen after release</text>
  <line x1="300" y1="258" x2="960" y2="258" stroke="#2a3450" stroke-width="2" />
</svg>
</div>

- A standard transformer has two speeds: attention changes every token, and the MLP weights stop changing at release
- The **continuum memory system** (CMS) replaces the MLP with blocks that each keep updating on their own schedule, including after release
- Fast levels adapt quickly and forget quickly. Slow levels keep knowledge longer
- The paper's analogy is brain waves, from fast gamma (30 to 150 Hz) down to slow delta and theta (0.5 to 8 Hz)

---

<!-- .slide: id="learning-hope" -->

## HOPE Stacks a Fast Learner on a Slow Memory

<div class="svg-figure">
<svg viewBox="0 0 1000 260" role="img" aria-label="Tokens enter a self-modifying Titans module, which is small but learns its own update rule. Its output passes through a continuum memory system of three MLP blocks that update at fast, medium, and slow frequencies. The result is the output.">
  <rect x="20" y="80" width="100" height="60" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="70" y="116" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">tokens</text>
  <path d="M 120 110 L 152 110" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 160 110 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="160" y="60" width="240" height="100" rx="8" fill="none" stroke="#f5a623" stroke-width="3" />
  <text x="280" y="98" fill="#f5a623" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Self-modifying Titans</text>
  <text x="280" y="126" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">learns its own update rule</text>
  <text x="280" y="148" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">small capacity</text>
  <path d="M 400 110 L 432 110" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 440 110 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="440" y="40" width="420" height="140" rx="8" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="650" y="30" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">Continuum memory system</text>
  <g fill="none" stroke-width="2">
    <rect x="460" y="80" width="110" height="60" stroke="#50c878" /><rect x="595" y="80" width="110" height="60" stroke="#4a9eff" /><rect x="730" y="80" width="110" height="60" stroke="#8892a4" />
  </g>
  <g font-size="17" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="515" y="108" fill="#50c878">MLP</text><text x="650" y="108" fill="#4a9eff">MLP</text><text x="785" y="108" fill="#e8eaf0">MLP</text>
    <text x="515" y="130" fill="#8892a4" font-size="15">fast</text><text x="650" y="130" fill="#8892a4" font-size="15">medium</text><text x="785" y="130" fill="#8892a4" font-size="15">slow</text>
  </g>
  <path d="M 570 110 L 593 110 M 705 110 L 728 110" stroke="#8892a4" stroke-width="2" fill="none" />
  <text x="650" y="166" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">large capacity, simple update rule</text>
  <path d="M 860 110 L 892 110" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 900 110 l -12 -6 l 0 12 z" fill="#8892a4" />
  <text x="945" y="116" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">output</text>
  <text x="500" y="230" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">The two halves cover each other: one adapts cleverly but holds little, the other holds a lot but adapts simply.</text>
</svg>
</div>

- In continual-learning tests, HOPE beat in-context learning alone and elastic weight consolidation
- On a sequence of new languages learned one after another, adding more memory levels reduced forgetting
- The paper itself asks "Is catastrophic forgetting solved?" and does not claim it is

---

<!-- .slide: id="learning-stakes" -->

## Consequences for Deployment

If this assumption falls, "deploying a model" stops meaning "shipping a frozen artifact."

<div class="bench-table">
<table>
<thead><tr><th>What breaks</th><th>Why</th></tr></thead>
<tbody>
<tr><td><strong>Evaluation</strong></td><td>Evaluate then ship assumes the thing you tested is the thing you serve. A learning model's eval results have an <strong>expiry date</strong></td></tr>
<tr><td><strong>Serving</strong></td><td>Weights are a read-only asset replicated across GPUs. Replicas that keep learning <strong>drift apart</strong>, and now you have a consistency problem</td></tr>
<tr><td><strong>Applications</strong></td><td>Retrieval exists because the model cannot remember. Some of that machinery becomes unnecessary, and some becomes load-bearing in a new way</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="learning-verdict" -->

## Continual Learning Is the Least Settled

The least settled of the three sections, by a wide margin.

<div class="card-grid cols-3">
<div class="card"><h4>Published</h4><p>Test-time training layers exist and work, demonstrated at <strong>small scale</strong>.</p></div>
<div class="card"><h4>Research program</h4><p>Nested learning is a framing and a set of results, <strong>not a product</strong>.</p></div>
<div class="card"><h4>Still open</h4><p>The distance between a frozen model you prompt and a system that <strong>accumulates experience over months</strong> is wide.</p></div>
</div>

The assumption whose removal would change the most, with the least agreement about how. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
