:::divider id="divider-neuro" title="Can a Network Run a Program?" sub="Neural networks and symbolic computation"
:::

---

<!-- .slide: id="neuro-gap" -->

## Networks Approximate Algorithms

A transformer that learned addition from text has not learned the addition algorithm. It fails on numbers longer than any it trained on.

:::columns cols="2" gap="34px"
**Neural computation**

- Learns its rules from data
- Handles noise, images, and language
- Fails silently outside the training range
- Hard to inspect
+++
**Symbolic computation**

- Rules are written down
- Exact on inputs of any length
- Breaks on messy, ambiguous input
- Every step can be checked
:::

**Neurosymbolic** research tries to get both columns at once. The question is where the symbolic part lives. <!-- .element: class="text-lg" style="margin-top: 12px;" -->


---

<!-- .slide: id="neuro-ntm" -->

## Neural Turing Machines Added a Memory Tape

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="A controller network reads from and writes to a memory matrix of eight rows. The read head spreads soft attention weights over the rows, peaked on row three. The write head puts most of its weight on row six. Because the weights are soft, the whole system is differentiable.">
  <rect x="40" y="110" width="200" height="90" fill="none" stroke="#e8eaf0" stroke-width="2" />
  <text x="140" y="150" fill="#e8eaf0" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Controller</text>
  <text x="140" y="178" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">LSTM or MLP</text>
  <path d="M 240 135 L 455 110" stroke="#50c878" stroke-width="2" fill="none" />
  <text x="300" y="104" fill="#50c878" font-size="18" font-family="Inter, sans-serif">read head</text>
  <path d="M 240 175 L 455 200" stroke="#f5a623" stroke-width="2" fill="none" />
  <text x="290" y="226" fill="#f5a623" font-size="18" font-family="Inter, sans-serif">write head</text>
  <text x="660" y="30" fill="#4a9eff" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">memory matrix</text>
  <text x="488" y="30" fill="#50c878" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">read weights</text>
  <text x="832" y="30" fill="#f5a623" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">write weights</text>
  <rect x="560" y="40" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="542" y="46" width="6" height="16" fill="#50c878" fill-opacity="0.6" />
  <rect x="560" y="68" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="536" y="74" width="12" height="16" fill="#50c878" fill-opacity="0.6" />
  <rect x="560" y="96" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="464" y="102" width="84" height="16" fill="#50c878" fill-opacity="0.6" />
  <rect x="560" y="124" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="536" y="130" width="12" height="16" fill="#50c878" fill-opacity="0.6" />
  <rect x="560" y="152" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="542" y="158" width="6" height="16" fill="#50c878" fill-opacity="0.6" />
  <rect x="772" y="158" width="12" height="16" fill="#f5a623" fill-opacity="0.6" />
  <rect x="560" y="180" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="772" y="186" width="96" height="16" fill="#f5a623" fill-opacity="0.6" />
  <rect x="560" y="208" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <rect x="772" y="214" width="12" height="16" fill="#f5a623" fill-opacity="0.6" />
  <rect x="560" y="236" width="200" height="28" fill="none" stroke="#4a9eff" stroke-width="1.5" />
  <text x="500" y="292" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">Addressing is soft attention over rows, so gradients reach every read and write.</text>
</svg>
</div>

- **NTM** (Graves et al., 2014) learned to copy, sort, and recall sequences from examples alone
- The **Differentiable Neural Computer** (2016) learned shortest paths on the London Underground map
- Both were hard to train and never scaled. Transformers kept the soft addressing and replaced the tape with the context window

---

<!-- .slide: id="neuro-rasp" -->

## RASP Is a Language for Transformers

Weiss, Goldberg, and Yahav (2021) wrote down what one transformer layer can do as two primitives.

:::columns cols="2" gap="34px"
**The primitives**

- `select` compares every query position with every key position. That is an **attention pattern**
- `aggregate` averages values under that pattern. That is the **attention output**
- Elementwise operations on each position are the **MLP**
+++
**Reversing a sequence**

```python
opp = length - indices - 1
flip = select(indices, opp, ==)
reverse = aggregate(flip, tokens)
```

Two attention layers: one to compute `length`, one to move each token to its mirrored position.
:::

The longest chain of dependent `select` calls predicts how many layers a trained transformer needs for the task. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="neuro-compile" -->

## Compilers Turn Programs Into Weights

<div class="svg-figure">
<svg viewBox="0 0 1000 250" role="img" aria-label="A pipeline. A program written in RASP or ALTA goes into a compiler, Tracr or the ALTA compiler, which sets every weight of a transformer by hand. The resulting transformer runs the algorithm exactly, with no training.">
  <rect x="20" y="70" width="230" height="90" rx="8" fill="none" stroke="#a06bd4" stroke-width="2" />
  <text x="135" y="108" fill="#a06bd4" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Program</text>
  <text x="135" y="138" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">RASP or ALTA source</text>
  <path d="M 250 115 L 352 115" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 360 115 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="370" y="70" width="230" height="90" rx="8" fill="none" stroke="#f5a623" stroke-width="2" />
  <text x="485" y="108" fill="#f5a623" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Compiler</text>
  <text x="485" y="138" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">Tracr (2023), ALTA (2024)</text>
  <path d="M 600 115 L 702 115" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 710 115 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="720" y="70" width="260" height="90" rx="8" fill="none" stroke="#50c878" stroke-width="2" />
  <text x="850" y="108" fill="#50c878" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Transformer weights</text>
  <text x="850" y="138" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">every value set by hand</text>
  <text x="500" y="214" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">No gradient descent anywhere. The network runs the algorithm exactly.</text>
</svg>
</div>

- Interpretability tools get a test case: a network whose circuit is already known
- Compiled modules can be attached to a trained model to give it exact skills
- A compiled program proves that some set of weights can do the task. Whether training would ever find those weights is a separate question

---

<!-- .slide: id="neuro-conn" -->

## Compiled Modules Inside a Language Model

<div class="svg-figure">
<svg viewBox="0 0 1000 280" role="img" aria-label="A pretrained language model and a set of compiled networks for addition, subtraction, and parity read the same input. A gate at each position chooses between the language model's prediction and the compiled network's exact output. The chosen token is emitted.">
  <rect x="40" y="30" width="200" height="56" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="140" y="64" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">"731 + 498 ="</text>
  <path d="M 140 86 L 140 112 L 260 112 L 260 132" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 140 86 L 140 112 L 600 112 L 600 132" stroke="#8892a4" stroke-width="2" fill="none" />
  <rect x="140" y="132" width="240" height="70" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="260" y="162" fill="#4a9eff" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Pretrained LM</text>
  <text x="260" y="188" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">language, fuzzy arithmetic</text>
  <rect x="480" y="132" width="240" height="70" fill="none" stroke="#a06bd4" stroke-width="2" />
  <text x="600" y="162" fill="#a06bd4" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Compiled networks</text>
  <text x="600" y="188" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">add, subtract, parity: exact</text>
  <path d="M 260 202 L 260 226 L 840 226 L 840 204" stroke="#4a9eff" stroke-width="2" fill="none" stroke-dasharray="6 5" />
  <path d="M 720 167 L 808 167" stroke="#a06bd4" stroke-width="2" fill="none" />
  <circle cx="840" cy="172" r="30" fill="none" stroke="#f5a623" stroke-width="2" />
  <text x="840" y="179" fill="#f5a623" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">gate</text>
  <path d="M 870 172 L 900 172" stroke="#f5a623" stroke-width="2" fill="none" />
  <text x="940" y="179" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">1229</text>
  <text x="500" y="262" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">At each position the gate picks whichever module should produce the next token.</text>
</svg>
</div>

- **Neural Comprehension** (Weng et al., ICLR 2024): the compiled networks' attention weights are set by hand to encode the rule
- Arithmetic stays exact on numbers longer than chain-of-thought prompting can handle
- No tool call and no round trip: the rule runs inside the forward pass, with few added parameters

---

<!-- .slide: id="neuro-alta" -->

## ALTA Compiles Loops and Tests Learnability

Shaw et al. (2024, TMLR 2025) extend RASP and Tracr in two directions.

:::columns cols="2" gap="34px"
**Loops**

- ALTA programs can loop
- They compile to **Universal Transformer** weights, one shared block run repeatedly
- The looped block from the previous section, written by hand
- Length-invariant parity and addition, without scratchpad tokens
+++
**Learnability**

- The weights exist. Does training on examples find them?
- Often it does not, even when the program is small
- Supervising each step of the program's execution trace helps
- Relative position encodings let some length generalization appear; absolute ones allow none
:::

Expressiveness asks whether a solution exists in weight space. Learnability asks whether gradient descent on a given dataset reaches it. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="neuro-scallop" -->

## Scallop Puts Logic Inside the Training Loop

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="Two handwritten digit images go into a neural network that outputs a probability for each digit, for example 3 with probability 0.9 and 5 with probability 0.8. A logic rule, sum equals a plus b, combines them into a probability for each possible sum. The only label is the sum, 8. The gradient flows back through the logic rule into the network.">
  <rect x="30" y="50" width="70" height="70" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="65" y="98" fill="#e8eaf0" font-size="40" font-family="Georgia, serif" text-anchor="middle" font-style="italic">3</text>
  <rect x="30" y="150" width="70" height="70" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="65" y="198" fill="#e8eaf0" font-size="40" font-family="Georgia, serif" text-anchor="middle" font-style="italic">5</text>
  <path d="M 100 85 L 158 120" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 100 185 L 158 150" stroke="#8892a4" stroke-width="2" fill="none" />
  <rect x="160" y="95" width="150" height="80" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="235" y="130" fill="#4a9eff" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Neural net</text>
  <text x="235" y="156" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">perception</text>
  <path d="M 310 135 L 362 135" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 370 135 l -12 -6 l 0 12 z" fill="#4a9eff" />
  <text x="380" y="110" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">digit 1 = 3 (0.9)</text>
  <text x="380" y="140" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">digit 2 = 5 (0.8)</text>
  <text x="380" y="170" fill="#8892a4" font-size="16" font-family="Inter, sans-serif">plus the other digits</text>
  <path d="M 560 135 L 582 135" stroke="#a06bd4" stroke-width="2" fill="none" />
  <path d="M 590 135 l -12 -6 l 0 12 z" fill="#a06bd4" />
  <rect x="595" y="95" width="250" height="80" fill="none" stroke="#a06bd4" stroke-width="2" />
  <text x="720" y="130" fill="#a06bd4" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Logic rule</text>
  <text x="720" y="156" fill="#e8eaf0" font-size="15" font-family="Menlo, monospace" text-anchor="middle">sum(a+b) :- d1(a), d2(b)</text>
  <path d="M 845 135 L 862 135" stroke="#a06bd4" stroke-width="2" fill="none" />
  <path d="M 870 135 l -12 -6 l 0 12 z" fill="#a06bd4" />
  <text x="925" y="141" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">sum = 8</text>
  <path d="M 900 200 C 900 260, 240 260, 240 182" stroke="#f5a623" stroke-width="2" stroke-dasharray="7 6" fill="none" />
  <path d="M 240 178 l -6 12 l 12 0 z" fill="#f5a623" />
  <text x="570" y="284" fill="#f5a623" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">gradient flows back through the rule; the only label is the sum</text>
</svg>
</div>

- **Scallop** (Li, Huang, and Naik, PLDI 2023) is a Datalog-style logic language whose facts can carry probabilities
- The network never sees a digit label. It learns to read digits because the logic needs them to make the sum right
- Successor to DeepProbLog (2018). The rules are exact and readable; the perception is learned

---

<!-- .slide: id="neuro-spectrum" -->

## Where the Symbolic Part Lives

<div class="bench-table">
<table>
<thead><tr><th>Design</th><th>Examples</th><th>Trade-off</th></tr></thead>
<tbody>
<tr><td><strong>Compiled into the weights</strong></td><td>Tracr, CoNN, ALTA</td><td>Exact and fast, but every rule must be written by hand</td></tr>
<tr><td><strong>A differentiable logic layer</strong></td><td>Scallop, DeepProbLog</td><td>Readable rules trained end to end, but logic inference gets slow as rules grow</td></tr>
<tr><td><strong>The model calls a symbolic tool</strong></td><td>Code interpreters, AlphaGeometry, AlphaProof with Lean</td><td>Easy to bolt on and what ships today, but nothing is learned across the boundary</td></tr>
</tbody>
</table>
</div>

The tool-use loop from Module 11 is the third row. AlphaGeometry (2024) pairs a language model that proposes constructions with a symbolic engine that checks every deduction. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
