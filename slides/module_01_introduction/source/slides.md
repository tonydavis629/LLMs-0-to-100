:::divider id="title" title="LLMs 0 to 100" sub="Module 1: Course Introduction"
From Shannon to ChatGPT
:::

---

:::figure img="images/hopper.jpg" name="Grace Hopper" kicker="Pioneered the First Compiler"
- Created the A-0 System (1952) &mdash; the first compiler
- Believed computers should speak human languages, not the other way around
- Led development of COBOL &mdash; programs written in near-English
- Rear Admiral in the U.S. Navy
- **LLMs are the next step:** natural language compilers that translate plain text into tasks
:::

---

<!-- .slide: id="course-structure-1" -->

## Course Structure

<div class="content">

The course has four module groups: foundations, training, applications, and evaluation. <!-- .element: class="text-xl course-summary" -->

<div class="course-track-grid">
<div class="module-track-card">
<div class="module-track-kicker">Modules 1-4</div>

### Foundations

Concepts and vocabulary. <!-- .element: class="module-track-intro" -->

- **1. Introduction** &mdash; information theory and n-gram models
- **2. Perceptrons** &mdash; the simplest neural network
- **3. Attention** &mdash; deciding which earlier words matter
- **4. Transformers** &mdash; GPT-style architectures
<!-- .element: class="text-lg compact-list" -->

</div>
<div class="module-track-card">
<div class="module-track-kicker">Modules 5-7</div>

### Training

How models learn and improve. <!-- .element: class="module-track-intro" -->

- **5. Pretraining** &mdash; learning broad patterns from large text corpora
- **6. Fine-tuning** &mdash; adapting a model to a task
- **7. Reinforcement Learning** &mdash; improving responses with preference data
<!-- .element: class="text-lg compact-list" -->

</div>
</div>

</div>

---

<!-- .slide: id="course-structure-2" -->

## Course Structure

<div class="content">

The second half covers real-world use, evaluation, and newer research directions. <!-- .element: class="text-xl course-summary" -->

<div class="course-track-grid">
<div class="module-track-card">
<div class="module-track-kicker">Modules 8-10</div>

### Advanced Topics

Systems and applications. <!-- .element: class="module-track-intro" -->

- **8. Multimodal** &mdash; models that handle text plus images or audio
- **9. Deployment** &mdash; making models fast and reliable in production
- **10. Applications** &mdash; retrieval-augmented generation (RAG), agents, and tools
<!-- .element: class="text-lg compact-list" -->

</div>
<div class="module-track-card">
<div class="module-track-kicker">Modules 11-12</div>

### Evaluation &amp; Future Directions

Measurement and research frontiers. <!-- .element: class="module-track-intro" -->

- **11. Evaluation** &mdash; measuring quality, safety, and failure modes
- **12. Future Directions** &mdash; scaling laws, state space models (SSMs), and diffusion
<!-- .element: class="text-lg compact-list" -->

</div>
</div>

<div class="course-glossary">
<div class="glossary-item">

**RAG**

Retrieval-augmented generation: the model looks up outside information before answering.

</div>
<div class="glossary-item">

**SSM**

State space model: an alternative architecture for processing long sequences.

</div>
</div>

</div>

---

<!-- .slide: id="course-structure-3" -->

## How Each Module Works

<div class="content">

Each module follows the same pattern: lecture, exercise, quiz. <!-- .element: class="text-xl course-summary" -->

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 25px;">
<div class="highlight-box">

### Lecture

Concepts, history, and intuition. <!-- .element: class="module-track-intro" -->

- Slides with animated diagrams
- History and key figures
- Math and intuition
- Side quests for curious minds

</div>
<div class="highlight-box">

### Exercise

Code practice with small TODOs. <!-- .element: class="module-track-intro" -->

- Real Python project
- Skeleton with one-line TODOs
- Reference solution provided
- Extra credit for extra challenge

</div>
<div class="highlight-box">

### Quiz

Concept checks with click-to-reveal answers. <!-- .element: class="module-track-intro" -->

- Questions embedded in the lecture
- Answer to yourself first, next slide reveals answer
- Tests your understanding of key concepts

</div>
</div>

</div>

---

:::divider id="divider-talking-machines" title="We Built Talking Machines" sub="WHAT? How?"
:::

---

      <section id="timeline" style="padding: 40px 40px 50px 40px !important;">
        <h2>The Road to LLMs</h2>
        <div class="content" style="justify-content: center; padding-top: 15px;">
          <!-- Proportionally scaled timeline: 1948-2024 = 76 years -->
          <!-- All labels above the axis, alternating tall/short stems to avoid overlap in the dense right side -->
          <div style="position: relative; width: 100%; height: 300px;">
            <!-- Horizontal axis line at y=200px -->
            <div style="position: absolute; top: 200px; left: 0; right: 0; height: 3px; background: #2a3450;"></div>
            <!-- Tick marks on the axis for each milestone -->
            <!-- 1948: 0% | 1965: 22.4% | 1986: 50% | 2012: 84.2% | 2017: 90.8% | 2018: 92.1% | 2022: 97.4% | 2024: 100% -->
            <!-- All milestones: dot on axis, stem up, label above -->
            <!-- 1948: 0%, tall stem -->
            <div class="fragment tl-node" style="left: 0%; bottom: 98px; width: 80px;">
              <p class="tl-label">Information<br>Theory</p>
              <p class="tl-year" style="color: #f5a623;">1948</p>
              <div class="tl-stem" style="height: 50px; background: #f5a623;"></div>
              <div class="tl-dot" style="background: #f5a623;"></div>
            </div>
            <!-- ~1965: 22.4%, short stem -->
            <div class="fragment tl-node" style="left: 22.4%; bottom: 98px; width: 80px;">
              <p class="tl-label">GOFAI Era</p>
              <p class="tl-year" style="color: #6cb4ff;">~1965</p>
              <div class="tl-stem" style="height: 50px; background: #6cb4ff;"></div>
              <div class="tl-dot" style="background: #6cb4ff;"></div>
            </div>
            <!-- 1986: 50%, tall stem -->
            <div class="fragment tl-node" style="left: 50%; bottom: 98px; width: 80px;">
              <p class="tl-label">Back-<br>propagation</p>
              <p class="tl-year" style="color: #f5a623;">1986</p>
              <div class="tl-stem" style="height: 50px; background: #f5a623;"></div>
              <div class="tl-dot" style="background: #f5a623;"></div>
            </div>
            <!-- 2012: 84.2%, short stem -->
            <div class="fragment tl-node" style="left: 84.2%; bottom: 98px; width: 60px;">
              <p class="tl-label">AlexNet</p>
              <p class="tl-year" style="color: #50c878;">2012</p>
              <div class="tl-stem" style="height: 50px; background: #50c878;"></div>
              <div class="tl-dot" style="background: #50c878;"></div>
            </div>
            <!-- 2017: 90.8%, tall stem (to clear 2018) -->
            <div class="fragment tl-node" style="left: 90.8%; bottom: 98px; width: 70px;">
              <p class="tl-label">Transformers</p>
              <p class="tl-year" style="color: #e06c75;">2017</p>
              <div class="tl-stem" style="height: 80px; background: #e06c75;"></div>
              <div class="tl-dot" style="background: #e06c75;"></div>
            </div>
            <!-- 2018: 92.1%, below the line -->
            <div class="fragment tl-node below" style="left: 92.1%; top: 203px; width: 70px;">
              <div class="tl-dot" style="background: #e06c75;"></div>
              <div class="tl-stem" style="height: 50px; background: #e06c75;"></div>
              <p class="tl-year" style="color: #e06c75;">2018</p>
              <p class="tl-label">BERT / GPT</p>
            </div>
            <!-- 2022: 97.4%, tall stem -->
            <div class="fragment tl-node" style="left: 97.4%; bottom: 98px; width: 60px;">
              <p class="tl-label">ChatGPT</p>
              <p class="tl-year" style="color: #e06c75;">2022</p>
              <div class="tl-stem" style="height: 80px; background: #e06c75;"></div>
              <div class="tl-dot" style="background: #e06c75;"></div>
            </div>
            <!-- 2024: 100%, below the line -->
            <div class="fragment tl-node below" style="right: 0; top: 203px; width: 70px; transform: translateX(50%);">
              <div class="tl-dot" style="background: #c792ea;"></div>
              <div class="tl-stem" style="height: 50px; background: #c792ea;"></div>
              <p class="tl-year" style="color: #c792ea;">2024</p>
              <p class="tl-label">RL Post-<br>Training</p>
            </div>
          </div>
        </div>
      </section>

---

:::figure img="images/shannon.jpg" name="Claude Shannon" kicker="The Father of Information Theory"
- Created the mathematical theory of communication (1948)
- Defined the **"bit"** as a unit of information
- Showed that language has measurable statistical structure
- Proved reliable communication is possible over noisy channels
- His work is the theoretical foundation of every LLM
:::

---

:::figure img="images/vonneumann.gif" name="John von Neumann" kicker="The Architecture of Computation"
- Designed the stored-program architecture (von Neumann architecture)
- Programs and data stored in the same memory &mdash; the basis of all modern computers
- Contributed to game theory, quantum mechanics, and nuclear physics
- Worked on ENIAC and EDVAC at Los Alamos
- Without his architecture, training neural networks would not be possible
:::

---

<!-- .slide: id="info-theory" -->

## Information Theory (1940s-50s)

- **Shannon** proved communication could be quantified in *bits*
- Any signal can be transmitted reliably through a noisy channel
- Gave us a formal way to think about **language as information**
- **Von Neumann's** stored-program architecture made general computation possible
- Together: the theoretical bedrock for everything that follows

> "The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point."
>
> &mdash; Claude Shannon, 1948

---

<!-- .slide: id="gofai" -->

## GOFAI (1950s-80s)

**"Good Old-Fashioned AI"** <!-- .element: class="text-xl" style="color: var(--secondary-color); margin-bottom: 10px;" -->

:::columns cols="2"
### The Approach

- Hand-written rules and symbolic logic
- Expert systems, search trees, knowledge bases
- If-then rules authored by domain experts
<!-- .element: class="text-lg" -->
+++
### The Problem

- Worked for narrow, well-defined tasks
- Couldn't handle ambiguity
- Couldn't scale to real language
- Brittle: one unexpected input breaks everything
<!-- .element: class="text-lg" -->
:::

---

<section id="gofai-diagram">
  <h2>GOFAI vs. Learned Models</h2>
  <div class="content">
    <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: start;">
      <div>
        <h3 style="color: #6cb4ff; text-align: center;">Rule-Based (GOFAI)</h3>
        <div style="background: rgba(74, 158, 255, 0.08); border: 1px solid rgba(74, 158, 255, 0.25); border-radius: 8px; padding: 15px; text-align: center;">
          <p style="font-size: 14pt; color: var(--muted-color);">Input: "I went to the bank"</p>
          <p style="font-size: 20pt; margin: 10px 0;">&#8595;</p>
          <div style="border: 1px solid #6cb4ff; border-radius: 6px; padding: 10px; margin: 8px 0;">
            <p style="font-size: 13pt; font-family: monospace; color: #e8eaf0;">IF "bank" + "river" &#8594; riverbank</p>
            <p style="font-size: 13pt; font-family: monospace; color: #e8eaf0;">IF "bank" + "money" &#8594; financial</p>
          </div>
          <p style="font-size: 20pt; margin: 10px 0;">&#8595;</p>
          <p style="font-size: 14pt; color: #e06c75; font-weight: bold;">??? (no matching rule)</p>
        </div>
      </div>
      <div style="display: flex; align-items: center; padding-top: 60px;">
        <span style="font-size: 14pt; color: var(--muted-color); writing-mode: vertical-lr; border-left: 1px dashed #2a3450; padding-left: 15px; padding-right: 15px; height: 200px;">&nbsp;</span>
      </div>
      <div>
        <h3 style="color: #50c878; text-align: center;">Learned Model</h3>
        <div style="background: rgba(80, 200, 120, 0.08); border: 1px solid rgba(80, 200, 120, 0.25); border-radius: 8px; padding: 15px; text-align: center;">
          <p style="font-size: 14pt; color: var(--muted-color);">Input: "I went to the bank"</p>
          <p style="font-size: 20pt; margin: 10px 0;">&#8595;</p>
          <div style="border: 1px solid #50c878; border-radius: 6px; padding: 10px; margin: 8px 0;">
            <p style="font-size: 14pt; color: #e8eaf0;">Neural Network</p>
            <p style="font-size: 12pt; color: var(--muted-color);">Learns patterns from data</p>
          </div>
          <p style="font-size: 20pt; margin: 10px 0;">&#8595;</p>
          <p style="font-size: 14pt; color: #50c878; font-weight: bold;">financial (95%)</p>
        </div>
      </div>
    </div>
  </div>
</section>

---

<!-- .slide: id="eliza" -->

## <span class="side-quest-badge">Side Quest</span> ELIZA (1966)

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: start;">
<div>

- **Joseph Weizenbaum's** chatbot at MIT
- Simple pattern matching that convinced people it was a **therapist**
- The **"ELIZA effect":** people attribute understanding to systems that have none
- A reminder to be skeptical about AI "understanding"
- [Try ELIZA yourself](https://www.masswerk.at/elizabot/)
<!-- .element: class="text-lg" -->

</div>
<div style="text-align: center;">

![ELIZA conversation](images/eliza.png)

An ELIZA conversation (1966) <!-- .element: class="text-muted" style="font-size: 11pt; margin-top: 8px;" -->

</div>
</div>

---

<section id="summers-winters" style="display: flex !important; flex-direction: column !important; height: 100%; padding: 40px 60px 60px 60px !important;">
        <h2 style="flex-shrink: 0; margin: 0; padding-bottom: 12px; border-bottom: 2px solid var(--line-color);">AI Summers and Winters</h2>
        <p style="font-size: 14pt; color: var(--muted-color); margin-top: 10px; flex-shrink: 0;">Cycles of hype and disappointment. Funding surged when demos impressed, collapsed when promises weren't met.</p>
        <div style="flex: 1; position: relative; min-height: 0; margin-top: 10px;">
          <canvas id="hypeChart"></canvas>
        </div>
      </section>

---

<!-- .slide: id="ai-megaproject" -->

## The Scale of Today's AI Boom

<div style="text-align: center;">

![Data centers vs. historical megaprojects](images/funding.png) <!-- .element: style="max-height: 540px; width: auto; margin: 0 auto;" -->

</div>

Data-center capex now dwarfs the great megaprojects of the 20th century. <!-- .element: class="text-muted" style="font-size: 12pt; text-align: center; margin-top: 8px;" -->

---

<!-- .slide: id="deep-learning" -->

## Deep Learning Revolution

Instead of writing rules, **learn patterns from data**. <!-- .element: class="text-xl" -->

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 15px;">
<div>

### The Breakthrough

- **AlexNet (2012)** &mdash; Krizhevsky, Sutskever, Hinton
- Trained a deep neural network on an **NVIDIA GPU**
- Crushed the ImageNet competition
- Error rate: 26% &#8594; 16% (overnight)
<!-- .element: class="text-lg" -->

</div>
<div>

### Why It Mattered

- Proved GPUs could make deep learning practical
- Neural networks with **many layers** trained on **large datasets**
- Scale was the missing ingredient
<!-- .element: class="text-lg" -->

</div>
</div>

---

:::figure img="images/huang.jpg" name="Jensen Huang" kicker="The Man Behind the GPU Revolution"
- Co-founded NVIDIA in 1993
- CUDA (2006) gave researchers programmable parallel computing
- GPUs became the backbone of deep learning
- NVIDIA is now the most valuable company in the world
<!-- .element: class="text-lg" -->
:::

---

<!-- .slide: id="transformers" class="turning-point" -->

## Transformers (2017)

> "Attention Is All You Need"
>
> &mdash; Vaswani et al., 2017

- Replaced recurrence with **self-attention**
- Enabled **massive parallelism** &mdash; every token attends to every other token simultaneously
- No more sequential bottleneck of RNNs/LSTMs
- The architecture behind **every modern LLM**
<!-- .element: class="text-xl" -->

We'll build one from scratch in Module 4. <!-- .element: class="text-muted text-center" style="margin-top: 20px; font-size: 14pt;" -->

---

<!-- .slide: id="prediction-to-assistants" -->

## From Prediction to Assistants

:::columns cols="2"
### Two Learning Strategies

- **Masked LM (BERT)**<br><span class="text-muted" style="font-size: 14pt;">Predict a hidden word. Devlin et al., 2018.</span>
- **Next Token (GPT)**<br><span class="text-muted" style="font-size: 14pt;">Predict the next word. Radford et al., 2018.</span>
- Sutskever showed next token prediction at scale produces **surprisingly capable** models
+++
### The Assistant Era

- **InstructGPT (2022)**<br><span class="text-muted" style="font-size: 14pt;">Trained GPT to follow instructions using human feedback</span>
- Turned a text predictor into an **assistant**
- **RL post-training**<br><span class="text-muted" style="font-size: 14pt;">RLHF, GRPO, DPO &mdash; aligns models to desired behavior</span>
:::

---

:::divider id="divider-prediction" title="Language as Prediction" sub="The core insight behind every LLM"
:::

---

<!-- .slide: id="shannon-insight" -->

## Shannon's Key Insight (1948)

- **Language is not random.** Given context, the next letter or word is partially predictable.
- Shannon asked humans to guess the next character in a sequence &mdash; they did **far better than chance**
- This is the same idea behind modern LLMs, just at vastly greater scale with learned representations
<!-- .element: class="text-lg" -->

---

<section id="shannon-prediction-demo">
        <h2>Shannon's Prediction Experiment</h2>
        <div class="content">
          <p class="text-lg" style="font-family: monospace; text-align: center; margin-bottom: 15px;">THE NEXT LETTER IS PROBABLY _</p>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
            <div>
              <h3 style="color: #50c878;">With context (English statistics)</h3>
              <table style="width: 100%; font-size: 13pt; margin-top: 10px;">
                <tr><td style="font-family: monospace; padding: 3px 8px;">' '</td><td style="padding: 3px 0;"><div style="background: #50c878; height: 14px; width: 88%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">22%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">E</td><td style="padding: 3px 0;"><div style="background: #50c878; height: 14px; width: 72%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">18%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">A</td><td style="padding: 3px 0;"><div style="background: #50c878; height: 14px; width: 48%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">12%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">T</td><td style="padding: 3px 0;"><div style="background: #50c878; height: 14px; width: 40%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">10%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">S</td><td style="padding: 3px 0;"><div style="background: #50c878; height: 14px; width: 32%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">8%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">other</td><td style="padding: 3px 0;"><div style="background: #50c878; height: 14px; width: 92%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">30%</td></tr>
              </table>
            </div>
            <div>
              <h3 style="color: #e06c75;">Without context (uniform)</h3>
              <table style="width: 100%; font-size: 13pt; margin-top: 10px;">
                <tr><td style="font-family: monospace; padding: 3px 8px;">' '</td><td style="padding: 3px 0;"><div style="background: #e06c75; height: 14px; width: 15%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">3.7%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">E</td><td style="padding: 3px 0;"><div style="background: #e06c75; height: 14px; width: 15%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">3.7%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">A</td><td style="padding: 3px 0;"><div style="background: #e06c75; height: 14px; width: 15%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">3.7%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">T</td><td style="padding: 3px 0;"><div style="background: #e06c75; height: 14px; width: 15%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">3.7%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">S</td><td style="padding: 3px 0;"><div style="background: #e06c75; height: 14px; width: 15%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">3.7%</td></tr>
                <tr><td style="font-family: monospace; padding: 3px 8px;">other</td><td style="padding: 3px 0;"><div style="background: #e06c75; height: 14px; width: 85%; opacity: 0.7;"></div></td><td style="padding: 3px 8px; color: var(--muted-color);">81%</td></tr>
              </table>
            </div>
          </div>
          <p class="text-xl" style="text-align: center; margin-top: 20px; color: #e5c07b; font-weight: 600;">Context makes prediction much easier.</p>
        </div>
      </section>

---

<!-- .slide: id="chain-rule" -->

## The Chain Rule of Probability

How do we compute the probability of an entire sequence $P(w_1, w_2, \ldots, w_n)$? <!-- .element: class="text-lg" -->

$$P(w_1 w_2 \ldots w_n) = \prod_{k=1}^{n} P(w_k \mid w_1 \ldots w_{k-1})$$

This decomposes a joint probability into a product of conditional probabilities. But we still need to estimate $P(w_k \mid w_1 \ldots w_{k-1})$, and we cannot count occurrences of every possible history. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="markov-assumption" -->

## The Markov Assumption

Instead of conditioning on the **entire** history, approximate with just the last few symbols: <!-- .element: class="text-lg" -->

$$P(w_k \mid w_1 \ldots w_{k-1}) \approx P(w_k \mid w_{k-n+1} \ldots w_{k-1})$$

For example, instead of computing: <!-- .element: class="text-lg" -->

$P(\texttt{blue} \mid \texttt{The water of Walden Pond is so beautifully})$

A bigram approximates this as: <!-- .element: class="text-lg" -->

$P(\texttt{blue} \mid \texttt{beautifully})$ <!-- .element: style="color: var(--secondary-color);" -->

This is the core trade-off of n-gram models: more context is better, but the data becomes sparser. <!-- .element: class="text-muted" style="font-size: 13pt;" -->

---

<!-- .slide: id="ngram-models" -->

## N-gram Model Estimation

Given the Markov assumption, we estimate each conditional probability by counting: <!-- .element: class="text-lg" -->

$$P(c_t \mid c_{t-n+1}, \ldots, c_{t-1}) = \frac{\text{count}(c_{t-n+1} \ldots c_t)}{\text{count}(c_{t-n+1} \ldots c_{t-1})}$$

- **Uniform random:** $P(c) = \frac{1}{27}$
- **Unigram:** $P(c) = \frac{\text{count}(c)}{\text{total characters}}$
- **Bigram:** $P(c_t \mid c_{t-1})$
- **Trigram:** $P(c_t \mid c_{t-2}, c_{t-1})$

Shannon described this exact procedure in 1948. Let's build it ourselves. <!-- .element: class="text-muted" style="font-size: 13pt;" -->

---

<section id="ngram-order-demo">
        <h2>N-gram Model Quality</h2>
        <div class="content">
          <div style="margin-bottom: 18px;">
            <p style="font-size: 14pt; color: #e06c75; font-weight: 600; margin-bottom: 4px;">Uniform Random</p>
            <pre style="margin: 0; font-size: 13pt;"><code class="language-text" style="padding: 8px 15px;">xqj bz fmk ort wc aelp gh</code></pre>
          </div>
          <div style="margin-bottom: 18px;">
            <p style="font-size: 14pt; color: #d19a66; font-weight: 600; margin-bottom: 4px;">Unigram</p>
            <pre style="margin: 0; font-size: 13pt;"><code class="language-text" style="padding: 8px 15px;">e tah oin sr dlcu mfpg ywb</code></pre>
          </div>
          <div style="margin-bottom: 18px;">
            <p style="font-size: 14pt; color: #e5c07b; font-weight: 600; margin-bottom: 4px;">Bigram</p>
            <pre style="margin: 0; font-size: 13pt;"><code class="language-text" style="padding: 8px 15px;">th an in he re ou at on er</code></pre>
          </div>
          <div style="margin-bottom: 18px;">
            <p style="font-size: 14pt; color: #50c878; font-weight: 600; margin-bottom: 4px;">Trigram</p>
            <pre style="margin: 0; font-size: 13pt;"><code class="language-text" style="padding: 8px 15px;">the ing and ion tio for ent</code></pre>
          </div>
        </div>
      </section>

---

<!-- .slide: id="entropy" -->

## Shannon Entropy

How many bits do we need, on average, to encode a symbol? <!-- .element: class="text-lg" -->

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

For a **uniform** distribution over 27 characters, every $p(x_i) = \frac{1}{27}$: <!-- .element: class="text-lg" -->

$$H = -\sum_{i=1}^{27} \tfrac{1}{27} \log_2 \tfrac{1}{27} = -27 \cdot \tfrac{1}{27} \cdot \log_2 \tfrac{1}{27} = \log_2(27) = 4.75 \text{ bits/char}$$

- This is the **maximum entropy** for 27 symbols. Any structure in the distribution lowers it.
- English text: $H \approx 1.0\text{--}1.5$ bits/char (Shannon, 1951). Most of each character is **predictable** from context.
- The gap (4.75 vs ~1.2) = the **redundancy** that makes language models possible.
<!-- .element: class="text-lg" -->

---

<!-- .slide: id="cross-entropy" -->

## Cross-Entropy

How well does a **model** $q$ predict text? For each character in the corpus, ask: how many bits does the model need to encode it? <!-- .element: class="text-lg" -->

$$H(q) = -\frac{1}{N} \sum_{i=1}^{N} \log_2 q(c_i \mid \text{context}_i)$$

This is the average number of bits per character the model needs. For each character $c_i$, the model assigns a probability $q(c_i \mid \text{context})$. Higher probability = fewer bits ($-\log_2$ of a large number is small). <!-- .element: class="text-lg" -->

**Perplexity** converts this to an intuitive scale: <!-- .element: class="text-lg" -->

$$\text{Perplexity} = 2^{H(q)}$$

A perplexity of 10 means the model is as uncertain as choosing uniformly from 10 options. Lower = better. <!-- .element: class="text-muted" style="font-size: 13pt;" -->

---

:::video id="compression-demo-1" src="media/compression_uniform.mp4"
:::

---

:::video id="compression-demo-2" src="media/compression_code_table.mp4"
:::

---

:::video id="compression-demo-3" src="media/compression_variable.mp4"
:::

---

:::divider id="divider-setup" title="Getting Set Up" sub="Python environment with uv"
:::

---

<!-- .slide: id="setup-uv" -->

## Setting Up Your Environment

:::columns cols="2"
### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

uv is a fast Python package manager. It replaces pip, venv, and pyenv. <!-- .element: class="text-muted" style="font-size: 13pt; margin-top: 10px;" -->
+++
### 2. Set up the course

```bash
# Clone and sync
git clone <repository-url>
cd LLMs-0-to-100
uv sync

# Verify
uv run python -c "print('Ready.')"
```

One environment for all modules. Run `uv sync` once. <!-- .element: class="text-muted" style="font-size: 13pt; margin-top: 10px;" -->
:::

---

:::divider id="divider-exercise" title="Exercise" sub="N-gram Language Models"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_01_introduction/exercise.py` and fill in the `NotImplementedError` lines. Run after each step &mdash; unfinished functions are skipped automatically. <!-- .element: class="text-lg" -->

```bash
# Run all models (skips any not yet implemented)
cd exercises
uv run python module_01_introduction/src/main.py

# Run a single model
uv run python module_01_introduction/src/main.py --model char3
```

---

<!-- .slide: id="exercise-overview" -->

## Exercise: N-gram Language Models

We just learned that language has statistical structure. Now you will build models that exploit it, recreating Shannon's 1948 experiment. <!-- .element: class="text-lg" -->

Your goal: generate text from **Alice in Wonderland** (~144,000 characters), starting with pure randomness and progressively adding more structure. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

Each function is mostly written for you. You fill in **one key line** per function. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-step1-context" -->

## Step 1: Load the Training Data

Every language model needs a **corpus** &mdash; a body of text to learn from. The file comes from Project Gutenberg, which wraps the book in headers and footers. Strip those out. <!-- .element: class="text-lg" -->

:::note
**Output:** <!-- .element: class="text-lg" -->

```text
Loaded 144603 characters

Alice's Adventures in Wonderland

by Lewis Carroll

CHAPTER I. Down the Rabbit-Hole ...
```
:::

---

:::step id="exercise-step1-code" title="Step 1: load_text()"
```python
    # Find the line numbers where the book starts and ends
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if start_marker in line.upper():
            start_idx = i + 1       # book starts on the NEXT line
        if end_marker in line.upper():
            end_idx = i             # book ends BEFORE this line
            break

    # TODO: Join lines[start_idx:end_idx] with newlines and return the result
    raise NotImplementedError("TODO: return the joined lines")
```
+++
**Hint:** Python's `str.join()` method can combine a list of strings with a separator.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
return "\n".join(lines[start_idx:end_idx])
```
:::

---

:::terminal id="exercise-step1-output" title="Step 1: Output" cmd="uv run python module_01_introduction/src/main.py" caption="The program loads the text, then skips every model you haven't built yet."
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="skipped">=== 0th Order: Uniform Random Characters ===
  [skipped: TODO: return a random string of the given length]

=== 1st Order: Character Unigrams ===
  [skipped: TODO: return a frequency-weighted random string]

=== 2nd Order: Character Bigrams ===
  [skipped: TODO: set context and next_char from text[i:]]

...</span>
:::

---

<!-- .slide: id="exercise-step2-context" -->

## Step 2: The Null Hypothesis

What does text look like with **zero knowledge** of English? Pick each character uniformly at random from a-z and space. This is the baseline &mdash; 0th order. <!-- .element: class="text-lg" -->

:::note
**Output:** <!-- .element: class="text-lg" -->

```text
=== 0th Order: Uniform Random Characters ===
rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg
```
:::

No structure at all. Rare letters like **q**, **x**, **z** appear as often as **e** or **t**. This is maximum entropy. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step2-code" title="Step 2: char_uniform()"
```python
    # The 27 characters we can pick from (26 letters + space)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")

    # TODO: Sample `length` random characters from alphabet and join into a string
    # Use random.choices(population, k=...) to pick, then "".join() to combine
    raise NotImplementedError("TODO: return a random string of the given length")
```
+++
**Hint:** `random.choices(population, k=...)` returns a list of `k` random items. Join them into a single string.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
return "".join(random.choices(alphabet, k=length))
```
:::

---

:::terminal id="exercise-step2-output" title="Step 2: Output" cmd="uv run python module_01_introduction/src/main.py --model uniform" caption="Pure noise. Rare letters appear as often as common ones."
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== 0th Order: Uniform Random Characters ===</span>
rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg
iudmpwmvbyqkflxjiupmlehmjbkzqhsvchnyawijuydkl
:::

---

<!-- .slide: id="exercise-step3-context" -->

## Step 3: Learning Letter Frequencies

Now use the corpus. Count how often each character appears and sample proportionally. This is **1st order** &mdash; we know the frequency of each letter, but nothing about what follows what. <!-- .element: class="text-lg" -->

:::note
**Output:** <!-- .element: class="text-lg" -->

```text
=== 1st Order: Character Unigrams ===
 irt  flniteit et b
b as,allh b e"oeh h  itrltlr
```
:::

Better &mdash; **e**, **t**, and **space** now dominate. But still gibberish because we ignore which letters tend to follow each other. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step3-code" title="Step 3: char_unigram()"
```python
    # Separate the characters and their counts into two parallel lists
    chars = list(counts.keys())       # e.g. ["a", "b", " ", "e", ...]
    weights = list(counts.values())   # e.g. [2, 1, 5, 3, ...]

    # TODO: Sample `length` characters using the weights, join into a string
    # Same as char_uniform but pass weights=weights to random.choices()
    raise NotImplementedError("TODO: return a frequency-weighted random string")
```
+++
**Hint:** Same as Step 2, but `random.choices` accepts a `weights` parameter to bias toward more frequent characters.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
return "".join(random.choices(chars, weights=weights, k=length))
```
:::

---

:::terminal id="exercise-step3-output" title="Step 3: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Spaces and <strong>e</strong> dominate now, but letters still appear in random order."
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== 0th Order: Uniform Random Characters ===</span>
<span class="t-gray">rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg</span>

<span class="header t-orange">=== 1st Order: Character Unigrams ===</span>
 irt  flniteit et b
b as,allh b e"oeh h  itrltlr
:::

---

<!-- .slide: id="exercise-step4-context" -->

## Step 4: Adding Context

This is the key insight. Instead of asking "how common is **e**?", ask "given the previous character was **t**, how common is **e**?" This is a **bigram** model (2nd order). <!-- .element: class="text-lg" -->

:::columns cols="2" gap="20px"
<div class="note">

**Bigram (n=2):** <!-- .element: class="text-lg" -->

```text
_ s icha athap se cker lid
the an n ch f auphtomothe
```

</div>
+++
<div class="note">

**Trigram (n=3):** <!-- .element: class="text-lg" -->

```text
the glar all thed
be falice moce lied alls
```

</div>
:::

With bigrams, common pairs like "th" and "he" emerge. With trigrams, fragments of real words: "the", "alice". <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step4-code" title="Step 4: build_char_ngram_model()"
```python
    # Slide a window of size n across the text
    for i in range(len(text) - n + 1):
        # TODO: Extract the context (first n-1 chars) and next_char (the nth char)
        context = None
        next_char = None
        if context is None or next_char is None:
            raise NotImplementedError("TODO: set context and next_char from text[i:]")

        # Create a Counter for this context if we haven't seen it before
        if context not in model:
            model[context] = Counter()
        # Increment the count for this (context -> next_char) pair
        model[context][next_char] += 1

    return model
```
+++
**Hint:** You have a window of `n` characters at position `i`. The context is the first `n-1`; the next character is the last one.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
context = text[i : i + n - 1]
next_char = text[i + n - 1]
```
:::

---

:::terminal id="exercise-step4-output" title="Step 4: Output" cmd="uv run python module_01_introduction/src/main.py" caption="The model is built, but we can't generate from it yet &mdash; that's step 5."
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== 0th Order: Uniform Random Characters ===</span>
<span class="t-gray">rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg</span>

<span class="header t-orange">=== 1st Order: Character Unigrams ===</span>
<span class="t-gray"> irt  flniteit et b as,allh b e"oeh h  itrltlr</span>

<span class="header t-yellow">=== 2nd Order: Character Bigrams ===</span>
<span class="skipped">[skipped: TODO: set context from the last n-1 chars of result]</span>
:::

---

<!-- .slide: id="exercise-step5-context" -->

## Step 5: Generating from the Model

Now that we have a table of "given context X, character Y appears Z times," we can **generate** text. Start with a seed, look up the context, sample the next character proportionally, repeat. <!-- .element: class="text-lg" -->

This is exactly what Shannon described in 1948 — and it is conceptually the same process used by modern language models, just with far more context and parameters. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step5-code" title="Step 5: generate_from_char_model()"
```python
    while len(result) < length:
        # TODO: Get the current context -- the last (n-1) characters joined together
        # e.g. if n=3 and result ends with ['a','t'], the context should be "at"
        context = None
        if context is None:
            raise NotImplementedError("TODO: set context from the last n-1 chars of result")

        if context in model:
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        else:
            # Context not in model -- fall back to a random context
            context = random.choice(list(model.keys()))
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        result.append(next_char)

    # Join all characters into a single string and return
    return "".join(result[:length])
```
+++
**Hint:** The context is the last `n-1` characters of the result so far. Use negative indexing, then join into a string.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
context = "".join(result[-(n - 1):])
```
:::

---

:::terminal id="exercise-step5-output" title="Step 5: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Build + generate work together. Word fragments emerge: &quot;the&quot;, &quot;alice&quot;, &quot;she&quot;."
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== 0th Order: Uniform Random Characters ===</span>
<span class="t-gray">rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg</span>

<span class="header t-orange">=== 1st Order: Character Unigrams ===</span>
<span class="t-gray"> irt  flniteit et b as,allh b e"oeh h  itrltlr</span>

<span class="header t-yellow">=== 2nd Order: Character Bigrams ===</span>
<span class="t-gray">_ s icha athap se cker lid the an n ch</span>

<span class="header t-green">=== 3rd Order: Character Trigrams ===</span>
the glar all thed be falice moce lied alls
she triede knigh yought alice begal senter
:::

---

<!-- .slide: id="exercise-step6-context" -->

## Step 6: From Characters to Words

The same n-gram idea works at the **word level**. Instead of predicting the next character, predict the next word given the previous words. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="20px"
<div class="note">

**Word unigram:** <!-- .element: class="text-lg" -->

```text
"what the it sitting
history, them," hare. of
take the very said
```

</div>
+++
<div class="note">

**Word trigram:** <!-- .element: class="text-lg" -->

```text
late much accustomed to
usurpation and conquest.
edwin and morcar, the earls
```

</div>
:::

Word trigrams produce surprisingly coherent phrases — sometimes entire sentences lifted from the source. The model is memorizing, not understanding. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step6-code" title="Step 6: build_word_ngram_model()"
Same pattern as the character model, but the context is now a **tuple of words** instead of a string of characters. <!-- .element: class="text-lg" style="margin-bottom: 10px;" -->

```python
for i in range(len(words) - n + 1):
    # TODO: Extract context tuple and next_word
    context = None
    next_word = None
    if context is None or next_word is None:
        raise NotImplementedError("TODO: set context and next_word from words[i:]")

    if context not in model:
        model[context] = Counter()
    model[context][next_word] += 1

return model
```
+++
**Hint:** Use the same slicing pattern as Step 4, but wrap the previous words in `tuple(...)`.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
context = tuple(words[i : i + n - 1])
next_word = words[i + n - 1]
```
:::

---

:::step id="exercise-step6-code-generate" title="Step 6: generate_from_word_model()"
Generation is also the same pattern as Step 5, except the current context is a tuple of the last `n-1` words. <!-- .element: class="text-lg" style="margin-bottom: 8px;" -->

```python
while len(result) < length:
    # TODO: Get the current context as a tuple of the last (n-1) words
    # Same as the char version but use tuple() instead of "".join()
    context = None
    if context is None:
        raise NotImplementedError("TODO: set context from the last n-1 words of result")

    if context in model:
        counter = model[context]
        words = list(counter.keys())
        weights = list(counter.values())
        next_word = random.choices(words, weights=weights, k=1)[0]
    else:
        context = random.choice(list(model.keys()))
        counter = model[context]
        words = list(counter.keys())
        weights = list(counter.values())
        next_word = random.choices(words, weights=weights, k=1)[0]
    result.append(next_word)

return " ".join(result[:length])
```
+++
**Hint:** Same as the character version, but use `tuple()` instead of `"".join()`.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
context = tuple(result[-(n - 1):])
```
:::

---

:::terminal id="exercise-step6-output" title="Step 6: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Coherent phrases, sometimes entire sentences lifted from Alice in Wonderland."
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== 0th Order: Uniform Random Characters ===</span>
<span class="t-gray">rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg</span>
<span class="header t-orange">=== 1st Order: Character Unigrams ===</span>
<span class="t-gray"> irt  flniteit et b as,allh b e"oeh h  itrltlr</span>
<span class="header t-yellow">=== 2nd Order: Character Bigrams ===</span>
<span class="t-gray">_ s icha athap se cker lid the an n ch</span>
<span class="header t-green">=== 3rd Order: Character Trigrams ===</span>
<span class="t-gray">the glar all thed be falice moce lied alls</span>
<span class="header t-cyan">=== Word Unigrams ===</span>
<span class="t-gray">"what the it sitting history, them," hare.</span>

<span class="header t-blue">=== Word Trigrams ===</span>
late much accustomed to usurpation and conquest.
edwin and morcar, the earls of mercia and
:::

---

:::terminal id="exercise-together" title="All Models: More Context, Better Output" cmd="uv run python module_01_introduction/src/main.py" maxw="900px"
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== 0th Order: Uniform Random Characters ===
<span class="t-fg">rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg</span></span>
<span class="header t-orange">=== 1st Order: Character Unigrams ===
<span class="t-fg"> irt  flniteit et b as,allh b e"oeh h  itrltlr</span></span>
<span class="header t-yellow">=== 2nd Order: Character Bigrams ===
<span class="t-fg">_ s icha athap se cker lid the an n ch</span></span>
<span class="header t-green">=== 3rd Order: Character Trigrams ===
<span class="t-fg">the glar all thed be falice moce lied alls</span></span>
<span class="header t-cyan">=== Word Unigrams ===
<span class="t-fg">"what the it sitting history, them," hare.</span></span>
<span class="header t-blue">=== Word Trigrams ===
<span class="t-fg">late much accustomed to usurpation and conquest.</span></span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

If you finish early, try the optional extra credit: <!-- .element: class="text-lg" -->

- **`cross_entropy()`** &mdash; measure how surprised the model is: $H(p, q) = -\frac{1}{N} \sum \log_2 q(c_i \mid \text{context})$
- **`perplexity()`** &mdash; convert to perplexity: $\text{PPL} = 2^{H(p,q)}$, a standard metric for language models

Lower perplexity = better model. You should see perplexity drop as you increase the n-gram order. This is the same metric used to evaluate GPT and other modern LLMs. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::divider id="quiz-divider" title="Quiz" sub="Test your understanding"
:::

---

:::quiz id="quiz-q1" title="Q1: Why Not Just Use Rules?"
GOFAI systems used hand-written rules to process language. Why did this approach ultimately fail for general language understanding? <!-- .element: class="text-lg" -->
+++
**Answer:** Language is inherently ambiguous and context-dependent. You cannot enumerate rules for every possible meaning of every word in every context. "Bank" alone has dozens of meanings, and the rules would need to cover all of them and all their interactions. Statistical/learned models discover these patterns from data instead of requiring manual specification. <!-- .element: class="text-lg" -->
:::

---

:::quiz id="quiz-q2" title="Q2: Shannon's Entropy"
Shannon estimated English has $H \approx 1.0\text{--}1.5$ bits/char. A uniform 27-character alphabet has $H = \log_2(27) = 4.75$ bits/char. <!-- .element: class="text-lg" -->

What does this gap tell us about English as a communication system? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** English is highly redundant. About 70-80% of each character is predictable from context. This redundancy is what makes autocomplete, spell-check, and language models possible, and it is why you can read text with missing letters. Shannon showed this redundancy is a measurable, mathematical property. <!-- .element: class="text-lg" -->
:::

---

:::quiz id="quiz-q3" title="Q3: The Memorization Problem"
A word-level 5-gram model trained on *Alice in Wonderland* produces output that is mostly exact quotes from the book. <!-- .element: class="text-lg" -->

Why does this happen, and how do modern LLMs avoid it? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** With high-order n-grams and limited training data, most contexts appear only once, so the model can only reproduce the original. Modern LLMs avoid this by (1) training on vastly more data, (2) using continuous representations instead of exact count tables, and (3) generalizing across similar contexts via learned embeddings. <!-- .element: class="text-lg" -->
:::

---

:::quiz id="quiz-q4" title="Q4: Conditional Probability"
A trigram model trained on English text has these counts for the context "th": <!-- .element: class="text-lg" -->

```python
model["th"] = Counter({"e": 50, "a": 20, "i": 15, "o": 10, "u": 5})
```

What is $P(\texttt{e} \mid \texttt{th})$, and what does it mean for the model's behavior? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** $P(\texttt{e} \mid \texttt{th}) = \frac{50}{50+20+15+10+5} = \frac{50}{100} = 0.5$. Half the time the model encounters the context "th", it will generate "e" as the next character. This is a conditional probability: it only applies when "th" is already the context. The other half of the time, the model generates "a" (20%), "i" (15%), "o" (10%), or "u" (5%). This is why trigram output produces "the" so frequently, but also generates "tha", "thi", and other combinations. <!-- .element: class="text-lg" -->
:::

---

:::quiz id="quiz-q5" title="Q5: Compression and Prediction"
Shannon proved that a perfect language model would also be a perfect compressor, and vice versa. <!-- .element: class="text-lg" -->

Why are prediction and compression fundamentally the same problem? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** If you can predict the next symbol with high confidence, you need fewer bits to encode it: a symbol with probability $p$ requires $-\log_2(p)$ bits. A model that predicts perfectly ($p = 1$) needs 0 bits. Conversely, any compression scheme implicitly defines a probability model over the next symbol. Shannon's source coding theorem formalizes this: the minimum average code length equals the entropy $H(X)$. <!-- .element: class="text-lg" -->
:::

---

<!-- .slide: id="resources" -->

## References and Further Reading

- Shannon, C. E. (1948). [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- Shannon, C. E. (1951). "Prediction and Entropy of Printed English." *Bell System Technical Journal*, 30(1), 50&ndash;64.
- Von Neumann, J. (1945). "First Draft of a Report on the EDVAC." Moore School of Electrical Engineering.
- Jurafsky & Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), Ch. 3 (N-gram Language Models)
- Gleick, J. (2011). *The Information: A History, a Theory, a Flood*
- Weizenbaum, J. (1966). "ELIZA." *Communications of the ACM*, 9(1), 36&ndash;45. [Try it yourself](https://www.masswerk.at/elizabot/)
- Haugeland, J. (1985). *Artificial Intelligence: The Very Idea*. MIT Press.
<!-- .element: class="text-lg" -->

---

:::divider id="end" title="Questions?" sub="Next: Module 2 &mdash; Perceptrons and Optimization"
:::
