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
