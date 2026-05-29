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

<!-- .slide: id="compression-demo-1" -->

<div class="video-container">
  <img src="media/compression_uniform.png" alt='Uniform encoding of "the cat sat on the mat": 22 chars x 5 bits = 110 bits total'>
</div>

---

<!-- .slide: id="compression-demo-2" -->

<div class="video-container">
  <img src="media/compression_code_table.png" alt="Prefix code table: common characters (space, t) get 2 bits, rare characters (c, m) get 7 bits">
</div>

---

<!-- .slide: id="compression-demo-3" -->

<div class="video-container">
  <img src="media/compression_variable.png" alt="Variable-length encoding: 70 bits, 36% smaller than uniform 110 bits">
</div>
