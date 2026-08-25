<!-- .slide: id="judge-no-key" -->

## When There Is No Right Answer

Summaries, explanations, advice, code review, tone: no key to compare against. The field has tried two answers.

:::columns cols="2" gap="34px"
**Old: overlap with a reference**

- **BLEU** (2002, translation), **ROUGE** (2004, summarization)
- Count n-grams shared with human-written references
- Drove twenty years of measurable progress
- Cannot tell whether an answer is **true or useful**: a fluent wrong answer scores well
+++
**Current: comparison**

- Show two answers to the same prompt, ask **which is better**
- No reference text, no vocabulary matching
- Measures what users care about
- Cost: you need a judge, and judges have biases
:::

---

:::figure img="images/papineni_lin.jpg" name="Kishore Papineni &amp; Chin-Yew Lin" kicker="BLEU (IBM, 2002) and ROUGE (2004)" alt="Kishore Papineni and Chin-Yew Lin"
Papineni and collaborators scored translations by **n-gram overlap with human references**. Before BLEU, comparing two translation systems meant hiring translators. After it, a research group could iterate overnight.

Lin did the same for summarization. Both metrics correlate **weakly** with human judgment on modern systems and are still reported anyway: a flawed number everyone computes the same way is at least comparable.
:::

---

<!-- .slide: id="judge-arena" -->

## Human Preference at Scale: Chatbot Arena

Anonymous side-by-side votes from **real users on real prompts**, aggregated into an Elo-style rating. The industry's de facto public scoreboard.

<div class="card-grid cols-2">
<div class="card"><h4>Why it worked</h4><p>Measures what people <strong>actually prefer</strong>, on prompts people actually send, not an academic test set.</p></div>
<div class="card warn"><h4>The caveat</h4><p><strong>Preference is not correctness.</strong> Longer, better-formatted, more agreeable answers win votes: Module 7's length and sycophancy bias, now measured instead of trained.</p></div>
</div>

A model can climb the Arena by getting friendlier and more verbose, not more accurate. <!-- .element: class="text-lg" -->

---

:::figure img="images/chiang_zheng.jpg" name="Wei-Lin Chiang &amp; Lianmin Zheng" kicker="Chatbot Arena and MT-Bench (UC Berkeley and LMSYS, 2023)" alt="Wei-Lin Chiang and Lianmin Zheng"
With **Ying Sheng** and collaborators, they turned **anonymous preference votes** into a continuously updated public ranking, and published the paired-comparison data.

Their **MT-Bench** paper is the standard reference for LLM-as-a-judge: a strong model's verdicts agree with humans enough to be useful, and are biased enough to need controls.

Both rest on one shift: with no answer key, you can still ask **which of two answers is better.**
:::

---

<!-- .slide: id="judge-llm" -->

## LLM-as-a-Judge

A strong model grades answers against a rubric, or picks a winner between two. Correlates well with human preference, costs almost nothing, now everywhere.

<div class="card-grid cols-2">
<div class="card"><h4>MT-Bench (2023)</h4><p>Eighty multi-turn questions, scored 1&ndash;10 by a strong judge model against a rubric.</p></div>
<div class="card"><h4>AlpacaEval (2023)</h4><p>Head-to-head win rate against a fixed reference model, judged automatically. Version 2 <strong>controls for length</strong> explicitly.</p></div>
</div>

A judge is a **proxy**. Validate it against a few hundred human-labeled examples **from your own task** before trusting its numbers. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="judge-biases" -->

## Judge Biases You Should Assume Are Present

<div class="card-grid cols-4">
<div class="card warn"><h4>Position bias</h4><p>Prefers whichever answer came first. <strong>Fix:</strong> run both orders and average.</p></div>
<div class="card warn"><h4>Verbosity bias</h4><p>Prefers longer answers. <strong>Fix:</strong> control for length, or report length alongside the win rate.</p></div>
<div class="card warn"><h4>Self-preference</h4><p>Favors outputs that look like its own. <strong>Fix:</strong> use a judge from a different family.</p></div>
<div class="card warn"><h4>Competence ceiling</h4><p>A judge cannot reliably grade work it could not do itself. <strong>Fix:</strong> no fix. Use humans.</p></div>
</div>

---

<!-- .slide: id="sq-swap-answers" -->

## Side Quest: Swap the Answers

Ask a judge model which of two answers is better. Record the verdict. Present **the same answers in the opposite order** and ask again.

<div class="card-grid cols-2">
<div class="card"><h4>What often happens</h4><p>The verdict flips. The judge preferred <strong>the first position</strong>, not the better answer.</p></div>
<div class="card"><h4>What it costs to fix</h4><p>Double the judge calls. Every serious LLM-judge harness randomizes or averages over order for exactly this reason.</p></div>
</div>

A two-minute experiment that changes how you read every LLM-judged leaderboard. The extra credit builds a rule-based version you can watch flip. <!-- .element: class="text-lg" -->
