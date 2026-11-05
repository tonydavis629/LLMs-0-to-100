<!-- .slide: id="judge-no-key" -->

## When There Is No Right Answer

Summaries, explanations, advice, code review, tone: no key to compare against. The field has tried two answers.

:::columns cols="2" gap="34px"
**Old: overlap with a reference**

- **BLEU** (2002, translation), **ROUGE** (2004, summarization)
- Count how many short word sequences (**n-grams**) the output shares with a reference a person wrote
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
Papineni and collaborators scored a translation by counting how many of its word sequences also appear in **human reference translations**. Before BLEU, comparing two translation systems meant hiring translators. After it, a research group could iterate overnight.

Lin turned the count around for summarization: ROUGE asks how much of the **reference summary** the candidate recovers. Both metrics correlate **weakly** with human judgment on modern systems and are still reported anyway: a flawed number everyone computes the same way is at least comparable.
:::

---

<!-- .slide: id="judge-rouge" -->

## How ROUGE Scores a Summary

$$\text{ROUGE-}N = \frac{\text{n-grams shared with the reference}}{\text{n-grams in the reference}}$$

An n-gram is a run of n consecutive words: ROUGE-1 counts single words, ROUGE-2 counts word pairs. This is the recall from the token F1 slide, extended to word sequences. BLEU runs the same count from the other side: it divides by the candidate's n-grams (precision) and penalizes outputs that are too short.

Reference: **"the cat sat on the mat"**

<div class="compare-table">
<table>
<thead><tr><th>Candidate</th><th>ROUGE-1</th><th>ROUGE-2</th><th>Same meaning?</th></tr></thead>
<tbody>
<tr><td>the cat was on the mat</td><td class="num">0.83</td><td class="num">0.60</td><td>Yes</td></tr>
<tr><td>the cat never sat on the mat</td><td class="num">1.00</td><td class="num">0.80</td><td>No, it says the opposite</td></tr>
<tr><td>a kitten rested on the rug</td><td class="num">0.33</td><td class="num">0.20</td><td>Yes</td></tr>
</tbody>
</table>
</div>

The contradiction gets the best score because it contains every word of the reference. Overlap measures shared vocabulary, and meaning is invisible to it. <!-- .element: class="text-lg" -->

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

<!-- .slide: id="sq-arena-vote" -->

## <span class="side-quest-badge">Side Quest</span> Cast a Vote in the Arena

Chatbot Arena now runs as **Arena** at <a href="https://arena.ai/" target="_blank">arena.ai</a>. Open it, type a prompt, and vote for one of two anonymous answers. The model names appear only after you vote.

<div class="card-grid cols-2">
<div class="card"><h4>Try a question you can check</h4><p>Pick something where you know the right answer: a calculation, a fact from your field, a bug in a short function. Did you vote for the <strong>correct</strong> answer or the more convincing one?</p></div>
<div class="card"><h4>Then try an open-ended one</h4><p>Ask for advice or an explanation. Notice what decided your vote: length, formatting, tone, or content. Your vote goes into the same rating as everyone else's.</p></div>
</div>

Every number on the leaderboard is built from votes like yours. <!-- .element: class="text-lg" -->

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

## <span class="side-quest-badge">Side Quest</span> Swap the Answers

Ask a judge model which of two answers is better. Record the verdict. Present **the same answers in the opposite order** and ask again.

<div class="card-grid cols-2">
<div class="card"><h4>What often happens</h4><p>The verdict flips. The judge preferred <strong>the first position</strong>, not the better answer.</p></div>
<div class="card"><h4>What it costs to fix</h4><p>Double the judge calls. Every serious LLM-judge harness randomizes or averages over order for exactly this reason.</p></div>
</div>

A two-minute experiment that changes how you read every LLM-judged leaderboard. The extra credit builds a rule-based version you can watch flip. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-decision-judge" -->

## <span class="side-quest-badge">Side Quest</span> A Judge That Writes No Text

A **decision model** such as TypeSafe's Jev is closer to a text classifier than a chatbot. It answers typed questions (yes or no, pick one, a score) and returns a **probability** for each answer. <a href="https://github.com/osolmaz/switch" target="_blank">switch</a> is an open-source harness that runs one as a rubric judge.

<div class="card-grid cols-3">
<div class="card"><h4>Nothing to parse</h4><p>A chat judge writes a verdict that you then extract with a regex. A decision model returns P(yes) for each rubric item directly, so there is no answer extraction to get wrong.</p></div>
<div class="card"><h4>Two questions per item</h4><p>Does the answer meet the item (<em>a</em>)? Does it miss or contradict a requirement (<em>b</em>)? The item passes when min(<em>a</em>, 1 &minus; <em>b</em>) &ge; 0.5.</p></div>
<div class="card"><h4>Cheap</h4><p>It generates no tokens, and it picks what to read (a section of a file, a page image, a search hit) instead of loading every document into context.</p></div>
</div>

Treat it like any other judge: score it against human labels before you trust it. switch's rubric judge ships a <code>score</code> command for that. <!-- .element: class="text-lg" -->
