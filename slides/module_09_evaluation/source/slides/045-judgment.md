<!-- .slide: id="judge-no-key" -->

## When There Is No Right Answer

Summaries, explanations, advice: there is no key to compare against.

:::columns cols="2" gap="34px"
**Old: overlap with a reference**

- **BLEU** (2002), **ROUGE** (2004)
- Count the word sequences (**n-grams**) shared with a human-written reference
- A fluent wrong answer scores well
+++
**Current: comparison**

- Show two answers, ask **which is better**
- No reference needed
- Needs a judge, and judges have biases
:::

---

:::figure img="images/papineni_lin.jpg" name="Kishore Papineni &amp; Chin-Yew Lin" kicker="BLEU (IBM, 2002) and ROUGE (2004)" alt="Kishore Papineni and Chin-Yew Lin"
Papineni built **BLEU**: score a translation by its word overlap with human references. Lin built **ROUGE** for summaries.

Both track human judgment weakly, and both are still reported because everyone computes them the same way.
:::

---

<!-- .slide: id="judge-rouge" -->

## How ROUGE Scores a Summary

$$\text{ROUGE-}N = \frac{\text{n-grams shared with the reference}}{\text{n-grams in the reference}}$$

ROUGE-1 counts single words, ROUGE-2 counts word pairs. BLEU divides by the candidate's n-grams instead.

Reference: **"the cat sat on the mat"**

<div class="compare-table">
<table>
<thead><tr><th>Candidate</th><th>ROUGE-1</th><th>ROUGE-2</th><th>Same meaning?</th></tr></thead>
<tbody>
<tr><td>the cat was on the mat</td><td class="num">0.83</td><td class="num">0.60</td><td>Yes</td></tr>
<tr><td>the cat never sat on the mat</td><td class="num">1.00</td><td class="num">0.80</td><td>No, the opposite</td></tr>
<tr><td>a kitten rested on the rug</td><td class="num">0.33</td><td class="num">0.20</td><td>Yes</td></tr>
</tbody>
</table>
</div>

The contradiction scores best because it uses every reference word. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="judge-arena" -->

## Human Preference at Scale

**Chatbot Arena**: anonymous side-by-side votes from real users on real prompts, turned into an Elo-style rating.

<div class="card-grid cols-2">
<div class="card"><h4>Why it worked</h4><p>Measures what people <strong>actually prefer</strong> on prompts they actually send.</p></div>
<div class="card warn"><h4>The caveat</h4><p><strong>Preference is not correctness.</strong> Longer, friendlier answers win votes.</p></div>
</div>

---

:::figure img="images/chiang_zheng.jpg" name="Wei-Lin Chiang &amp; Lianmin Zheng" kicker="Chatbot Arena and MT-Bench (UC Berkeley and LMSYS, 2023)" alt="Wei-Lin Chiang and Lianmin Zheng"
Built **Chatbot Arena**, a public ranking from anonymous votes, with Ying Sheng and collaborators.

Their **MT-Bench** paper is the standard reference for using an LLM as a judge.
:::

---

<!-- .slide: id="sq-arena-vote" -->

## <span class="side-quest-badge">Side Quest</span> Cast a Vote in the Arena

Chatbot Arena now runs as **Arena** at <a href="https://arena.ai/" target="_blank">arena.ai</a>. Type a prompt, vote for one of two anonymous answers, then see which models they were.

<div class="card-grid cols-2">
<div class="card"><h4>Ask something you can check</h4><p>Did you vote for the <strong>correct</strong> answer or the more convincing one?</p></div>
<div class="card"><h4>Ask something open-ended</h4><p>What decided your vote: length, formatting, tone, or content?</p></div>
</div>

---

<!-- .slide: id="judge-llm" -->

## LLM-as-a-Judge

A strong model grades answers against a rubric, or picks the better of two. Cheap, and now everywhere.

<div class="card-grid cols-2">
<div class="card"><h4>MT-Bench (2023)</h4><p>80 multi-turn questions, each scored 1&ndash;10 by a judge model.</p></div>
<div class="card"><h4>AlpacaEval (2023)</h4><p>Win rate against a fixed reference model. Version 2 <strong>controls for length</strong>.</p></div>
</div>

Check a judge against a few hundred human labels **from your own task** before trusting it. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="judge-biases" -->

## Judge Biases You Should Assume Are Present

<div class="card-grid cols-4">
<div class="card warn"><h4>Position</h4><p>Prefers the first answer. <strong>Fix:</strong> run both orders.</p></div>
<div class="card warn"><h4>Verbosity</h4><p>Prefers longer answers. <strong>Fix:</strong> control for length.</p></div>
<div class="card warn"><h4>Self-preference</h4><p>Prefers its own style. <strong>Fix:</strong> judge from another family.</p></div>
<div class="card warn"><h4>Competence</h4><p>Cannot grade what it cannot do. <strong>Fix:</strong> use humans.</p></div>
</div>

---

<!-- .slide: id="sq-swap-answers" -->

## <span class="side-quest-badge">Side Quest</span> Swap the Answers

Ask a judge model which of two answers is better. Then ask again with **the order reversed**.

<div class="card-grid cols-2">
<div class="card"><h4>What often happens</h4><p>The verdict flips. The judge preferred <strong>the first position</strong>.</p></div>
<div class="card"><h4>The fix</h4><p>Run both orders, at twice the cost.</p></div>
</div>

The extra credit builds a rule-based judge you can watch flip. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-decision-judge" -->

## <span class="side-quest-badge">Side Quest</span> A Judge That Writes No Text

A **decision model** such as TypeSafe's Jev works like a text classifier: it answers yes/no or pick-one questions with a **probability**. <a href="https://github.com/osolmaz/switch" target="_blank">switch</a> runs one as a rubric judge.

<div class="card-grid cols-3">
<div class="card"><h4>Nothing to parse</h4><p>It returns P(yes) per rubric item. No answer extraction.</p></div>
<div class="card"><h4>Two questions per item</h4><p>Meets it (<em>a</em>)? Misses something (<em>b</em>)? Pass when min(<em>a</em>, 1 &minus; <em>b</em>) &ge; 0.5.</p></div>
<div class="card"><h4>Cheap</h4><p>Generates no tokens and reads only what it picks.</p></div>
</div>

Like any judge, check it against human labels first. <!-- .element: class="text-lg" -->
