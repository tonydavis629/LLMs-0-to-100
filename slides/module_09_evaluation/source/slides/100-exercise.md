:::divider id="divider-exercise" title="Exercise" sub="Build a small benchmark suite and score two models"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Fill in the eight `NotImplementedError` lines in `module_09_evaluation/exercise.py`. Everything else (checkpoints, tokenizer, sampler, data, plotting, runner) is provided. Run after each step; unfinished steps are skipped. <!-- .element: class="text-lg" -->

```bash
# Score the Module 6 instruct model against the Module 7 GRPO model
cd exercises
uv run python module_09_evaluation/src/main.py
```

Both checkpoints ship with the repo, trained for this module by `solution/src/make_checkpoints.py` from the Module 5 base model. <!-- .element: class="text-md" style="margin-top: 22px;" -->

The runner prints the **protocol**, then perplexity, per-task tables (exact match, F1, pass@1, pass@5), multiple choice, and the suite score. It saves a bar chart to `output/task_comparison.png`. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Nothing Here Trains a Model

Both checkpoints are finished:

- `instruct_model.pt`: the Module 6 model, finetuned on four toy tasks
- `rl_model.pt`: the same model after Module 7's GRPO run on **reverse only**

:::columns cols="2" gap="30px"
**You write the metrics**

Perplexity, normalization, exact match, token F1, likelihood-scored multiple choice, pass@k, per-task accuracy, suite average. Each is one line.
+++
**The payoff is the table**

- RL model wins its trained task by 17 points
- Loses 50 to 75 points on the other three
- Perplexity calls it the **better** model
- Multiple choice cannot see the problem at all
:::

---

<!-- .slide: id="exercise-suite" -->

## The Suite

<div class="bench-table">
<table>
<thead><tr><th>File</th><th>Contents</th><th>Scored by</th></tr></thead>
<tbody>
<tr><td><code>held_out.txt</code></td><td>3,916 characters of text from the Module 5 corpus, never trained on</td><td>Perplexity (no labels, no generation)</td></tr>
<tr><td><code>tasks.jsonl</code></td><td>50 held-out cases across <code>uppercase</code>, <code>repeat</code>, <code>reverse</code>, and <code>qa</code></td><td>Exact match, token F1, pass@k on generated answers</td></tr>
<tr><td><code>multiple_choice.jsonl</code></td><td>16 four-option questions over the same material</td><td>Likelihood (nothing is generated)</td></tr>
</tbody>
</table>
</div>

One built-in trap: `uppercase`, `repeat`, and `reverse` use words that appear nowhere in finetuning. All eight `qa` facts were memorized verbatim. That task is **contaminated by construction**. <!-- .element: class="text-lg" -->

---

:::step id="exercise-step1" title="Step 1: perplexity()"
```python
def perplexity(mean_token_loss: float) -> float:
    """Convert an average per-token cross-entropy loss (in nats) into perplexity."""
    # TODO: Return the perplexity that corresponds to this average loss.
    raise NotImplementedError("TODO: convert average token loss into perplexity")
```
+++
**Hint:** perplexity is the exponential of the mean loss; `math.exp` does this.
+++
**Answer:**

```python
return math.exp(mean_token_loss)
```
:::

---

:::terminal id="exercise-output-1" title="After Step 1: The Protocol, Then One Number" cmd="uv run python module_09_evaluation/src/main.py" caption="The protocol is printed before any score. Note the result: the RL model has LOWER perplexity, which is about to turn out to mean nothing."
<span class="header">MODULE 9: evaluating two finished checkpoints</span>
Models under test
  instruct  data/instruct_model.pt   Module 6: multi-task SFT
  rl        data/rl_model.pt         Module 7: GRPO on `reverse` only

Protocol
  chat template     &lt;|user|&gt; PROMPT &lt;|end|&gt; &lt;|assistant|&gt; ANSWER &lt;|end|&gt;
  normalization     lowercase, strip punctuation, collapse whitespace
  generation budget 14 tokens    context 128
  decoding          greedy for exact match and F1; 5 samples at temperature 0.8
  seed              1337 (per case, so both models see the same draws)
  suite             50 generated cases across 4 tasks, 16 multiple-choice questions
  held-out text     3,916 characters, never seen in training

<span class="header">1. PERPLEXITY ON HELD-OUT TEXT</span>
    model          loss (nats)    perplexity
    instruct            7.5941       1986.48
    <span class="success">rl                  7.3773       1599.26</span>

<span class="skipped">2. TASK SUITE  [skipped: implement normalize_answer() and exact_match()]</span>
<span class="skipped">3. MULTIPLE CHOICE  [skipped: implement score_multiple_choice()]</span>
<span class="skipped">4. SUITE SCORE  [skipped: implement task_accuracy() and suite_score()]</span>
:::

---

:::step id="exercise-step2" title="Step 2: normalize_answer()"
```python
def normalize_answer(text: str) -> str:
    """Put a generated answer into the canonical form that scoring compares."""
    lowered = text.lower()
    stripped = "".join(ch for ch in lowered if ch not in string.punctuation)
    # TODO: Return `stripped` with leading and trailing whitespace removed and every
    #       run of internal whitespace collapsed to a single space.
    raise NotImplementedError("TODO: collapse the whitespace in the normalized answer")
```
+++
**Hint:** `.split()` with no argument splits on any run of whitespace and drops the empties; `" ".join(...)` puts the pieces back together.
+++
**Answer:**

```python
return " ".join(stripped.split())
```
:::

---

:::step id="exercise-step3" title="Step 3: exact_match()"
```python
def exact_match(prediction: str, answers: list[str]) -> float:
    """Score 1.0 if the normalized prediction equals any acceptable answer, else 0.0."""
    # TODO: Return 1.0 when the normalized prediction matches any normalized
    #       acceptable answer, otherwise 0.0.
    raise NotImplementedError("TODO: score the prediction against the acceptable answers")
```
+++
**Hint:** normalize both sides; `any(...)` over a generator, wrapped in `float()`.
+++
**Answer:**

```python
normalized = normalize_answer(prediction)
return float(any(normalized == normalize_answer(answer) for answer in answers))
```
:::

---

:::step id="exercise-step4" title="Step 4: token_f1()"
```python
def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and one reference answer."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    shared = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    precision = shared / len(predicted_tokens)
    recall = shared / len(reference_tokens)
    # TODO: Return the F1 score: the harmonic mean of precision and recall.
    raise NotImplementedError("TODO: combine precision and recall into F1")
```
+++
**Hint:** `2 * precision * recall`, divided by their sum.
+++
**Answer:**

```python
return 2 * precision * recall / (precision + recall)
```
:::

---

:::step id="exercise-step5" title="Step 5: task_accuracy()"
```python
def task_accuracy(scores_by_task: dict[str, list[float]]) -> dict[str, float]:
    """Average each task's per-case scores into one number per task."""
    # TODO: Return a dict mapping each task to the mean of its scores.
    raise NotImplementedError("TODO: average the per-case scores within each task")
```
+++
**Hint:** a dict comprehension over `.items()`; `sum(scores) / len(scores)`.
+++
**Answer:**

```python
return {task: sum(scores) / len(scores) for task, scores in scores_by_task.items()}
```
:::

---

:::step id="exercise-step6" title="Step 6: suite_score()"
```python
def suite_score(per_task: dict[str, float]) -> float:
    """Average the per-task scores into the single headline number."""
    # TODO: Return the mean of the per-task scores.
    raise NotImplementedError("TODO: average the per-task scores into the suite score")
```
+++
**Hint:** `per_task.values()` gives the scores; `sum(...) / len(...)`.
+++
**Answer:**

```python
return sum(per_task.values()) / len(per_task)
```
:::

---

:::terminal id="exercise-output-2" title="After Step 6: The Report Appears" cmd="uv run python module_09_evaluation/src/main.py" caption="Actual output. Steps 2-4 produce the per-case scores, step 5 groups them by task, step 6 averages them. Read the headline number last."
<span class="header">2. TASK SUITE</span>
  Generating 50 greedy + 250 sampled answers per model...

  EXACT MATCH (greedy decoding)
    task          cases    instruct        rl      diff
    uppercase        10       80.0%     30.0%    <span class="t-fail">-50.0%</span>
    repeat            8      100.0%     25.0%    <span class="t-fail">-75.0%</span>
    reverse          24       75.0%     91.7%    <span class="success">+16.7%</span>
    qa                8      100.0%     75.0%    <span class="t-fail">-25.0%</span>

  TOKEN F1 (greedy decoding)
    task          cases    instruct        rl      diff
    uppercase        10       80.0%     30.0%    <span class="t-fail">-50.0%</span>
    repeat            8      100.0%     25.0%    <span class="t-fail">-75.0%</span>
    reverse          24       75.0%     91.7%    <span class="success">+16.7%</span>
    qa                8      100.0%     83.3%    <span class="t-fail">-16.7%</span>

<span class="header">4. SUITE SCORE</span>
    instruct       88.8%   (mean of the four task scores)
    rl             55.4%   (mean of the four task scores)
  Overall difference: <span class="t-fail">-33.3%</span>
  Read the per-task table above before believing that number.
:::

---

:::terminal id="exercise-output-cases" title="The Same Run, One Case Per Task" cmd="uv run python module_09_evaluation/src/main.py" caption="Actual output. Keeping the per-case generations, not just the averages, is where a number turns into a diagnosis: the RL model transposes characters, runs past its answer, and on the last case produces nothing resembling English."
  Sample cases (greedy), one per task:
    [uppercase] 'uppercase: metric'   want 'METRIC'
        instruct 'METRIC'
        rl       'METRCI'
    [repeat]    'repeat: task'        want 'task'
        instruct 'task'
        rl       'taskto'
    [reverse]   'reverse: bamkf'      want 'fkmab'
        instruct 'fkmaq'
        rl       'fkmab'
    [qa]        'opposite of up?'     want 'it is down'
        instruct 'it is down'
        rl       'utripso&lt;|user|&gt;&lt;|user|...'
:::

---

:::step id="exercise-step7" title="Step 7: score_multiple_choice()"
```python
def score_multiple_choice(option_log_probs: list[float], option_lengths: list[int]) -> int:
    """Pick the option the model finds most likely, per token."""
    # TODO: Return the index of the option with the highest AVERAGE log-probability
    #       per token (total log-probability divided by number of tokens).
    raise NotImplementedError("TODO: choose the option with the best per-token log-probability")
```
+++
**Hint:** `range(len(option_log_probs))` gives the indices; `max(..., key=...)` picks the best one, and the key is a lambda dividing one list by the other.
+++
**Answer:**

```python
return max(range(len(option_log_probs)),
           key=lambda i: option_log_probs[i] / option_lengths[i])
```
:::

---

:::step id="exercise-step8" title="Step 8: pass_at_k()"
```python
def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k draws from n samples is correct."""
    if n - c < k:
        return 1.0  # too few wrong samples to fill a k-subset: some draw must hit
    # TODO: Return the pass@k estimate from the formula above.
    raise NotImplementedError("TODO: compute the pass@k estimate")
```
+++
**Hint:** `math.comb(a, b)` is the binomial coefficient C(a, b).
+++
**Answer:**

```python
return 1.0 - math.comb(n - c, k) / math.comb(n, k)
```
:::

---

:::terminal id="exercise-output-3" title="After Steps 7 and 8: Two More Benchmarks" cmd="uv run python module_09_evaluation/src/main.py" caption="Actual output, appended to the report above. Three benchmarks, three different stories about the same two checkpoints."
  pass@1 (sampled at temperature 0.8)
    task          cases    instruct        rl      diff
    uppercase        10       68.0%     28.0%    <span class="t-fail">-40.0%</span>
    repeat            8       92.5%     22.5%    <span class="t-fail">-70.0%</span>
    reverse          24       73.3%     87.5%    <span class="success">+14.2%</span>
    qa                8      100.0%     70.0%    <span class="t-fail">-30.0%</span>

  pass@5 (sampled at temperature 0.8)
    task          cases    instruct        rl      diff
    uppercase        10       80.0%     30.0%    <span class="t-fail">-50.0%</span>
    repeat            8      100.0%     37.5%    <span class="t-fail">-62.5%</span>
    reverse          24       79.2%     95.8%    <span class="success">+16.7%</span>
    qa                8      100.0%    100.0%     <span class="t-cyan">+0.0%</span>

<span class="header">3. MULTIPLE CHOICE, SCORED BY LIKELIHOOD</span>
    instruct     16/16   100.0%
    rl           16/16   100.0%
  Chance is 25%. Nothing was generated: each option was scored under
  the model and the highest per-token log-probability won.
:::

---

<!-- .slide: id="exercise-three-stories" -->

## Three Benchmarks, Three Stories, Two Models

<div class="card-grid cols-3">
<div class="card"><h4>Perplexity says: RL is better</h4><p>1599 versus 1986 on held-out text. Lower is better, so by this metric the RL model wins.</p></div>
<div class="card"><h4>Multiple choice says: identical</h4><p>16/16 for both. Likelihood scoring never asks the model to <strong>produce</strong> anything, so it cannot see a generation collapse.</p></div>
<div class="card warn"><h4>The task suite says: disaster</h4><p>&minus;33 points overall, with a &minus;75 on <code>repeat</code>. Only the metric that made the model <strong>write an answer</strong> found the problem.</p></div>
</div>

**The scoring shape decides what you can see.** The likelihood-versus-generation distinction, running live. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="exercise-passk-detail" -->

## One Row Worth Staring At

On the `qa` task, the RL model scores **70% pass@1** and **100% pass@5**.

<div class="metric-box">
<p>The right answer is still in the distribution: five samples find it every time. It is <strong>no longer the top guess</strong>, so the single-sample numbers collapse while pass@5 does not move.</p>
</div>

The pass@1 versus pass@k gap, in reverse: RL sharpened the distribution toward `reverse` and **de-sharpened** it everywhere else. Each number alone tells a different story about the same weights. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="exercise-chart" -->

## The Picture

<div class="img-figure">
  <img src="images/task_comparison.png" alt="Grouped bar chart of per-task exact-match accuracy for the instruct and RL models, with the suite average at right">
</div>

The two rightmost bars are the headline number. The four to their left are why it is not enough. (Actual exercise output.) <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="exercise-ship" -->

## Which One Would You Ship?

:::columns cols="2" gap="34px"
**For the RL model**

- Only model good at the task we were paid to improve
- Lower perplexity
- 100% on multiple choice
+++
**Against**

- Worse at three of four tasks, one catastrophically
- Degenerate failures: emits text past the end of its answer
- Suite average fell 33 points
:::

**No metric answers this.** It depends on what the model is for. Hand the decision-maker the per-task table, not the average. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Prompt sensitivity.** Change the chat template in `_prefix_ids()` and re-run. Report how far the scores move on identical weights.
- **Contamination check.** Search the finetuning pairs for each evaluation prompt and count the exact overlaps. Then re-run with `qa` excluded and watch the headline number change.
- **Length normalization.** Score the multiple-choice set with **total** log-probability instead of the per-token average, and explain which options newly win.
- **Bootstrap confidence interval.** Resample the 50 cases 1,000 times and report the 5th and 95th percentiles of the accuracy difference. Is the `repeat` gap larger than the noise on eight cases?
- **Judge order bias.** Write a rule-based judge that prefers the longer answer, score the answer pairs in both orders, and count the flips.
- **pass@k curve.** Raise `N_SAMPLES` to 20 and plot pass@k for k = 1, 2, 5, 10, 20 on the reverse task. <!-- .element: class="text-lg" style="margin-top: 8px;" -->
