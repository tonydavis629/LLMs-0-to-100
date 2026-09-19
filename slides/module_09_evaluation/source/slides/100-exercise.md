:::divider id="divider-exercise" title="Exercise" sub="Build a small benchmark suite and score two models"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Fill in the eight `NotImplementedError` lines in `module_09_evaluation/exercise.py`, the only file you edit. Everything else (checkpoints, tokenizer, sampler, data, plotting, runner) is provided in `src/`. Run after each step. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_09_evaluation/src/main.py

# Run a single step (1-8)
uv run python module_09_evaluation/src/main.py --step 4
```

Both checkpoints ship with the repo, trained for this module by `src/make_checkpoints.py` from the Module 5 base model. <!-- .element: class="text-md" style="margin-top: 22px;" -->

A full run prints the **protocol** first, then perplexity, per-task tables (exact match, F1, pass@1, pass@5), multiple choice, and the suite score. Step 6 saves a bar chart to `output/task_comparison.png`. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your metric reports about the two models, then runs the step's **tests** from `tests/`. Each test calls your function on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`, or it needs an earlier step that does

The tag sits on the step's header line. A metric can print believable numbers on real model output and still be wrong, so trust the tests, not the table. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_09_evaluation/src/main.py --step 4" maxw="980px" caption="Here <code>token_f1()</code> returned the arithmetic mean <code>(precision + recall) / 2</code>. On the one real case precision equals recall, so both means give 0.67 and the output looks fine. The test with P = 1 and R = 1/3 catches it and names the mistake."
<span class="header">=== Step 4: token_f1() ===</span> <span class="t-fail">INCORRECT</span>
  Generating 50 greedy + 250 sampled answers per model...
  Greedy answers that earn partial credit (F1 between 0 and 1):
    [qa] 'opposite of day?'   want 'it is night'
        rl       'it is cold'                 0.67
  <span class="success">CORRECT</span>    partial credit: 'it is bluu' against 'it is blue' has P = R = 2/3, so F1 = 0.667
  <span class="t-fail">INCORRECT</span>  P and R differ: 'blue' against 'it is blue' has P = 1, R = 1/3, so F1 = 0.5
             expected 0.5000, got 0.6667 (that is the arithmetic mean; F1 is the harmonic mean)
  <span class="success">CORRECT</span>    a correct answer scores 1.0: 'It is blue.' against 'it is blue'
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Nothing Here Trains a Model

Both checkpoints are finished:

- `instruct_model.pt`: the Module 6 model, finetuned on four toy tasks
- `rl_model.pt`: the same model after Module 7's GRPO run on **reverse only**

:::columns cols="2" gap="30px"
**You write the metrics**

Perplexity, normalization, exact match, token F1, likelihood-scored multiple choice, pass@k, per-task accuracy, suite average. Each is one line, and every step has its own tests in `tests/`.
+++
**The payoff is the table**

- RL model wins its trained task by 17 points
- Loses 25 to 75 points on the other three
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
    """Convert an average per-token cross-entropy loss (in nats) into perplexity.

    This is the base-model metric from Module 5, and the only number in this suite
    that needs no labels at all: just held-out text and the model's own loss on it.
    Perplexity is the model's average branching factor &mdash; roughly, how many
    equally likely tokens it is choosing among at each position. Lower is better,
    and a perplexity of 1.0 would mean the model was never surprised.

    Args:
        mean_token_loss: Average -log p(token) over the held-out tokens, in nats.

    Returns:
        The perplexity as a plain float.
    """
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

:::terminal id="exercise-output-1" title="After Step 1: The Protocol, Then One Number" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="The protocol is printed before any score. Note the result: the RL model has LOWER perplexity, which is about to turn out to mean nothing."
<span class="header">MODULE 9: evaluating two finished checkpoints</span>
<span class="skipped">...</span>
Protocol
  chat template     &lt;|user|&gt; PROMPT &lt;|end|&gt; &lt;|assistant|&gt; ANSWER &lt;|end|&gt;
  normalization     lowercase, strip punctuation, collapse whitespace
  generation budget 14 tokens    context 128
  decoding          greedy for exact match and F1
                    5 samples at temperature 0.8 for pass@k
  seed              1337 (per case, so both models see the same draws)
  suite             50 generated cases across 4 tasks, 16 multiple-choice questions
  held-out text     3,916 characters, never seen in training

<span class="header t-green">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
    model          loss (nats)    perplexity
    instruct            7.5941       1986.48
    <span class="success">rl                  7.3773       1599.26</span>
  Lower is better. This says nothing about whether either model
  follows instructions, which is exactly its limitation.
  <span class="success">CORRECT</span>    a loss of 0 nats (never surprised) gives perplexity 1.0
  <span class="success">CORRECT</span>    a loss of ln 2 = 0.693 nats (a coin flip per token) gives perplexity 2.0
  <span class="success">CORRECT</span>    a loss of ln 27 = 3.296 nats (uniform over 27 characters) gives perplexity 27.0

<span class="header">=== Step 2: normalize_answer() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: collapse the whitespace in the normalized answer</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step2" title="Step 2: normalize_answer()"
```python
def normalize_answer(text: str) -> str:
    """Put a generated answer into the canonical form that scoring compares.

    "4", " 4 ", and "4." are the same answer and three different strings, so every
    benchmark ships a normalization step before it compares anything. Without one,
    exact match measures formatting instead of correctness. The first two lines are
    provided; you write the whitespace rule.

    Returns:
        The normalized string.
    """
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
    """Score 1.0 if the normalized prediction equals any acceptable answer, else 0.0.

    Benchmarks carry a *list* of acceptable answers because more than one string can
    be right ("it is blue" and "blue"). Exact match is the most transparent metric
    there is, and the most brittle: one extra word and a correct answer scores zero.

    Returns:
        1.0 or 0.0.
    """
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

:::terminal id="exercise-output-cases" title="After Step 3: One Case Per Task" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="Actual output. The cases show how the RL model fails: swapped letters, run-on answers, no English."
<span class="header">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: normalize_answer() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 3: exact_match() ===</span> <span class="success">CORRECT</span>
  Generating 50 greedy + 250 sampled answers per model...
  Greedy answers that match exactly: instruct 42/50, rl 33/50
  One case per task, with its exact-match score:
    [uppercase] 'uppercase: metric'   want 'METRIC'
        instruct 'METRIC'                     <span class="success">1.0</span>
        rl       'METRCI'                     <span class="t-fail">0.0</span>
    [repeat] 'repeat: task'   want 'task'
        instruct 'task'                       <span class="success">1.0</span>
        rl       'taskto'                     <span class="t-fail">0.0</span>
    [reverse] 'reverse: bamkf'   want 'fkmab'
        instruct 'fkmaq'                      <span class="t-fail">0.0</span>
        rl       'fkmab'                      <span class="success">1.0</span>
    [qa] 'opposite of up?'   want 'it is down'
        instruct 'it is down'                 <span class="success">1.0</span>
        rl       'utripso&lt;|user|&gt;&lt;|user|...'  <span class="t-fail">0.0</span>
  <span class="success">CORRECT</span>    formatting does not count: ' Blue. ' against ['blue'] scores 1.0
  <span class="success">CORRECT</span>    any acceptable answer counts: 'it is blue' against ['blue', 'It is blue.'] scores 1.0
  <span class="success">CORRECT</span>    an extra word scores zero: 'it is blue' against ['blue'] scores 0.0
  <span class="success">CORRECT</span>    returns the float 1.0 or 0.0, not True or False

<span class="header">=== Step 4: token_f1() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: combine precision and recall into F1</span>
:::

---

:::step id="exercise-step4" title="Step 4: token_f1()"
```python
def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and one reference answer.

    F1 gives partial credit where exact match gives none: "it is bluu" shares two of
    its three tokens with "it is blue", which is worth more than zero. Precision is
    the fraction of predicted tokens that are correct, recall the fraction of
    reference tokens that were produced, and F1 is their harmonic mean &mdash; the
    information-retrieval metric that entered question answering through SQuAD.

    Everything up to the harmonic mean is provided.

    Returns:
        A score in [0.0, 1.0].
    """
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    # Counter intersection counts each shared token no more times than it appears
    # in both sides, so repeating a word cannot inflate the overlap.
    shared = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if shared == 0:
        return 0.0
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

:::terminal id="exercise-output-f1" title="After Step 4: Partial Credit" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="Actual output. Exact match scores the RL model's wrong answer 0. F1 gives it two thirds credit for the shared words 'it is', which come from the answer template, not the fact."
<span class="header">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: normalize_answer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: exact_match() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 4: token_f1() ===</span> <span class="success">CORRECT</span>
  Greedy answers that earn partial credit (F1 between 0 and 1):
    [qa] 'opposite of day?'   want 'it is night'
        rl       'it is cold'                 <span class="t-yellow">0.67</span>
  <span class="success">CORRECT</span>    partial credit: 'it is bluu' against 'it is blue' has P = R = 2/3, so F1 = 0.667
  <span class="success">CORRECT</span>    P and R differ: 'blue' against 'it is blue' has P = 1, R = 1/3, so F1 = 0.5
  <span class="success">CORRECT</span>    a correct answer scores 1.0: 'It is blue.' against 'it is blue'

<span class="header">=== Step 5: task_accuracy() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: average the per-case scores within each task</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step5" title="Step 5: task_accuracy()"
```python
def task_accuracy(scores_by_task: dict[str, list[float]]) -> dict[str, float]:
    """Average each task's per-case scores into one number per task.

    Keeping the breakdown is the habit this whole exercise is arguing for. An
    aggregate can hide a total regression on one task inside a small overall gain,
    and only the per-task view shows it.

    Args:
        scores_by_task: Task name -> the list of per-case scores for that task.

    Returns:
        Task name -> mean score.
    """
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

:::terminal id="exercise-output-2" title="After Step 5: The Per-Task Tables" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="Actual output. RL gains 16.7 points on reverse, the task it trained on, and loses on the other three."
<span class="header">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: normalize_answer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: exact_match() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: token_f1() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 5: task_accuracy() ===</span> <span class="success">CORRECT</span>
  EXACT MATCH (greedy decoding)
  The strictest metric: the normalized answer must equal an acceptable answer.
    task          cases    instruct        rl      diff
    uppercase        10       80.0%     30.0%    <span class="t-fail">-50.0%</span>
    repeat            8      100.0%     25.0%    <span class="t-fail">-75.0%</span>
    reverse          24       75.0%     91.7%    <span class="success">+16.7%</span>
    qa                8      100.0%     75.0%    <span class="t-fail">-25.0%</span>

  TOKEN F1 (greedy decoding)
  Partial credit for overlapping tokens; equals exact match on one-word answers.
    task          cases    instruct        rl      diff
    uppercase        10       80.0%     30.0%    <span class="t-fail">-50.0%</span>
    repeat            8      100.0%     25.0%    <span class="t-fail">-75.0%</span>
    reverse          24       75.0%     91.7%    <span class="success">+16.7%</span>
    qa                8      100.0%     83.3%    <span class="t-fail">-16.7%</span>
  <span class="success">CORRECT</span>    averages within each task: qa [1, 0, 1, 1] gives 0.75, repeat [0, 1] gives 0.5
  <span class="success">CORRECT</span>    works on partial-credit scores: qa [0.5, 1.0, 0.0] gives 0.5
  <span class="success">CORRECT</span>    each task is divided by its own case count: 24 right of 24 is 1.0, 0 of 8 is 0.0

<span class="header">=== Step 6: suite_score() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: average the per-task scores into the suite score</span>
:::

---

:::step id="exercise-step6" title="Step 6: suite_score()"
```python
def suite_score(per_task: dict[str, float]) -> float:
    """Average the per-task scores into the single headline number.

    Note what this choice does: averaging the four *task* scores weights every task
    equally, while averaging all 50 *cases* would let the task with the most cases
    dominate. Neither is more correct, but they give different numbers on the same
    model, which is the whole reason a benchmark has to publish its protocol.

    Returns:
        The mean of the per-task scores.
    """
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

:::terminal id="exercise-output-suite" title="After Step 6: The Headline Number" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="Actual output. Averaging the four task scores puts the gap at 33.3 points; averaging all 50 cases puts it at 18.0, because <code>reverse</code> has 24 of the cases. Same answers, same metric, two protocols."
<span class="header">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: normalize_answer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: exact_match() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: token_f1() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: task_accuracy() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 6: suite_score() ===</span> <span class="success">CORRECT</span>
    instruct       88.8%   (mean of the four task scores)
    rl             55.4%   (mean of the four task scores)
  Overall difference: <span class="t-fail">-33.3%</span>
  Averaging all 50 cases instead gives instruct 84.0%, rl 66.0% (<span class="t-fail">-18.0%</span>).
  Read the per-task table before believing either number.
  Saved chart to output/task_comparison.png
  <span class="success">CORRECT</span>    the instruct model's four task scores (80%, 100%, 75%, 100%) average to 0.8875
  <span class="success">CORRECT</span>    one task at 1.0 and one at 0.0 give 0.5
  <span class="success">CORRECT</span>    a one-task suite scores the same as its task: {'qa': 0.25} gives 0.25

<span class="header">=== Step 7: score_multiple_choice() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: choose the option with the best per-token log-probability</span>

<span class="header">=== Step 8: pass_at_k() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: compute the pass@k estimate</span>
:::

---

:::step id="exercise-step7" title="Step 7: score_multiple_choice()"
```python
def score_multiple_choice(option_log_probs: list[float], option_lengths: list[int]) -> int:
    """Pick the option the model finds most likely, per token.

    This is how MMLU, HellaSwag, and ARC are actually run: nothing is generated. The
    runner scores each candidate answer under the model and hands you its total
    log-probability and its length in tokens. Dividing by the length is what makes
    the comparison fair &mdash; every extra token adds another negative number, so
    total log-probability systematically prefers the shortest option.

    Args:
        option_log_probs: Total log-probability of each option, one per option.
        option_lengths: Number of scored tokens in each option, same order.

    Returns:
        The index of the chosen option.
    """
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

:::terminal id="exercise-output-mc" title="After Step 7: Multiple Choice" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="Actual output. Both models pick the right option on all 16 questions. Likelihood scoring never asks a model to write anything, so it cannot see the RL model's broken generations."
<span class="header">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: normalize_answer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: exact_match() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: token_f1() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: task_accuracy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: suite_score() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 7: score_multiple_choice() ===</span> <span class="success">CORRECT</span>
    instruct     16/16   100.0%
    rl           16/16   100.0%
  Chance is 25%. Nothing was generated: each option was scored under
  the model and the highest per-token log-probability won.
  <span class="success">CORRECT</span>    equal lengths: totals [-4, -2, -6] over 2 tokens each pick option 1
  <span class="success">CORRECT</span>    divides by length: [-3, -8] over [1, 4] tokens picks option 1 (-2.0 beats -3.0)
  <span class="success">CORRECT</span>    four options: [-9, -6, -12, -7] over [3, 2, 3, 4] tokens pick option 3

<span class="header">=== Step 8: pass_at_k() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: compute the pass@k estimate</span>
:::

---

:::step id="exercise-step8" title="Step 8: pass_at_k()"
```python
def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k draws from n samples is correct.

    From the HumanEval paper. Sampling n completions and observing c correct ones,
    the unbiased estimate of pass@k is one minus the probability that a random
    k-subset misses every correct sample:

        pass@k = 1 - C(n - c, k) / C(n, k)

    pass@1 is the ordinary accuracy of a single sample. Large-k pass@k asks a
    different question: is the right answer anywhere in the model's distribution?
    Module 7's headline claim &mdash; that RL sharpens the distribution rather than
    expanding it &mdash; is exactly a claim about the gap between these two.

    Args:
        n: Total samples drawn for this case.
        c: How many of them were correct.
        k: Budget of attempts to score.

    Returns:
        A probability in [0.0, 1.0].
    """
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

:::terminal id="exercise-output-3" title="After Step 8: pass@1 and pass@5" cmd="uv run python module_09_evaluation/src/main.py" maxw="920px" caption="Actual output. Three benchmarks, three different stories about the same two checkpoints."
<span class="header">=== Step 1: perplexity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: normalize_answer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: exact_match() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: token_f1() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: task_accuracy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: suite_score() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 7: score_multiple_choice() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 8: pass_at_k() ===</span> <span class="success">CORRECT</span>
  pass@1 (sampled at temperature 0.8)
  Accuracy of a single sampled answer, estimated from N=5 samples per case.
    task          cases    instruct        rl      diff
    uppercase        10       68.0%     28.0%    <span class="t-fail">-40.0%</span>
    repeat            8       92.5%     22.5%    <span class="t-fail">-70.0%</span>
    reverse          24       73.3%     87.5%    <span class="success">+14.2%</span>
    qa                8      100.0%     70.0%    <span class="t-fail">-30.0%</span>

  pass@5 (sampled at temperature 0.8)
  Is the right answer anywhere in the model's distribution across 5 tries?
    task          cases    instruct        rl      diff
    uppercase        10       80.0%     30.0%    <span class="t-fail">-50.0%</span>
    repeat            8      100.0%     37.5%    <span class="t-fail">-62.5%</span>
    reverse          24       79.2%     95.8%    <span class="success">+16.7%</span>
    qa                8      100.0%    100.0%     <span class="t-cyan">+0.0%</span>
  <span class="success">CORRECT</span>    pass@1 is plain accuracy: n=5, c=2, k=1 gives 2/5 = 0.4
  <span class="success">CORRECT</span>    n=5, c=2, k=3 gives 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
  <span class="success">CORRECT</span>    no correct samples scores 0.0 at any k: n=5, c=0, k=5
  <span class="success">CORRECT</span>    agrees with counting every k-subset of n=6 samples, for every c and k
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
- **Length normalization.** Score the multiple-choice set with **total** log-probability instead of the per-token average. Both models still get 16/16 here: explain why, then add a question where the two methods disagree.
- **Bootstrap confidence interval.** Resample the 50 cases 1,000 times and report the 5th and 95th percentiles of the accuracy difference. Is the `repeat` gap larger than the noise on eight cases?
- **Judge order bias.** Write a rule-based judge that prefers the longer answer, score the answer pairs in both orders, and count the flips.
- **pass@k curve.** Raise `N_SAMPLES` to 20 and plot pass@k for k = 1, 2, 5, 10, 20 on the reverse task. <!-- .element: class="text-lg" style="margin-top: 8px;" -->
