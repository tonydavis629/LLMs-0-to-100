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

Recreate Shannon's 1948 experiment: <!-- .element: class="text-lg" -->

- Generate text from **Alice in Wonderland** (~144,000 characters)
- Start with pure randomness, add structure step by step
- Each function is mostly written &mdash; you fill in **one key line**

---

<!-- .slide: id="exercise-step1-context" -->

## Step 1: Load the Training Data

Load the **corpus** (the training text): <!-- .element: class="text-lg" -->

- The file comes from Project Gutenberg
- Gutenberg wraps the book in headers and footers
- Strip those out

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

    # TODO: return the cleaned book text
    raise NotImplementedError("TODO: return the joined lines")
```
+++
**Hint:** Join `lines[start_idx:end_idx]` with newlines and return the result.
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

The baseline (0th order): **zero knowledge** of English. <!-- .element: class="text-lg" -->

- Pick each character uniformly at random from a-z and space

:::note
**Output:** <!-- .element: class="text-lg" -->

```text
=== 0th Order: Uniform Random Characters ===
rahgtsyclafnafrofpvavsjezjccwqvto kowqxptbghcg
```
:::

Maximum entropy: **q**, **x**, **z** appear as often as **e** or **t**. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step2-code" title="Step 2: char_uniform()"
```python
    # The 27 characters we can pick from (26 letters + space)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")

    # TODO: Sample `length` random characters from alphabet and join them into a string
    raise NotImplementedError("TODO: return a random string of the given length")
```
+++
**Hint:** `random.choices(alphabet, k=length)` returns a list; `"".join(...)` combines it.
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

**1st order:** use the corpus. <!-- .element: class="text-lg" -->

- Count how often each character appears
- Sample proportionally to those counts
- Letter frequencies are known; letter order is not

:::note
**Output:** <!-- .element: class="text-lg" -->

```text
=== 1st Order: Character Unigrams ===
 irt  flniteit et b
b as,allh b e"oeh h  itrltlr
```
:::

**e**, **t**, and **space** now dominate. Still gibberish: no context. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step3-code" title="Step 3: char_unigram()"
```python
    # Separate the characters and their counts into two parallel lists
    chars = list(counts.keys())       # e.g. ["a", "b", " ", "e", ...]
    weights = list(counts.values())   # e.g. [2, 1, 5, 3, ...]

    # TODO: Sample `length` characters using the weights, then join them into a string
    raise NotImplementedError("TODO: return a frequency-weighted random string")
```
+++
**Hint:** Same as `char_uniform`, but pass `weights=weights` to `random.choices()`.
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

**Bigram** (2nd order): condition on the previous character. <!-- .element: class="text-lg" -->

- Unigram asks: how common is **e**?
- Bigram asks: after **t**, how common is **e**?

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

Bigrams produce common pairs ("th", "he"). Trigrams produce word fragments ("the", "alice"). <!-- .element: class="text-lg" style="margin-top: 15px;" -->

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
**Hint:** The window starts at `i` &mdash; the context is its first `n-1` characters, and `next_char` is the one right after.
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

Generate text from the count table: <!-- .element: class="text-lg" -->

- Start with a seed
- Look up the current context
- Sample the next character proportionally
- Repeat

Modern LLMs run this same loop with more context and more parameters. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step5-code" title="Step 5: generate_from_char_model()"
```python
    while len(result) < length:
        # TODO: Get the current context: the last (n-1) characters joined together
        context = None
        if context is None:
            raise NotImplementedError("TODO: set context from the last n-1 chars of result")

        if context in model:
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        else:
            # Context not in model: fall back to a random context
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
**Hint:** `"".join(result[-(n - 1):])` gives the last `n-1` characters as a string.
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

Same n-gram idea at the **word level**: predict the next word from the previous words. <!-- .element: class="text-lg" -->

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

Word trigrams produce coherent phrases, sometimes whole sentences lifted from the source. The model memorizes, it does not understand. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step6-code" title="Step 6: build_word_ngram_model()"
Same pattern as the character model, but the context is now a **tuple of words** instead of a string of characters. <!-- .element: class="text-lg" style="margin-bottom: 10px;" -->

```python
for i in range(len(words) - n + 1):
    # TODO: Extract the context tuple (first n-1 words) and next_word (the nth word)
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
**Hint:** Like the char model &mdash; the context is the first `n-1` words as a tuple, and `next_word` is the one after.
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
**Hint:** `tuple(result[-(n - 1):])` gives the last `n-1` words as a tuple.
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

Optional: <!-- .element: class="text-lg" -->

- **`cross_entropy()`** &mdash; measure how surprised the model is: $H(p, q) = -\frac{1}{N} \sum \log_2 q(c_i \mid \text{context})$
- **`perplexity()`** &mdash; convert to perplexity: $\text{PPL} = 2^{H(p,q)}$, a standard metric for language models

Perplexity should drop as n-gram order rises. The same metric evaluates modern LLMs. <!-- .element: class="text-lg" style="margin-top: 15px;" -->
