:::divider id="divider-exercise" title="Exercise" sub="N-gram Language Models"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_01_introduction/exercise.py` &mdash; the only file you edit &mdash; and fill in the `NotImplementedError` lines. Everything already written for you lives in `src/`. Run after each step. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_01_introduction/src/main.py

# Run a single step (1-6, or ec for the extra credit)
uv run python module_01_introduction/src/main.py --step 4
```

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints the text your code generates, then runs the step's **tests** from `tests/`: each calls your function on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT** &mdash; every test for the step passed
- **INCORRECT** &mdash; your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE** &mdash; the function still raises `NotImplementedError`

The tag sits on the step's header line; the generated text and the individual test results follow it. Generated text alone is hard to judge by eye; the tests are how you know a step is done. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_01_introduction/src/main.py --step 3" maxw="920px" caption="Here <code>char_unigram()</code> forgot the weights, so every character came out equally often. The failing test says what it expected, what it got, and where to look."
<span class="header">=== Step 3: char_unigram() ===</span> <span class="t-fail">INCORRECT</span>
[v
*_ljnn?zt
w[g!osd‘t*vf’z‘xoùs-rj‘) ]“’yit[lly0“tyxuhw‘iyy
  <span class="success">CORRECT</span>    returns exactly as many characters as requested
  <span class="success">CORRECT</span>    only produces characters that appear in the training text
  <span class="t-fail">INCORRECT</span>  samples in proportion to frequency (9 a's : 1 b gives ~90% a)
             expected about 90% 'a', got 50% (did you pass weights=weights to random.choices?)
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: N-gram Language Models

Recreate Shannon's 1948 experiment: <!-- .element: class="text-lg" -->

- Generate text from **Alice in Wonderland** (~144,000 characters)
- Start with pure randomness, add structure step by step
- Each function is mostly written &mdash; you fill in **one key line**
- Six steps, each with its own tests in `tests/`, then an optional extra credit that scores the models

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
=== Step 1: load_text() === CORRECT
Loaded 144603 characters from module_01_introduction/data/alice.txt
  CORRECT    the Project Gutenberg header is removed
  CORRECT    the Project Gutenberg license footer is removed
  CORRECT    the book runs from the title page to THE END
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

:::terminal id="exercise-step1-output" title="Step 1: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Step 1 passes its three tests. Every later step is tagged INCOMPLETE with the TODO it is waiting on."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="success">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>
  <span class="success">CORRECT</span>    the Project Gutenberg header is removed
  <span class="success">CORRECT</span>    the Project Gutenberg license footer is removed
  <span class="success">CORRECT</span>    the book runs from the title page to THE END

<span class="header">=== Step 2: char_uniform() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return a random string of the given length</span>

<span class="header">=== Step 3: char_unigram() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return a frequency-weighted random string</span>

<span class="header">=== Step 4: build_char_ngram_model() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: set context and next_char from text[i:]</span>

<span class="header">=== Step 5: word_unigram() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return frequency-weighted random words</span>

<span class="header">=== Step 6: build_word_ngram_model() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: set context and next_word from words[i:]</span>

<span class="header">=== Extra Credit: cross_entropy() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">needs Step 4 (build_char_ngram_model) to build the models it scores</span>
:::

---

<!-- .slide: id="exercise-step2-context" -->

## Step 2: The Null Hypothesis

The baseline (0th order): **zero knowledge** of English. <!-- .element: class="text-lg" -->

- Pick each character uniformly at random from a-z and space

:::note
**Output:** <!-- .element: class="text-lg" -->

```text
=== Step 2: char_uniform() === CORRECT
pdxvbjivfqzoybrrbgjmjedbrdiuijygra sxlvewsfhbyasanasrbagcdlaibntnpt
  CORRECT    returns exactly as many characters as requested
  CORRECT    uses only the letters a-z and space
  CORRECT    every character is about equally common (z as often as e)
```
:::

Maximum entropy: **q**, **x**, **z** appear as often as **e** or **t**. The third test measures exactly that. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

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

:::terminal id="exercise-step2-output" title="Step 2: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Pure noise. Rare letters appear as often as common ones, which is exactly what the third check confirms."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-red">=== Step 2: char_uniform() ===</span> <span class="success">CORRECT</span>
pdxvbjivfqzoybrrbgjmjedbrdiuijygra sxlvewsfhbyasanasrbagcdlaibntnpt
  <span class="success">CORRECT</span>    returns exactly as many characters as requested
  <span class="success">CORRECT</span>    uses only the letters a-z and space
  <span class="success">CORRECT</span>    every character is about equally common (z as often as e)

<span class="header">=== Step 3: char_unigram() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return a frequency-weighted random string</span>

<span class="skipped">...</span>
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
=== Step 3: char_unigram() === CORRECT
 wgm oetgtee ,nei mae  tyah hur esseaca e ”yie f t o yae met
  CORRECT    returns exactly as many characters as requested
  CORRECT    only produces characters that appear in the training text
  CORRECT    samples in proportion to frequency (9 a's : 1 b gives ~90% a)
```
:::

**e**, **t**, and **space** now dominate. Still gibberish. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

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

:::terminal id="exercise-step3-output" title="Step 3: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Spaces and <strong>e</strong> dominate now, but letters still appear in random order. The third test trains on a 9:1 text and confirms the 90/10 split."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>
<span class="header t-red">=== Step 2: char_uniform() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">rqfdpqnzejwixfbxwmqegqlsogsm fmrhrbpzygktimnmqqryowwicwnoieqk</span>

<span class="header t-orange">=== Step 3: char_unigram() ===</span> <span class="success">CORRECT</span>
 wgm oetgtee ,nei mae  tyah hur esseaca e ”yie f t o yae met
  <span class="success">CORRECT</span>    returns exactly as many characters as requested
  <span class="success">CORRECT</span>    only produces characters that appear in the training text
  <span class="success">CORRECT</span>    samples in proportion to frequency (9 a's : 1 b gives ~90% a)

<span class="header">=== Step 4: build_char_ngram_model() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: set context and next_char from text[i:]</span>

<span class="skipped">...</span>
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
emofithan’” anghallit quspehisima
ttoute or it,” alkid d anaved-u d
```

</div>
+++
<div class="note">

**Trigram (n=3):** <!-- .element: class="text-lg" -->

```text
yn’t she dioug the make a
doicedly, voisaid, “don ey all v.
```

</div>
:::

Bigrams produce common pairs ("th", "he"). Trigrams produce word fragments ("the", "she"). One test confirms the model learned that **q** is followed by **u**. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-generation" -->

## The Sampling Loop (Provided)

`generate_from_char_model()` is written for you in `src/sampling.py`. It is the loop every language model runs: <!-- .element: class="text-lg" -->

```python
while len(result) < length:
    # The current context is the last n-1 characters we have generated
    context = "".join(result[-(n - 1):])

    counter = model[context]                  # what followed this context?
    chars = list(counter.keys())
    weights = list(counter.values())
    next_char = random.choices(chars, weights=weights, k=1)[0]
    result.append(next_char)
```

Build the count table and text comes out. Modern LLMs run this same loop with a learned distribution in place of the counts. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

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

:::terminal id="exercise-step4-output" title="Step 4: Output" cmd="uv run python module_01_introduction/src/main.py" caption="One fill-in and both the bigram and trigram models produce text. The tests build tiny models from &quot;abab&quot; and &quot;abcabd&quot; where the right counts are known."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>
<span class="header t-red">=== Step 2: char_uniform() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">otsfyup sszaiuenfz ptaeaejtjilonpdw ohegksbswbutbndrngennjzd</span>
<span class="header t-orange">=== Step 3: char_unigram() ===</span> <span class="success">CORRECT</span>
<span class="t-gray"> wgm oetgtee ,nei mae  tyah hur esseaca e ”yie f t o yae met</span>

<span class="header t-yellow">=== Step 4: build_char_ngram_model() ===</span> <span class="success">CORRECT</span>
--- bigram (n=2) ---
emofithan’” anghallit quspehisima ttoute or it,” alkid d ana
--- trigram (n=3) ---
yn’t she dioug the make a
doicedly, voisaid, “don ey all v. “it, an’t ask they everep,
  <span class="success">CORRECT</span>    bigram counts for 'abab' are a->b twice and b->a once
  <span class="success">CORRECT</span>    trigram contexts are 2 characters long ('abcabd' gives ab->c, ab->d, ...)
  <span class="success">CORRECT</span>    in Alice, the most common character after 'q' is 'u'

<span class="header">=== Step 5: word_unigram() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: return frequency-weighted random words</span>

<span class="skipped">...</span>
:::

---

<!-- .slide: id="exercise-step5-context" -->

## Steps 5 and 6: From Characters to Words

Same n-gram idea at the **word level**: predict the next word from the previous words. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="20px"
<div class="note">

**Word unigram (Step 5):** <!-- .element: class="text-lg" -->

```text
so the of trial: look and late,
as some voice, rabbit solemnly
```

</div>
+++
<div class="note">

**Word trigram (Step 6):** <!-- .element: class="text-lg" -->

```text
to take &#95;more&#95; than nothing.”
“nobody asked &#95;your&#95; opinion,”
```

</div>
:::

Word trigrams produce coherent phrases, sometimes whole sentences lifted from the source. The model memorizes, it does not understand. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-step5-code" title="Step 5: word_unigram()"
```python
    # Count how often each word appears
    counts = Counter(words)
    word_list = list(counts.keys())
    weights = list(counts.values())

    # TODO: Sample `length` words using the weights, then join them with spaces
    raise NotImplementedError("TODO: return frequency-weighted random words")
```
+++
**Hint:** Same pattern as `char_unigram`, but use `" ".join()` instead of `"".join()`.
+++
**Answer:** <!-- .element: class="text-lg" -->

```python
return " ".join(random.choices(word_list, weights=weights, k=length))
```
:::

---

:::terminal id="exercise-step5-output" title="Step 5: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Real words in a random order. The tests mirror Step 3, with words in place of characters."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>
<span class="header t-red">=== Step 2: char_uniform() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">hvxsycxjgcgkpqxzwowhftedcvygtdxxyveowzqallqdvpum ebxmwueehljtcz</span>
<span class="header t-orange">=== Step 3: char_unigram() ===</span> <span class="success">CORRECT</span>
<span class="t-gray"> wgm oetgtee ,nei mae  tyah hur esseaca e ”yie f t o yae met</span>
<span class="header t-yellow">=== Step 4: build_char_ngram_model() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">emofithan’” anghallit quspehisima ttoute or it,” alkid d ana</span>

<span class="header t-green">=== Step 5: word_unigram() ===</span> <span class="success">CORRECT</span>
so the of trial: look and late, as some voice, rabbit solemnly sage,
  <span class="success">CORRECT</span>    returns exactly as many words as requested, separated by spaces
  <span class="success">CORRECT</span>    only produces words that appear in the training text
  <span class="success">CORRECT</span>    samples in proportion to frequency (9 the : 1 cat gives ~90% the)

<span class="header">=== Step 6: build_word_ngram_model() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: set context and next_word from words[i:]</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step6-code" title="Step 6: build_word_ngram_model()"
Same pattern as the character model, but the context is now a **tuple of words** instead of a string of characters. Generation is provided again in `src/sampling.py`. <!-- .element: class="text-lg" style="margin-bottom: 10px;" -->

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

:::terminal id="exercise-step6-output" title="Step 6: Output" cmd="uv run python module_01_introduction/src/main.py" caption="Coherent phrases, sometimes entire sentences lifted from Alice in Wonderland. A list instead of a tuple as the context would crash here, and the runner would report that too."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>
<span class="header t-red">=== Step 2: char_uniform() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">gjliuzsopn jgzwwodhgxloqputeou rhnckckecvrgueuhycmfvwwnhu gft</span>
<span class="header t-orange">=== Step 3: char_unigram() ===</span> <span class="success">CORRECT</span>
<span class="t-gray"> wgm oetgtee ,nei mae  tyah hur esseaca e ”yie f t o yae met</span>
<span class="header t-yellow">=== Step 4: build_char_ngram_model() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">emofithan’” anghallit quspehisima ttoute or it,” alkid d ana</span>
<span class="header t-green">=== Step 5: word_unigram() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">so the of trial: look and late, as some voice, rabbit solemnly sage,</span>

<span class="header t-cyan">=== Step 6: build_word_ngram_model() ===</span> <span class="success">CORRECT</span>
--- bigram (n=2) ---
it!” said the patriotic archbishop of the cat, “a dog’s not remember
--- trigram (n=3) ---
to take &#95;more&#95; than nothing.” “nobody asked &#95;your&#95; opinion,” said al
  <span class="success">CORRECT</span>    bigram contexts are 1-word tuples ('the cat the dog the cat')
  <span class="success">CORRECT</span>    trigram contexts are 2-word tuples
  <span class="success">CORRECT</span>    in Alice, the most common word after 'the white' is 'rabbit'

<span class="header">=== Extra Credit: cross_entropy() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">Extra credit: implement cross_entropy()</span>
:::

---

:::terminal id="exercise-together" title="All Six Steps: More Context, Better Output" cmd="uv run python module_01_introduction/src/main.py" maxw="900px"
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="header t-red">=== Step 2: char_uniform() ===</span> <span class="success">CORRECT</span>
<span class="t-fg">gjliuzsopn jgzwwodhgxloqputeou rhnckckecvrgueuhycmfvwwnhu gft</span>
<span class="header t-orange">=== Step 3: char_unigram() ===</span> <span class="success">CORRECT</span>
<span class="t-fg"> wgm oetgtee ,nei mae  tyah hur esseaca e ”yie f t o yae met</span>
<span class="header t-yellow">=== Step 4: build_char_ngram_model() ===</span> <span class="success">CORRECT</span>
<span class="t-fg">emofithan’” anghallit quspehisima ttoute or it,” alkid d ana</span>
<span class="t-fg">yn’t she dioug the make a</span>
<span class="header t-green">=== Step 5: word_unigram() ===</span> <span class="success">CORRECT</span>
<span class="t-fg">so the of trial: look and late, as some voice, rabbit solemnly sage,</span>
<span class="header t-cyan">=== Step 6: build_word_ngram_model() ===</span> <span class="success">CORRECT</span>
<span class="t-fg">it!” said the patriotic archbishop of the cat, “a dog’s not remember</span>
<span class="t-fg">to take &#95;more&#95; than nothing.” “nobody asked &#95;your&#95; opinion,” said al</span>
<span class="header">=== Extra Credit: cross_entropy() ===</span> <span class="skipped">INCOMPLETE</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit: Scoring the Models

Optional. Generated text is judged by eye; **cross-entropy** judges it with a number. <!-- .element: class="text-lg" -->

- **`cross_entropy()`** &mdash; how surprised the model is by held-out text: $H(p, q) = -\frac{1}{N} \sum \log_2 q(c_i \mid \text{context})$
- **`perplexity()`** (provided) &mdash; $\text{PPL} = 2^{H(p,q)}$, the standard metric for language models

The runner trains orders 1 through 5 on the first 90% of the book and scores each on the last 10%. Perplexity should drop as order rises, then climb again once contexts become too rare to have been seen in training. The same metric evaluates modern LLMs. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::step id="exercise-extra-code" title="Extra Credit: cross_entropy()"
This one is a whole function, not a single line. <!-- .element: class="text-lg" style="margin-bottom: 10px;" -->

```python
def cross_entropy(text: str, model: dict[str, Counter]) -> float:
    # TODO: Average -log2(P(next_char | context)) over every position in text
    raise NotImplementedError("Extra credit: implement cross_entropy()")
```
+++
**Hint:** `len(next(iter(model)))` is the context size; `counter[c] / sum(counter.values())` is `P(c | context)`; use `1e-6` if the context or character was never seen.
+++
**Answer:** on the next slide. Slide over the text one position at a time, look up the probability the model gives the character that actually came next, and average the negative log of those probabilities. <!-- .element: class="text-lg" -->
:::

---

<!-- .slide: id="exercise-extra-answer" -->

## Extra Credit: Answer

```python
text = text.lower()
# Every key in the model has the same length: the context size (n-1)
context_len = len(next(iter(model)))

total_bits = 0.0
count = 0
for i in range(len(text) - context_len):
    context = text[i : i + context_len]
    next_char = text[i + context_len]
    counter = model.get(context)
    if counter and counter[next_char] > 0:
        prob = counter[next_char] / sum(counter.values())
    else:
        prob = 1e-6  # never seen: tiny probability instead of log(0)
    total_bits += -math.log2(prob)
    count += 1

return total_bits / count if count else 0.0
```

---

:::terminal id="exercise-extra-output" title="Extra Credit: Output" cmd="uv run python module_01_introduction/src/main.py --step ec" caption="Trigrams win. At n=4 and n=5 most held-out contexts were never seen in training, so the 1e-6 fallback fires often and perplexity climbs. That is data sparsity, the wall that stopped n-gram models."
<span class="header">=== Step 1: load_text() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Loaded 144603 characters from module_01_introduction/data/alice.txt</span>

<span class="header t-blue">=== Extra Credit: cross_entropy() ===</span> <span class="success">CORRECT</span>
Train on the first 90% of Alice, score the held-out last 10%:

  order   bits/char   perplexity
  n=1        4.437        21.65
  n=2        3.458        10.99
  n=3        2.845         7.18
  n=4        3.123         8.71
  n=5        4.678        25.61

  <span class="success">CORRECT</span>    a model that predicts every character perfectly scores 0 bits
  <span class="success">CORRECT</span>    a 50/50 guess costs exactly 1 bit per character
  <span class="success">CORRECT</span>    an unseen character is smoothed to 1e-6 (about 19.93 bits)
  <span class="success">CORRECT</span>    on held-out Alice, bits/char drops from unigram to bigram to trigram
:::
