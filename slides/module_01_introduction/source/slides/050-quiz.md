:::divider id="quiz-divider" title="Quiz" sub="Test your understanding"
:::

---

:::quiz id="quiz-q1" title="Q1: Why Not Just Use Rules?"
GOFAI systems used hand-written rules to process language. Why did this approach ultimately fail for general language understanding? <!-- .element: class="text-lg" -->
+++
**Answer:** <!-- .element: class="text-lg" -->

- Language is ambiguous and context-dependent
- No rule set can cover every meaning of every word ("bank" alone has dozens)
- Learned models discover these patterns from data instead
:::

---

:::quiz id="quiz-q2" title="Q2: Shannon's Entropy"
Shannon estimated English has $H \approx 1.0\text{--}1.5$ bits/char. A uniform 27-character alphabet has $H = \log_2(27) = 4.75$ bits/char. <!-- .element: class="text-lg" -->

What does this gap tell us about English as a communication system? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** <!-- .element: class="text-lg" -->

- English is highly redundant: about 70-80% of each character is predictable from context
- Redundancy makes autocomplete, spell-check, and language models possible
- Shannon made this redundancy measurable
:::

---

:::quiz id="quiz-q3" title="Q3: The Memorization Problem"
A word-level 5-gram model trained on *Alice in Wonderland* produces output that is mostly exact quotes from the book. <!-- .element: class="text-lg" -->

Why does this happen, and how do modern LLMs avoid it? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** <!-- .element: class="text-lg" -->

- Most 5-word contexts appear only once in a small corpus, so the model can only reproduce the original
- Modern LLMs train on far more data
- They use learned embeddings that generalize across similar contexts, not exact count tables
:::

---

:::quiz id="quiz-q4" title="Q4: Conditional Probability"
A trigram model trained on English text has these counts for the context "th": <!-- .element: class="text-lg" -->

```python
model["th"] = Counter({"e": 50, "a": 20, "i": 15, "o": 10, "u": 5})
```

What is $P(\texttt{e} \mid \texttt{th})$, and what does it mean for the model's behavior? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** $P(\texttt{e} \mid \texttt{th}) = \frac{50}{50+20+15+10+5} = \frac{50}{100} = 0.5$ <!-- .element: class="text-lg" -->

- After "th", the model generates "e" half the time
- The rest: "a" (20%), "i" (15%), "o" (10%), "u" (5%)
- This is why trigram output contains "the" so often, plus "tha", "thi", and others
:::

---

:::quiz id="quiz-q5" title="Q5: Compression and Prediction"
Shannon proved that a perfect language model would also be a perfect compressor, and vice versa. <!-- .element: class="text-lg" -->

Why are prediction and compression fundamentally the same problem? <!-- .element: class="text-2xl" style="margin-top: 20px;" -->
+++
**Answer:** <!-- .element: class="text-lg" -->

- A symbol with probability $p$ needs $-\log_2(p)$ bits, so confident prediction means short codes
- Perfect prediction ($p = 1$) needs 0 bits
- Every compression scheme implicitly defines a probability model
- Source coding theorem: minimum average code length $=$ entropy $H(X)$
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
