<!-- .slide: id="close-keeping-up" -->

## How to Keep Up After This Course

Anything specific will be stale within a year, so this is deliberately generic.

<div class="card-grid cols-3">
<div class="card"><h4>Read the paper</h4><p>Not the thread about the paper. Most confusion in this field comes from the gap between what a paper claims and what gets said about it.</p></div>
<div class="card"><h4>Hold demos to eval standards</h4><p>What is the baseline? What is the eval set? Was it contaminated? A demo is not a measurement.</p></div>
<div class="card"><h4>Run things yourself</h4><p>Every reference implementation in this course was built to fit on a laptop so that you could.</p></div>
</div>

You now have enough background to read most LLM papers and, where you cannot follow them, to work out which piece you are missing. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="close-shannon" -->

## Shannon, Revisited

In 1951, Shannon measured the information in English text by having people **guess the next character.** He landed on roughly one bit each.

:::columns cols="2" gap="34px"
**Where the course began**

- You counted n-grams and measured the entropy of text
- Same question, same direction as Shannon
+++
**Where it went**

- You built a transformer, pretrained it, finetuned it, improved it with RL
- Every step descended the same curve
:::

Frontier labs now spend nine figures of compute on that curve. It has not flattened yet. Where it does is an open question. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
