<!-- .slide: id="eval-easier-than-gen" -->

## Evaluation Is Often Easier Than Generation

Writing a correct solution can be hard. Checking a proposed solution can be simple.

<div class="example-pair">
<div><p class="example-label">Generate</p><p>Write a program that sorts a list.</p><p class="example-detail">Many decisions; many possible programs.</p></div>
<div><p class="example-label">Evaluate</p><p>Run the program on test cases.</p><p class="example-detail">Turn the result into a score.</p></div>
</div>

**The opportunity:** learn from scores on the model's own attempts, even when we do not have an ideal response to copy.

---

<!-- .slide: id="resolve-three-limits" -->

## Change What Training Rewards

SFT asks: **How likely is the demonstrated answer?**

RL asks: **How much reward do the model's own answers earn?**

:::note
The model can explore alternative answers and reinforce those that score well, including answers absent from the demonstrations.
:::

The reward must distinguish better attempts from worse ones. It does not need to specify every correct token.
