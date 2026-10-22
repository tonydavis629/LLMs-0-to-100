<!-- .slide: id="reward-cheaper-supervision" -->

## A Score Can Supervise Many Attempts

:::columns cols="2" gap="34px"
**Demonstration**

Someone writes the response the model should imitate.
+++
**Reward**

A person or program evaluates a response the model generated.
:::

Human comparisons can be easier to collect than ideal answers. A reusable checker can score many new attempts automatically.

The feedback is **sparser**: a whole response may receive just one number.

---

<!-- .slide: id="reward-spectrum" -->

## Where Does the Reward Come From?

The reward is defined **outside the policy being trained**.

<table class="method-table reward-table">
<thead><tr><th>Reward source</th><th>What determines the score?</th><th>Example pairings</th></tr></thead>
<tbody>
<tr><td>Programmed environment</td><td>Game points, reaching a goal, simulator rules</td><td>REINFORCE or PPO in classical RL</td></tr>
<tr><td>Programmatic verifier</td><td>Passing tests or matching a known answer</td><td>REINFORCE or GRPO + verifier<br/><strong>RLVR</strong></td></tr>
<tr><td>Learned reward model</td><td>A model trained on human preferences</td><td>PPO + learned reward model<br/><strong>RLHF</strong></td></tr>
<tr><td>AI feedback</td><td>An AI judge applying a rubric or principles</td><td>PPO or GRPO + AI feedback<br/><strong>RLAIF</strong></td></tr>
</tbody>
</table>

**Reward source and optimizer are separate choices.** REINFORCE, PPO, and GRPO can each use programmed or learned rewards. Our visuals hold the checker fixed.
