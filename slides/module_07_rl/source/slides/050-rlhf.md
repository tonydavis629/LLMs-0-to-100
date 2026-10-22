:::divider id="divider-rlhf" title="RLHF and the Reward Model" sub="The recipe that produced ChatGPT"
:::

---

<!-- .slide: id="rlhf-completes-instructgpt" -->

## Completing the InstructGPT Recipe

Module 6 covered **stage 1**. InstructGPT added preference learning and PPO in stages 2 and 3.

:::columns cols="3" gap="22px"
**1. SFT** &check;

Finetune on human demonstrations. **Module 6.**
+++
**2. Reward model**

Train a model to predict human **preferences**. This section.
+++
**3. RL (PPO)**

Optimize the policy against the reward model. This section.
:::

Origins: **Christiano et al. (2017)** trained agents from human preferences; **Stiennon et al. (2020)** applied it to summarization; **Ouyang et al. (2022, InstructGPT)** made it the standard recipe. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="rlhf-pipeline" -->

## The Three-Stage Recipe

<div class="rlhf-flow">
<svg viewBox="0 0 960 500" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The three stages of RLHF: supervised finetuning, reward model, and PPO optimization">
<defs>
<marker id="rh-ah" markerWidth="10" markerHeight="8" refX="7.5" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#8892a4"></path></marker>
<marker id="rh-rm" markerWidth="10" markerHeight="8" refX="7.5" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#f5a623"></path></marker>
</defs>
<rect x="8" y="6" width="944" height="130" rx="16" fill="#4a9eff" fill-opacity="0.04" stroke="#2a3450" stroke-width="1.4"></rect>
<text x="28" y="34" fill="#f5a623" font-size="19" font-weight="700">Stage 1 &#183; Supervised finetuning (SFT)</text>
<rect x="30" y="50" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="129" y="79" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Prompts</text>
<text x="129" y="100" text-anchor="middle" fill="#8892a4" font-size="12">real user requests</text>
<path d="M232,83 L258,83" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="264" y="50" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="363" y="79" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Human demos</text>
<text x="363" y="100" text-anchor="middle" fill="#8892a4" font-size="12">labelers write answers</text>
<path d="M466,83 L492,83" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="498" y="50" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="597" y="79" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Finetune base LM</text>
<text x="597" y="100" text-anchor="middle" fill="#8892a4" font-size="12">supervised learning</text>
<path d="M700,83 L726,83" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="732" y="50" width="198" height="66" rx="12" fill="#f5a623" fill-opacity="0.14" stroke="#f5a623" stroke-width="2"></rect>
<text x="831" y="79" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">SFT model</text>
<text x="831" y="100" text-anchor="middle" fill="#8892a4" font-size="12">the initial policy</text>
<path d="M480,138 L480,166" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<text x="496" y="157" fill="#8892a4" font-size="12.5" font-style="italic">SFT model seeds the RL policy</text>
<rect x="8" y="170" width="944" height="130" rx="16" fill="#4a9eff" fill-opacity="0.04" stroke="#2a3450" stroke-width="1.4"></rect>
<text x="28" y="198" fill="#f5a623" font-size="19" font-weight="700">Stage 2 &#183; Reward model (RM)</text>
<rect x="30" y="214" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="129" y="243" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Sample responses</text>
<text x="129" y="264" text-anchor="middle" fill="#8892a4" font-size="12">k answers per prompt</text>
<path d="M232,247 L258,247" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="264" y="214" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="363" y="243" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Humans rank them</text>
<text x="363" y="264" text-anchor="middle" fill="#8892a4" font-size="12">e.g. A &gt; B &gt; C</text>
<path d="M466,247 L492,247" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="498" y="214" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="597" y="243" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Train reward model</text>
<text x="597" y="264" text-anchor="middle" fill="#8892a4" font-size="12">Bradley-Terry loss</text>
<path d="M700,247 L726,247" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="732" y="214" width="198" height="66" rx="12" fill="#f5a623" fill-opacity="0.14" stroke="#f5a623" stroke-width="2"></rect>
<text x="831" y="243" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Reward model</text>
<text x="831" y="264" text-anchor="middle" fill="#8892a4" font-size="12">one scalar score</text>
<path d="M831,280 L831,307 L597,307 L597,376" fill="none" stroke="#f5a623" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#rh-rm)"></path>
<text x="714" y="322" text-anchor="middle" fill="#f5a623" font-size="12" font-style="italic">reward model scores the answer</text>
<rect x="8" y="334" width="944" height="158" rx="16" fill="#4a9eff" fill-opacity="0.04" stroke="#2a3450" stroke-width="1.4"></rect>
<text x="28" y="362" fill="#f5a623" font-size="19" font-weight="700">Stage 3 &#183; RL optimization (PPO)</text>
<rect x="30" y="378" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="129" y="407" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">New prompt</text>
<text x="129" y="428" text-anchor="middle" fill="#8892a4" font-size="12">sample a task</text>
<path d="M232,411 L258,411" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="264" y="378" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="363" y="407" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Policy responds</text>
<text x="363" y="428" text-anchor="middle" fill="#8892a4" font-size="12">generate an answer</text>
<path d="M466,411 L492,411" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="498" y="378" width="198" height="66" rx="12" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="1.8"></rect>
<text x="597" y="407" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">Score answer</text>
<text x="597" y="428" text-anchor="middle" fill="#8892a4" font-size="12">reward &#8722; &#946;&#183;KL to ref</text>
<path d="M700,411 L726,411" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<rect x="732" y="378" width="198" height="66" rx="12" fill="#f5a623" fill-opacity="0.14" stroke="#f5a623" stroke-width="2"></rect>
<text x="831" y="407" text-anchor="middle" fill="#e8eaf0" font-size="16" font-weight="600">PPO update</text>
<text x="831" y="428" text-anchor="middle" fill="#8892a4" font-size="12">improve the policy</text>
<path d="M831,446 C831,483 470,483 363,470 L363,446" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#rh-ah)"></path>
<text x="640" y="465" text-anchor="middle" fill="#8892a4" font-size="12.5" font-style="italic">repeat: the improved policy generates again</text>
</svg>
</div>

Demonstrations teach the format, human comparisons train a reward model, and PPO optimizes the policy against that learned reward. <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="figure-christiano" -->

:::figure img="images/christiano.jpg" name="Paul Christiano and collaborators" kicker="Deep Reinforcement Learning from Human Preferences (2017)"
- Showed an agent could learn from **human comparisons** of trajectories, with no hand-coded reward
- A human picks which of two behaviors looks better; a reward model learns to predict the choice
- The origin of the reward-model recipe that RLHF scaled to language
:::

---

<!-- .slide: id="reward-model" -->

## Stage 2: The Reward Model

Humans rank two responses to the same prompt. A model learns to predict the preferred one.

The **Bradley-Terry** loss turns pairwise preferences into a scalar reward:

$$\mathcal L_{\text{RM}} = -\mathbb E_{(x, y_w, y_l)}\big[\log \sigma\big(\textcolor{#50c878}{r_\phi}(x, \textcolor{#4a9eff}{y_w}) - \textcolor{#50c878}{r_\phi}(x, \textcolor{#f5a623}{y_l})\big)\big]$$

:::columns cols="2" gap="34px"
- $\textcolor{#4a9eff}{y_w}$: the preferred (winning) response
- $\textcolor{#f5a623}{y_l}$: the rejected (losing) response
+++
- $\textcolor{#50c878}{r_\phi}$: the learned reward model
- The loss rises when the winner scores below the loser
:::

The reward model is a **learned, imperfect proxy** for human judgment. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="rlhf-objective" -->

## Stage 3: Optimize the Policy

PPO maximizes the learned reward. A **KL penalty** discourages drift away from the frozen SFT reference.

$$\max_{\textcolor{#4a9eff}{\pi_\theta}} \textcolor{#f5a623}{\mathbb E_{y \sim \pi_\theta}\big[ r_\phi(x, y) \big]} - \textcolor{#50c878}{\beta \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)\big)}$$

:::columns cols="2" gap="34px"
- First term: $\textcolor{#f5a623}{\mathbb E[r_\phi]}$ means **maximize reward**
- Second term: $\textcolor{#50c878}{\mathrm{KL}}$ means **stay close** to the reference policy
+++
- $\textcolor{#50c878}{\beta}$: how hard to pull back toward the reference
- The reference stays fixed; it is different from the sampling-policy snapshot used for PPO clipping
:::

---

<!-- .slide: id="kl-vs-cross-entropy" -->

## KL Divergence vs. Cross-Entropy

$$H(p, q) = H(p) + \textcolor{#50c878}{\mathrm{KL}(p \Vert q)}$$

:::columns cols="2" gap="34px"
- **Cross-entropy**: bits to encode samples from $p$ with a code built for $q$
- **Entropy** $H(p)$: bits even a perfect model pays, because $p$ is random
- $\textcolor{#50c878}{\mathrm{KL}}$: the extra bits from using $q$ in place of $p$
+++
**Fair coin, model matches exactly**

$p = q = [0.5, 0.5]$

- Cross-entropy: 1 bit
- $\textcolor{#50c878}{\mathrm{KL}}$: 0
:::

A drift penalty should be zero when the policy matches the reference. Only KL is. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

With one-hot SFT targets, $H(p) = 0$, so the two are equal. <!-- .element: class="text-muted" -->

**Measured by sampling:** draw responses from the policy, score each under both models, and average the log-ratio.

$$\textcolor{#50c878}{\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})} = \mathbb E_{y \sim \pi_\theta}\big[\log \pi_\theta(y) - \log \pi_{\text{ref}}(y)\big]$$

---

<!-- .slide: id="figure-ouyang-7" -->

:::figure img="images/ouyang.jpg" name="Long Ouyang and the InstructGPT team (OpenAI)" kicker="Training Language Models to Follow Instructions with Human Feedback (2022)"
- Made **SFT + reward model + PPO** the standard LLM post-training recipe
- We met them in Module 6 for stage 1; they own all three
- The direct technical ancestor of ChatGPT
:::


---

<!-- .slide: id="rlaif" -->

## RLAIF and Constitutional AI

Bai et al. (2022): replace or augment human feedback with an **AI judge** guided by written principles (a **constitution**).

:::columns cols="2" gap="34px"
- Human labeling does **not scale** to every comparison
- **AI feedback does**: a route toward **scalable oversight**
+++
- The model critiques and revises its own outputs against the principles
- A learned judge for the open-ended tasks no verifier can check
:::

---

<!-- .slide: id="figure-bai" -->

:::figure img="images/anthropic.svg" name="Yuntao Bai and the Anthropic team" kicker="Constitutional AI: Harmlessness from AI Feedback (2022)" alt="Anthropic logo"
- Introduced **RLAIF**, a form of **rubric learning**: an AI judge scores outputs against written principles
- Showed an assistant could be made harmless with far less human labeling
- A concrete step toward **scalable oversight** of models that may exceed us
:::
