:::divider id="divider-agents" title="The Future of Agents" sub="Many agents working together, and many people sharing one agent"
:::

---

<!-- .slide: id="agents-horizon" -->

## Agents Now Finish Tasks That Take Hours

METR measures the length of task, in human working time, that an agent completes half the time.

:::columns cols="2" gap="34px"
**The trend**

- From 2019 to 2025 that length doubled about every **7 months** (Kwa et al., 2025)
- METR's January 2026 update puts the doubling since 2023 at about **4 months**
- Frontier agents now handle tasks that take a person several hours
+++
**What it shifts**

- One agent in one context window stops being enough
- The limits become coordination, context management, and human oversight
:::

A measurement on software tasks, with wide error bars. Module 9's rules apply: check the task set before extrapolating. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agents-orchestrator" -->

## One Agent Becomes Many

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="A lead agent splits a task and starts four subagents. Each subagent works in its own fresh context window and returns only a short summary. The lead agent combines the summaries into the answer.">
  <rect x="380" y="20" width="240" height="60" rx="8" fill="none" stroke="#f5a623" stroke-width="3" />
  <text x="500" y="48" fill="#f5a623" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Lead agent</text>
  <text x="500" y="70" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">plans, splits, combines</text>
  <g stroke="#8892a4" stroke-width="2" fill="none">
    <path d="M 440 80 L 140 150" /><path d="M 480 80 L 380 150" /><path d="M 520 80 L 620 150" /><path d="M 560 80 L 860 150" />
  </g>
  <g fill="none" stroke="#4a9eff" stroke-width="2">
    <rect x="50" y="150" width="180" height="76" rx="6" /><rect x="290" y="150" width="180" height="76" rx="6" /><rect x="530" y="150" width="180" height="76" rx="6" /><rect x="770" y="150" width="180" height="76" rx="6" />
  </g>
  <g fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="140" y="180">subagent 1</text><text x="380" y="180">subagent 2</text><text x="620" y="180">subagent 3</text><text x="860" y="180">subagent 4</text>
  </g>
  <g fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="140" y="206">own context window</text><text x="380" y="206">own context window</text><text x="620" y="206">own context window</text><text x="860" y="206">own context window</text>
  </g>
  <text x="500" y="270" fill="#50c878" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">Each subagent returns a short summary, so the lead's context holds results, not every search.</text>
</svg>
</div>

- Anthropic's research system (2025): a lead agent with parallel subagents beat a single agent by **90%** on an internal research eval
- The same system used about **15 times** the tokens of a chat. On BrowseComp, token usage alone explained 80% of the variance in scores
- Parallel agents are a way to spend more inference compute, the same axis as reasoning tokens

---

<!-- .slide: id="agents-swarm" -->

## Kimi K2.5 Was Trained to Run a Swarm

<div class="svg-figure">
<svg viewBox="0 0 1000 290" role="img" aria-label="An execution graph. The orchestrator spends two steps planning, then four subagents run in parallel for three, six, two, and four steps, then the orchestrator spends two steps combining. The critical path is the planning steps, the longest subagent, and the combining steps: ten steps, while the total work is nineteen steps.">
  <text x="20" y="30" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">time, in agent steps</text>
  <path d="M 20 44 L 970 44" stroke="#8892a4" stroke-width="1.5" fill="none" />
  <path d="M 978 44 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="20" y="120" width="140" height="44" fill="#f5a623" fill-opacity="0.25" stroke="#f5a623" stroke-width="3" />
  <text x="90" y="148" fill="#f5a623" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">plan: 2</text>
  <g fill="none" stroke="#4a9eff" stroke-width="2">
    <rect x="170" y="62" width="210" height="36" /><rect x="170" y="166" width="140" height="36" /><rect x="170" y="218" width="280" height="36" />
  </g>
  <rect x="170" y="114" width="420" height="36" fill="#50c878" fill-opacity="0.25" stroke="#50c878" stroke-width="3" />
  <g fill="#e8eaf0" font-size="16" font-family="Inter, sans-serif">
    <text x="182" y="86">subagent A: 3</text><text x="182" y="138">subagent B: 6 (longest)</text><text x="182" y="190">subagent C: 2</text><text x="182" y="242">subagent D: 4</text>
  </g>
  <rect x="600" y="120" width="140" height="44" fill="#f5a623" fill-opacity="0.25" stroke="#f5a623" stroke-width="3" />
  <text x="670" y="148" fill="#f5a623" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">combine: 2</text>
  <text x="770" y="120" fill="#50c878" font-size="19" font-family="Inter, sans-serif">critical path: 10 steps</text>
  <text x="770" y="150" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">total work: 19 steps</text>
  <text x="770" y="190" fill="#8892a4" font-size="16" font-family="Inter, sans-serif">training rewards the</text>
  <text x="770" y="212" fill="#8892a4" font-size="16" font-family="Inter, sans-serif">short critical path</text>
</svg>
</div>

- **PARL** (Moonshot AI, 2026): RL updates only the orchestrator. Subagents are frozen copies of earlier checkpoints
- The reward counts the **critical path**, the longest chain of dependent steps, rather than total steps
- Two failure modes had to be trained away: falling back to doing everything in sequence, and spawning useless subagents to game the reward
- Up to 100 subagents. BrowseComp rose from 60.6% with one agent to 78.4% with the swarm

---

<!-- .slide: id="agents-when" -->

## When Splitting the Work Helps

:::columns cols="2" gap="34px"
**Where splitting helps**

- The parts are independent: search many sources, review many files
- Each subagent's context stays small and on topic
- Wall-clock time drops with the critical path
+++
**Where splitting hurts**

- The parts depend on each other's decisions, as in most code edits
- Subagents make conflicting assumptions they never see
- Cognition (2025) argued for one agent with one shared context for this reason
:::

Cemri et al. (2025) sorted failures in multi-agent systems into three groups: unclear task specifications, agents misaligned with each other, and weak checking of the final result. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agents-multiplayer" -->

## Many People Share One Session

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="One shared session. A product manager asks the agent to draft a launch plan. An engineer adds that the database migration must come first. A designer, talking to the engineer and not the agent, asks about the timeline. The agent revises the plan to put the migration first and stays quiet on the side conversation.">
  <rect x="20" y="14" width="960" height="272" rx="10" fill="none" stroke="#2a3450" stroke-width="2" />
  <text x="44" y="44" fill="#8892a4" font-size="17" font-family="Inter, sans-serif">one shared context window</text>
  <text x="44" y="88" fill="#f5a623" font-size="18" font-family="Inter, sans-serif">product manager</text>
  <text x="250" y="88" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">@agent draft the launch plan for Friday</text>
  <text x="44" y="132" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif">engineer</text>
  <text x="250" y="132" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">the database migration has to ship first</text>
  <text x="44" y="176" fill="#a06bd4" font-size="18" font-family="Inter, sans-serif">designer</text>
  <text x="250" y="176" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">to the engineer: is Friday still realistic?</text>
  <text x="44" y="220" fill="#50c878" font-size="18" font-family="Inter, sans-serif">agent</text>
  <text x="250" y="220" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">revised plan: migration Wednesday, launch Friday</text>
  <text x="250" y="262" fill="#8892a4" font-size="16" font-family="Inter, sans-serif">It took the engineer's constraint and did not answer the question meant for someone else.</text>
</svg>
</div>

- Products now ship this: ChatGPT group chats (2025) and Claude working inside shared Slack channels (2026)
- Chat models were trained on one user and one assistant taking turns. A group breaks that assumption

---

<!-- .slide: id="agents-multiplayer-problems" -->

## Shared Sessions Raise New Questions

<div class="bench-table">
<table>
<thead><tr><th>Question</th><th>Why it is hard</th></tr></thead>
<tbody>
<tr><td><strong>Whose instruction wins?</strong></td><td>Two people give conflicting requests. The system prompt from Module 11 outranks the user, but nothing ranks one user against another</td></tr>
<tr><td><strong>When should the agent speak?</strong></td><td>Most messages in a group are not addressed to the agent. Answering everything is noise; staying silent misses requests</td></tr>
<tr><td><strong>Who sees what?</strong></td><td>One person's private memory or files should not leak to the group. ChatGPT group chats leave personal memory out entirely</td></tr>
<tr><td><strong>Who approved the action?</strong></td><td>An agent that deploys code on a group's say-so needs a record of which person authorized it</td></tr>
</tbody>
</table>
</div>

None of these is a modeling problem alone. Each one is also a product and permissions problem. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
