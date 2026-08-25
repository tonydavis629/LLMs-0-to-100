:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-idle-gpu" title="The Idle Supercomputer"
A GPU has roughly 300 times more compute per second than bytes of memory bandwidth per second, yet it sits mostly idle while generating text for one user. Reconcile those two facts.
+++
**Short answer: decode has an arithmetic intensity of about one operation per byte, so the memory system is saturated while 299 of every 300 arithmetic slots go unused.**

- Each decoded token: every weight read from HBM, but only ~2 operations per parameter
- The workload sits far below the hardware's compute-to-bandwidth ratio
- The memory bus is full; the arithmetic units the spec sheet advertises sit idle
:::

---

:::quiz id="quiz-two-phases" title="Same Weights, Different Bottleneck"
Prefill and decode run the same weights through the same layers. Why is one compute bound and the other memory bound?
+++
**Short answer: prefill amortizes each weight read over the whole prompt; decode reads all the weights to produce one token.**

- Prefill: all prompt tokens in one parallel pass, so each weight read does a thousand tokens' worth of arithmetic. Compute bound.
- Decode: sequential, so each read serves exactly one token. Intensity near one, memory bound.
- Same weights; different number of tokens sharing each byte.
:::

---

:::quiz id="quiz-slow-start" title="Slow to Start, Fast Once Going"
A user complains the chatbot takes ages before text appears but streams quickly once it starts. Which phase is slow, and what property of their prompts would explain it?
+++
**Short answer: prefill is slow, and long prompts explain it.**

- TTFT is prefill: the whole prompt runs through the model before the first output token exists
- Prefill cost grows with prompt length; pasted documents produce exactly this signature
- Decode rate depends on bandwidth and model size, not prompt length, so the stream feels fine
- Server-side fix for repeated long prefixes: prefix caching
:::

---

:::quiz id="quiz-kv-gqa" title="The Cache and the Heads"
Estimate the KV cache per token for a model with 32 layers, 8 KV heads of dimension 128, in fp16. Why does grouped-query attention change this number a lot but barely change the parameter count?
+++
**Short answer: 2 &times; 32 &times; 8 &times; 128 &times; 2 bytes = 128 KB per token, about 1 GB for an 8K context. GQA divides the cache by the head reduction, but K/V projection weights are a small slice of total parameters.**

- The cache scales directly with KV head count: 32 heads to 8 cuts it 4x
- Parameters barely move: K/V projections are a small slice of a model dominated by FFN weights
- GQA shrinks what is stored per token, not what the model knows
:::

---

:::quiz id="quiz-free-lunch" title="Where Batching's Free Lunch Comes From"
Batching 32 users onto one GPU raises decode throughput nearly 32x. Where does that come from, and when does it stop?
+++
**Short answer: the weights were being read anyway; a batch makes one read produce 32 tokens instead of one. It stops when the workload becomes compute bound or the KV caches exhaust memory.**

- Single-user decode wastes almost all the compute the memory traffic could feed
- Batching shares each weight read across every sequence: higher arithmetic intensity, near-free tokens
- Two walls: arithmetic becomes the limit, or 32 conversations' KV caches no longer fit next to the weights
:::

---

:::quiz id="quiz-continuous" title="Why Static Batching Died"
What property of LLM requests, absent from most other web workloads, made static batching so wasteful that every serving engine replaced it within about a year?
+++
**Short answer: wildly variable and unpredictable output lengths.**

- A web request does bounded work; a generation runs anywhere from three tokens to three thousand
- Static batch: everyone waits for the longest reply while finished slots idle
- Continuous batching reschedules at every decode step, the workload's natural granularity
:::

---

:::quiz id="quiz-quant" title="4-bit Quantization"
Quantizing from fp16 to 4-bit roughly quadruples decode speed. Which formula predicts this, and what must you do before shipping the quantized model?
+++
**Short answer: the speed limit, bandwidth over bytes per token: one quarter the bytes means four times the ceiling. Before shipping, re-run the full Module 9 evaluation suite on the quantized model.**

- The speedup is pure arithmetic
- The asterisk is quality: loss is uneven, small on average, sometimes concentrated in one capability
- Average perplexity can hide a real regression
- A quantized model is a different model and ships like one: regression suite, per-task breakdown, canary
:::

---

:::quiz id="quiz-speculative" title="Verifying Is Cheaper Than Generating"
Speculative decoding uses a small model to draft k tokens and the large model to verify them. Why is verifying k tokens so much cheaper than generating them?
+++
**Short answer: verification checks all k drafts in one parallel forward pass, which is prefill-shaped; generating them would take k sequential memory-bound passes.**

- Given the drafts as input, the large model scores every position at once: one weight read amortized across all k, exactly like prefill
- Generation cannot parallelize that way: token n+1 needs token n
- The accept/reject rule preserves the output distribution exactly, so the speedup costs no quality
:::

---

:::quiz id="quiz-moe" title="Cheaper and Exactly as Expensive"
A 47B-parameter MoE activates 13B parameters per token. In what sense is it cheaper than a dense 47B model, and in what sense is it exactly as expensive?
+++
**Short answer: cheaper per token (compute and weight traffic scale with the 13B active), exactly as expensive in memory footprint (all 47B must be resident).**

- The router decides per token which experts fire, so every expert must stay reachable
- MoE converts a bandwidth problem into a capacity problem
- Hence hybrid GPU/CPU layouts (cold experts in cheap RAM), and serving math unlike a dense model of either size
:::

---

:::quiz id="quiz-buy-build" title="The Self-Hosting Spreadsheet"
Your startup serves 50 million tokens a day at a steady rate through a rented API. What numbers would you gather to decide whether to self-host?
+++
**Short answer: your current API bill per month; a candidate model that passes your Module 9 evals; its measured batched throughput on a rentable GPU; that GPU's cost per hour; and the ops cost you'd take on.**

- Core comparison, cost per million tokens: GPU $/hour &divide; batched throughput, against the API price
- Apply a utilization factor: you pay for peak capacity, the API bills per token
- Steady volume favors self-hosting; spiky load, frontier-model needs, and the ops line favor renting
:::
