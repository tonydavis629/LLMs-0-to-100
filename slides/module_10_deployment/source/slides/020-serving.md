:::divider id="divider-serving" title="The Serving Problem" sub="Many users, shared hardware, a meter that never stops"
:::

---

<!-- .slide: id="serving-metrics" -->

## Four Numbers Define the Problem

Serving: many users, one API, and a hardware bill that runs whether or not anyone calls.

<div class="card-grid cols-4">
<div class="card"><h4>Time to first token</h4><p><strong>TTFT.</strong> Wait before text appears.</p></div>
<div class="card"><h4>Per-user rate</h4><p><strong>Tokens/sec, one stream.</strong> How fast the reply flows.</p></div>
<div class="card"><h4>Throughput</h4><p><strong>Tokens/sec, whole machine.</strong> All users combined.</p></div>
<div class="card"><h4>Cost per token</h4><p><strong>Hardware $/hour &divide; throughput.</strong> Every API price comes from this division.</p></div>
</div>

The first two belong to the user. The last two belong to whoever pays for the GPU. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="serving-tension" -->

## The Central Tension: Latency vs Throughput

:::columns cols="2" gap="34px"
**Make one user fast**

- Whole machine for one request
- Fastest reply, idle GPU
- Enormous cost per token
+++
**Make the machine productive**

- Pack requests together
- Far more tokens per second
- Each reply slightly slower
:::

Every technique in this lecture is a position in this trade. Ask of each: **who gets faster, and who pays?** <!-- .element: class="text-lg" style="margin-top: 14px;" -->

:::note
Streaming: the model makes tokens one at a time anyway, so send each as it is made. Decent TTFT plus a tolerable stream rate feels fast even when the full reply takes ten seconds.
:::
