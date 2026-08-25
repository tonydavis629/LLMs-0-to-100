:::divider id="divider-distributed" title="Training at Scale" sub="When the model, batch, or run no longer fits on one device"
:::

---

<!-- .slide: id="distributed-intro" -->

## Pretraining Becomes a Systems Problem

- A frontier run does not fit on one GPU: not the batch, often not even the model
- Pretraining becomes a **distributed-systems** problem; throughput matters as much as the loss curve
- The most common tool, **data parallelism**: replicate the model, split the batch, synchronize gradients

---

:::manim id="data-parallel-anim" scene="data-parallel"
:::

---

<!-- .slide: id="parallelism-kinds" -->

## Two Ways to Split the Work

:::columns cols="2" gap="34px"
**Data parallelism**

- Every GPU holds a **full copy** of the model
- Each processes a different slice of the batch
- An **all-reduce** averages gradients so replicas stay identical
- Scales the batch
+++
**Model / tensor / pipeline parallelism**

Split the **model itself** when it does not fit on one GPU:

- **Tensor**: split individual weight matrices across GPUs
- **Pipeline**: put different layers on different GPUs
:::

**FSDP** is the hybrid: data-parallel structure, but parameters, gradients, and optimizer state are sharded too. Frontier runs combine all of these.

---

:::manim id="tensor-parallel-anim" scene="tensor-parallel"
:::

---

:::manim id="fsdp-anim" scene="fsdp"
:::

---

<!-- .slide: id="scale-realities" -->

## The Realities of Scale

:::columns cols="2" gap="34px"
**Thousands of GPUs, for months**

- **Networking, storage, and scheduling** become first-order concerns
- Moving gradients can cost as much time as computing them
+++
**Failures are expected**

- Over months on thousands of machines, hardware **will** fail
- **Checkpoint and resume** is mandatory: save training state so a crash costs hours, not weeks
:::

:::note
Implementation details of distributed training are the subject of Module 9.
:::
