:::divider id="divider-distributed" title="Training at Scale" sub="When the model, batch, or run no longer fits on one device"
:::

---

<!-- .slide: id="distributed-intro" -->

## Pretraining Becomes a Systems Problem

A frontier run does not fit on one GPU &mdash; not the batch, often not even the model. So pretraining turns into a **distributed-systems** problem, and throughput becomes as important as the loss curve.

The first and most common tool is **data parallelism**: replicate the model across GPUs, split the batch, and synchronize the gradients. The animation shows the idea.

---

:::manim id="data-parallel-anim" scene="data-parallel"
:::

---

<!-- .slide: id="parallelism-kinds" -->

## Two Ways to Split the Work

:::columns cols="2" gap="34px"
**Data parallelism**

Every GPU holds a **full copy** of the model and processes a different slice of the batch. After the backward pass, an **all-reduce** averages the gradients so all replicas stay identical. Scales the batch.
+++
**Model / tensor / pipeline parallelism**

Split the **model itself** when its parameters, activations, or layers do not fit on one GPU:

- **Tensor**: split individual weight matrices across GPUs
- **Pipeline**: put different layers on different GPUs
:::

Real frontier runs combine all of these at once.

---

<!-- .slide: id="scale-realities" -->

## The Realities of Scale

:::columns cols="2" gap="34px"
**Thousands of GPUs, for months**

At this scale, **networking, storage, and scheduling** become first-order concerns. Moving gradients between GPUs can cost as much time as computing them.
+++
**Failures are expected**

Over a months-long run on thousands of machines, hardware **will** fail. **Checkpoint and resume** is mandatory: periodically save the full training state so a crash costs hours, not weeks.
:::

:::note
The goal here is to make the scale concrete and explain why pretraining is an engineering problem. The implementation details of distributed training are the subject of Module 9.
:::
