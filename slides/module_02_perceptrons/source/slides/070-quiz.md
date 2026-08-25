:::divider id="divider-quiz" title="Quiz" sub="Test your understanding"
:::

---

:::quiz id="quiz-q1" title="Q1: Why Nonlinearity?"
A student proposes building a "deep" network with 100 linear layers (no activation functions) to model a complex relationship.

What will this network compute, and why is the student's approach fundamentally flawed?
+++
**Answer:** Linear layers compose into one linear transformation: $W_{100} \cdots W_2 W_1 \mathbf{x} = W' \mathbf{x}$. The 100-layer network equals a single layer. Without nonlinear activations, depth adds nothing.
:::

---

:::quiz id="quiz-q2" title="Q2: The XOR Barrier"
In the exercise, the single neuron achieved 50% accuracy on XOR-like data, with the loss stuck at 0.693.

Why exactly 50%, and what is special about the number 0.693?
+++
**Answer:** 50% is random chance for binary classification. 0.693 is $\ln(2)$: the cross-entropy of a fair coin, maximum uncertainty for two classes. No linear boundary beats guessing on XOR, so the neuron predicts ~0.5 for everything.
:::

---

:::quiz id="quiz-q3" title="Q3: Gradient Descent Trade-offs"
You are training a neural network and notice the loss is oscillating wildly instead of decreasing smoothly.

Name two possible causes and what you would try for each.
+++
**Answer:**
1. **Learning rate too high**: overshoots the minimum each step. Fix: reduce it.
2. **Batch size too small**: noisy gradients, erratic updates. Fix: increase it.
3. (Also: a rugged loss landscape. Fix: switch to Adam.)
:::

---

:::quiz id="quiz-q4" title="Q4: Backpropagation Efficiency"
GPT-4 is estimated to have over one trillion parameters.

Without backpropagation, how many forward passes would you need to estimate the gradient for a single training step? Why is this infeasible?
+++
**Answer:** One forward pass per parameter: over one trillion passes for a single update. Years per step at any realistic speed. Backprop computes all gradients in one backward pass, roughly 2&ndash;3x the cost of a forward pass, regardless of parameter count.
:::

---

:::quiz id="quiz-q5" title="Q5: Sharp vs. Flat Minima"
Two networks achieve the same training loss. Network A sits in a sharp minimum; Network B sits in a flat minimum.

Which network would you expect to generalize better to new data, and why?
+++
**Answer:** Network B. In a flat minimum, small weight perturbations (new data acts like one) barely change the loss. A sharp minimum is fragile: small changes spike the loss. SGD noise helps generalization for the same reason: it pushes the optimizer toward flatter regions.
:::

---

<!-- .slide: id="resources" -->

## References and Further Reading

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6).
- Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533–536.
- Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
- Li, H. et al. (2018). "Visualizing the Loss Landscape of Neural Nets." *NeurIPS*.
- [MIT Intro to Deep Learning Labs](https://github.com/MITDeepLearning/introtodeeplearning) — neural nets and optimization
