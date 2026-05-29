:::divider id="divider-quiz" title="Quiz" sub="Test your understanding"
:::

---

:::quiz id="quiz-q1" title="Q1: Why Nonlinearity?"
A student proposes building a "deep" network with 100 linear layers (no activation functions) to model a complex relationship.

What will this network compute, and why is the student's approach fundamentally flawed?
+++
**Answer:** The composition of any number of linear transformations is still a single linear transformation: $W_{100} \cdots W_2 W_1 \mathbf{x} = W' \mathbf{x}$. The 100-layer network has exactly the same representational power as a single layer. Without nonlinear activation functions between layers, depth adds nothing.
:::

---

:::quiz id="quiz-q2" title="Q2: The XOR Barrier"
In the exercise, the single neuron achieved 50% accuracy on XOR-like data, with the loss stuck at 0.693.

Why exactly 50%, and what is special about the number 0.693?
+++
**Answer:** 50% is random chance for a binary classification problem &mdash; the neuron cannot do better than guessing. The loss 0.693 is $\ln(2) \approx 0.6931$, which is the cross-entropy of a fair coin (maximum uncertainty for two classes). The neuron converges to predicting ~0.5 for everything because no linear boundary can improve on random guessing for XOR-structured data.
:::

---

:::quiz id="quiz-q3" title="Q3: Gradient Descent Trade-offs"
You are training a neural network and notice the loss is oscillating wildly instead of decreasing smoothly.

Name two possible causes and what you would try for each.
+++
**Answer:** (1) **Learning rate too high** &mdash; the optimizer overshoots the minimum on each step. Fix: reduce the learning rate. (2) **Batch size too small** &mdash; the gradient estimate is very noisy, causing erratic updates. Fix: increase the batch size for smoother gradients. A third possibility: the loss landscape itself is very rugged, in which case switching to Adam (which adapts per-parameter learning rates) can help.
:::

---

:::quiz id="quiz-q4" title="Q4: Backpropagation Efficiency"
GPT-4 is estimated to have over one trillion parameters.

Without backpropagation, how many forward passes would you need to estimate the gradient for a single training step? Why is this infeasible?
+++
**Answer:** You would need at least one forward pass per parameter to numerically estimate each partial derivative &mdash; over one trillion forward passes for a single update step. At any realistic speed, this would take years for one step. Backpropagation computes all gradients in a single backward pass, reducing the cost to roughly 2&ndash;3x a single forward pass regardless of parameter count.
:::

---

:::quiz id="quiz-q5" title="Q5: Sharp vs. Flat Minima"
Two networks achieve the same training loss. Network A sits in a sharp minimum; Network B sits in a flat minimum.

Which network would you expect to generalize better to new data, and why?
+++
**Answer:** Network B (flat minimum) should generalize better. In a flat minimum, small perturbations to the weights &mdash; which naturally occur when the model encounters slightly different data &mdash; do not significantly change the loss. A sharp minimum is fragile: small weight changes can dramatically increase the loss. This is also why SGD noise and small batch sizes can help generalization &mdash; the noise pushes the optimizer away from sharp minima toward flatter regions.
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
