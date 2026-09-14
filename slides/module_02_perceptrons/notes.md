# Module 2: Perceptrons and Optimization — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

## The Neuron Model

### Biological Analogy
- The artificial neuron is loosely inspired by biological neurons: dendrites receive signals (inputs), the cell body integrates them (weighted sum), and the axon fires an output (activation).
- The analogy is limited. Biological neurons communicate with spikes (timing matters), use complex dendritic computation, and exhibit plasticity mechanisms far beyond gradient descent.
- **Reference:** McCulloch, W. S. & Pitts, W. (1943). "A Logical Calculus of the Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5, 115-133.

### The Perceptron
- A single artificial neuron computes:

$$y = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

where $\mathbf{w}$ is the weight vector, $\mathbf{x}$ is the input, $b$ is the bias, and $\sigma$ is the activation function.

- **Reference:** Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6), 386-408.

### Activation Functions
- **Step function (original perceptron):** $\sigma(z) = 1$ if $z \geq 0$, else $0$. Not differentiable at $z = 0$.
- **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$. Smooth, output in $(0, 1)$, interpretable as probability. Suffers from vanishing gradients when $|z|$ is large.
- **Tanh:** $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Zero-centered, output in $(-1, 1)$. Same vanishing gradient issue.
- **ReLU:** $\sigma(z) = \max(0, z)$. Cheap to compute, avoids vanishing gradient for $z > 0$. Can suffer "dying ReLU" when neurons get stuck at zero. Derivative is $1$ for $z > 0$ and $0$ otherwise.
- **In this course:** the exercise MLP uses **ReLU on the hidden layer** and **sigmoid on the output neuron** (so the output is a probability usable with binary cross-entropy). This ReLU-hidden / sigmoid-output pairing is the standard modern choice for a binary classifier.
- **Reference:** Nair, V. & Hinton, G. E. (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines." *ICML*.

### Why Nonlinearity Matters
- Without an activation function, a composition of linear transformations is still linear: $W_2(W_1 \mathbf{x} + b_1) + b_2 = W_2 W_1 \mathbf{x} + (W_2 b_1 + b_2) = W' \mathbf{x} + b'$. Adding layers without nonlinearity gains nothing.

### Loss Functions

- **Mean Squared Error (MSE):** $L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$. Standard for regression tasks.
- **Binary Cross-Entropy (BCE):** $L = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)]$. Standard for binary classification. This is Shannon's entropy applied as a loss function: it measures the expected number of bits needed to encode the true labels under the model's predicted distribution.
- **Connection to Module 1:** Cross-entropy appeared in Module 1 as a way to evaluate n-gram models. Here the same formula becomes the training objective.

## Matrix-Graph Equivalence

- A neural network is a directed acyclic graph (DAG). Each layer's connections can be written as a weight matrix.
- Forward pass through one layer: $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$, where $W$ is the weight matrix, $\mathbf{x}$ is the input vector, $\mathbf{b}$ is the bias vector, and $\sigma$ is an elementwise nonlinearity.
- Natural graphs are often sparse. The same matrix representation applies: a zero entry means no edge, and a nonzero entry stores the edge weight. Dense neural layers are the special case where every possible connection is present.
- Matrix multiplication is $O(n^3)$ for dense $n \times n$ matrices but is embarrassingly parallel: each output element is an independent dot product. GPUs exploit this with thousands of cores executing in parallel.
- **Connection to Module 1:** Jensen Huang and NVIDIA bet that GPU computing would become essential. Neural network training is dominated by matrix multiplications, making GPUs the ideal hardware.

## XOR Problem

### Linear Separability
- A single perceptron defines a hyperplane in the input space. It can classify any linearly separable dataset: AND, OR, NAND.
- XOR is not linearly separable. No single hyperplane can separate $(0,0)$ and $(1,1)$ (output 0) from $(0,1)$ and $(1,0)$ (output 1).

### Minsky and Papert
- **Book:** Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
- Proved that single-layer perceptrons cannot compute XOR, parity, or connectedness.
- This result was widely interpreted as showing neural networks were fundamentally limited. Funding for neural network research dried up for over a decade (often called the "connectionist AI winter").
- **Note:** Minsky and Papert knew multi-layer nets could solve these problems. Their critique was that no one knew how to train multi-layer nets at the time.

### The Hidden Layer Solution
- Two neurons in a hidden layer create two linear boundaries. The output neuron combines them with AND-like logic, implementing XOR.
- This is the simplest example of why depth (multiple layers) adds computational power.

### Perceptron Convergence Theorem
- **Theorem:** If a training set is linearly separable, the perceptron learning algorithm will converge in a finite number of steps.
- **Paper:** Rosenblatt, F. (1962). *Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms*. Spartan Books.
- **Also:** Novikoff, A. B. (1962). "On convergence proofs on perceptrons." *Symposium on the Mathematical Theory of Automata*, 12, 615-622.

## Folding in High-Dimensional Space

- A single perceptron splits the input space with a hyperplane: a line in 2D, a plane in 3D, a hyperplane in higher dimensions.
- Each hidden layer applies a nonlinear transformation that "folds" the space. Points that were entangled can be separated; points that were distant can be brought together.
- With enough layers (depth), a network can separate arbitrarily complex class boundaries.
- **Visual intuition:** Imagine a sheet of paper with two colors of dots mixed together. A single fold (one hidden layer) can bring same-colored dots together. Multiple folds (deep network) can handle more complex arrangements.
- **Paper-folding slide ("Each Hidden Neuron Is One Fold"):** four static panels. A flat sheet holds the XOR corners. Folding along the diagonal $x_1 = x_2$ (neuron 1) reflects $(0,1)$ onto $(1,0)$, so the two green points coincide; folding along $x_1 + x_2 = 1$ (neuron 2) then brings $(0,0)$ onto $(1,1)$. After two folds a single straight cut (the output neuron) separates the colors. A paper fold is a reflection, so it is the absolute-value analogue of a ReLU crease: both are piecewise-linear maps whose only bend is on the line $\mathbf{w} \cdot \mathbf{x} + b = 0$. The correspondence "one hidden neuron = one crease" is the point of the slide.
- **Interactive widget (replaces the former Manim folding clip):** the slide now hosts a live `:::interactive widget="folding"` element. The left pane is the input space showing each hidden neuron's decision line $\mathbf{w}_j \cdot \mathbf{x} + b_j = 0$; the center pane shows the two-input hidden-layer graph with edge labels $w_1, w_2$ and bias labels; the right pane is the hidden space after $h_j = \mathrm{ReLU}(\mathbf{w}_j \cdot \mathbf{x} + b_j)$. Sliders for each neuron's $w_1, w_2, b$ update both the line labels and the graph edge labels. The neuron-count toggle (1/2) shows that one ReLU line cannot fold XOR apart while two folds can.

## Multi-Layer Decision Boundaries

- A ReLU network is a composition of linear pieces. Each hidden neuron defines one hyperplane ($\mathbf{w} \cdot \mathbf{x} + b = 0$); ReLU makes that neuron active on one side of the line and zero on the other.
- Combining several half-plane indicators partitions the input space into polygonal regions, so the overall decision boundary is **piecewise-linear**. It looks curved only because it is built from many short straight segments.
- More hidden units means more lines, so the boundary can wrap more tightly around a class.
- **Interactive widgets:** `:::interactive widget="singlePerceptron"` is a toy perceptron with adjustable weights $w_1, w_2$ and bias $b$. It shows the decision line $\mathbf{w} \cdot \mathbf{x} + b = 0$ on three 2D datasets (linear, XOR, moons). Sliding the controls updates the line and a real-time accuracy readout; on XOR, the line cannot reach 100%. This is used on the "A Single Neuron Is One Line" slide to crystallize the linear-classifier idea before introducing hidden layers.
- `:::interactive widget="boundaryExplorer"` trains a real MLP in JavaScript on four 2D datasets (linear, XOR, moons, spiral) with full-batch gradient descent (1200 steps, LR=0.5 with decay). Multiple random initializations are tried (3 restarts) and the best result is kept. It draws the learned decision boundary as a filled grid plus a 0.5 contour via marching squares, data scatter, and a network diagram. Used on the "Adding a Second Layer" slide (mode=depth): starts at 2 neurons in 1 hidden layer (single-layer model) so XOR initially fails (~50-75%); the user adds layers to see depth help. Controls: dataset buttons, Add neuron, Add layer, Reset, and an accuracy readout.
- **Three Inputs: The Boundary Is a Plane (`hyperplane3d` widget):** for $n$ inputs the perceptron boundary $\mathbf{w} \cdot \mathbf{x} + b = 0$ is an affine subspace of dimension $n-1$, a hyperplane, and $\mathbf{w}$ is its normal vector. The widget uses real data: Fisher's Iris (Fisher, "The Use of Multiple Measurements in Taxonomic Problems," *Annals of Eugenics*, 1936; UCI Machine Learning Repository), the 100 setosa and versicolor flowers with three of the four measurements (sepal length, petal length, petal width, in cm). A logistic-regression perceptron fitted on those three inputs (gradient descent on standardized features, then converted back to cm) gives $\mathbf{w} = (0.517, 2.347, 5.788)$, $b = -13.145$, 100% training accuracy; the two species are linearly separable. The plane is drawn clipped to the data box with labeled, ticked axes. Transformer hidden states are typically in the thousands of dimensions (768 for BERT-base, Devlin et al. 2019; 12,288 for GPT-3, Brown et al. 2020).
- **Clusters tab on the `mlpBoundary` widget:** four Gaussian green clusters on a red background, classified by a real ReLU MLP trained in the browser (`CLUSTER_MLP`: full-batch Adam, 700 steps, best of 5 seeded inits). View buttons overlay each hidden neuron's zero set $z_j(\mathbf{x}) = 0$ in input space via marching squares. For layer 1 these are straight lines; for layer $k > 1$ the pre-activation is piecewise-linear with bends only on the zero sets of layer $k-1$, so each neuron's zero set is a piecewise-linear curve built from the previous layer's pieces. Enclosing a convex cluster with one layer needs at least 3 half-planes, so four clusters need roughly 12 first-layer neurons; with two layers, 6 neurons per layer reach 100%. The claim that depth multiplies the number of linear regions while width adds to it is Montúfar, Pascanu, Cho, and Bengio, "On the Number of Linear Regions of Deep Neural Networks," NeurIPS 2014.
- **mlpBoundary widget (retained):** still shows the geometric polygon construction on a disk-in-ring dataset: each hidden unit contributes one straight cut (a half-plane $\mathbf{w}\cdot\mathbf{x}+b=0$); ANDing them encloses a convex region. With $S = \text{width}\times\text{depth}$ available pieces the boundary is a regular $S$-gon. Used on the "Many Lines Make a Curve" text slide.

## Backpropagation

### The Algorithm
- Backpropagation computes the gradient of the loss with respect to every weight in the network by applying the chain rule of calculus through the computation graph.
- **Forward pass:** Compute the output of each layer sequentially. Store intermediate values.
- **Backward pass:** Starting from the loss, compute $\frac{\partial L}{\partial w}$ for each weight $w$ by propagating gradients backward.
- **The path (shown as the diagram on the slide):** loss $\rightarrow$ output $\hat{y}$ $\rightarrow$ pre-activation $z$ $\rightarrow$ weight $w$. The chain rule multiplies one local derivative per stage: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}}\cdot\frac{\partial \hat{y}}{\partial z}\cdot\frac{\partial z}{\partial w}$. The update rule then adds the negative-gradient step: $w_{\text{new}} = w_{\text{old}} + \left(-\eta\,\frac{\partial L}{\partial w}\right)$.
- For a single neuron with sigmoid activation and BCE loss (the exercise's steps 1&ndash;4):

$$\frac{\partial L}{\partial w_j} = (\hat{y} - y) \cdot x_j$$
$$\frac{\partial L}{\partial b} = \hat{y} - y$$

This elegant simplification occurs because the derivative of sigmoid times BCE loss simplifies to the prediction error.

- **Paper:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533-536.
- **Historical note:** The chain rule application was discovered independently multiple times. Werbos (1974) described it in his PhD thesis; Rumelhart, Hinton, and Williams popularized it in 1986.

### The Chain Rule in the Backward Pass
- The chain rule is the mathematical engine of backpropagation. For a single weight $w$ deep in the network, the gradient of the loss is:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$
- Each factor is a **local derivative** computed at a single node in the computation graph. The chain rule stitches them together by multiplication.
- This is why autograd works: each node only needs to know its own derivative, not the entire network structure. The backward pass propagates the product through the graph.
- For an MLP with multiple hidden layers, the chain extends through all of them: $\frac{\partial L}{\partial w^{(1)}} = \frac{\partial L}{\partial \hat{y}} \cdot \prod_{\ell} \frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} \cdot \frac{\partial z^{(1)}}{\partial w^{(1)}}$.

### Derivatives Review

- $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ is the slope of the tangent line at $x$; $\frac{d}{dx}$ is the operator and $f'(x)$ its value. The chart uses $f(x) = 0.2(x-5)^2 + 1$ with tangents at $x = 2$ (slope $-1.2$), $x = 5$ (slope $0$, the minimum), and $x = 7$ (slope $0.8$), and shows the secant with $h = 1.5$.
- A partial derivative $\partial L / \partial w$ is the same limit with every other input held fixed. The sign of the derivative is what gradient descent uses: negative slope means increasing $w$ lowers $L$. Any calculus text covers this (e.g., Stewart, *Calculus*, ch. 2 and 14).

### Without Backprop
- The naive alternative: perturb each weight by a small $\epsilon$, recompute the loss, and estimate the gradient numerically as $\frac{L(w + \epsilon) - L(w)}{\epsilon}$. This requires $O(n)$ forward passes for $n$ weights. For modern networks with billions of parameters, this is completely infeasible.

### Computation Graphs and Autograd
- Frameworks like PyTorch build a dynamic computation graph during the forward pass. Each operation records the input values it saw and a rule for its own local derivative. The backward pass walks this graph in reverse, applying the chain rule at each node.
- The graph is built by execution (define-by-run), not declared ahead of time, so control flow such as `if` statements and Python loops is recorded exactly as it ran for that input. This is the difference from a hand derivation, which is tied to one fixed architecture and must be redone whenever the architecture changes.
- Reverse-mode automatic differentiation is neither symbolic nor numerical differentiation. It never constructs a formula for $\partial L / \partial w$, and it does not approximate with finite differences. It evaluates the exact derivative at the specific point given by the forward pass. Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (JMLR 2018) makes this three-way distinction explicitly.
- When a tensor is used in more than one downstream operation, gradients arriving from each path are summed at that node, which is the multivariate chain rule.
- Cost: one backward pass yields the gradient with respect to every parameter, roughly a constant factor (about 2x to 3x) of the forward pass cost, independent of parameter count. Finite differences require $O(n)$ forward passes for $n$ parameters.
- Reverse mode is efficient here because the loss is a single scalar output over many inputs. Forward-mode AD has the opposite cost profile and is preferred when there are few inputs and many outputs.
- PyTorch autograd reference: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (NeurIPS 2019).

## Gradient Descent

### The Update Rule
$$w_{\text{new}} = w_{\text{old}} + \left(-\eta \frac{\partial L}{\partial w}\right)$$

where $\eta$ is the learning rate.

### Stochastic Gradient Descent (SGD)
- Instead of computing the gradient over the entire dataset (batch gradient descent), SGD computes it over a random subset (mini-batch).
- Trade-offs: mini-batch gradients are noisier but much faster to compute. The noise can actually help escape local minima and saddle points.
- **Typical batch sizes:** 32, 64, 128, 256. Larger batches give smoother gradients but require more memory and may converge to sharper minima.
- **Reference:** Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent." *COMPSTAT*.

### Adam Optimizer
- Adam (Adaptive Moment Estimation) maintains per-parameter learning rates based on the first and second moments of the gradient history.
- The slide shows the four update equations. With $g_t = \nabla L(\mathbf{w}_t)$, Adam keeps two exponential moving averages, $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ (first moment, the mean of the gradient) and $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ (second moment, the mean of the squared gradient, taken elementwise). Both are initialized at zero, so for small $t$ they underestimate the true moments by a factor of $1-\beta^t$; dividing by that factor gives the bias-corrected $\hat{m}_t$ and $\hat{v}_t$. The update is $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.
- The first moment is momentum (Polyak, 1964): gradient components that flip sign from step to step average toward zero, while components that keep the same sign accumulate. The second moment is the idea from RMSProp (Tieleman and Hinton, 2012, Coursera lecture 6.5): dividing each coordinate by its own recent gradient magnitude normalizes the step so that every weight moves at roughly the learning rate $\eta$ regardless of how steep its coordinate is. Kingma and Ba's contribution is combining the two and adding the bias correction.
- Paper defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$; PyTorch's `torch.optim.Adam` uses the same values. With $\beta_2 = 0.999$ the second-moment bias correction $1-\beta_2^t$ is still only 0.63 at $t = 1000$, which is why the correction matters in practice rather than being a formality.
- **Paper:** Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
- Adam is the default optimizer in practice for most deep learning tasks.

## Loss Landscape Visualizations

- Both landscape widgets draw, at the current point, a red arrow along $\nabla L$ (the direction of steepest ascent, unit length in weight space) and a green arrow to where the next step lands: $-\eta \nabla_B L$ for SGD, $-\eta \hat{m} / (\sqrt{\hat{v}} + \epsilon)$ for Adam. $\nabla_B$ denotes the gradient estimated on one mini-batch $B$ whose size is the Batch slider; the widget models it as the true gradient plus zero-mean noise that shrinks as $1/|B|$. A green trail records every previous position.

- The loss function defines a surface in weight space. For a network with $n$ weights, this is a surface in $(n+1)$-dimensional space.
- Visualizations project this surface down to 2D or 3D using random directions or principal components.
- **Sharp vs flat minima:** Flat minima tend to generalize better because small perturbations to the weights do not significantly change the loss. Sharp minima are more sensitive to perturbation.
- **Saddle points:** In high-dimensional spaces, saddle points are far more common than local minima. At a saddle point, the gradient is zero but the Hessian has both positive and negative eigenvalues.
- **Course visualization:** the `lossLandscape` widget plots a non-convex toy surface $L(w_1,w_2)$ with multiple ridges, valleys, a saddle-like cross term, and high-frequency ripples. The displayed surface is a mini-batch estimate: lower batch sizes add more visible roughness, while larger batch sizes approach the smoother underlying loss. The surface is mouse-draggable to change azimuth and elevation; Shift+drag pans and scroll wheel zooms. A reset button randomizes the starting weights and restores the view. Students press Step to apply $\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + (-\eta \nabla_B L)$ using a noisy mini-batch gradient. The blue point marks the current position, one bright tangent line shows the local slope at that point, and the readout shows a color-coded symbolic update equation with the numeric substitution directly underneath.
- **Adam visualization:** the `adamLandscape` widget uses the same noisy non-convex loss surface but applies Adam's update rule instead of vanilla SGD. The sliders control $\eta$ (learning rate), $\beta_1$ (first moment decay), and $\beta_2$ (second moment decay). Each step updates the running first moment $\mathbf{m}$ and second moment $\mathbf{v}$, applies bias correction to get $\hat{\mathbf{m}}$ and $\hat{\mathbf{v}}$, then computes the adaptive step: $\mathbf{w} \leftarrow \mathbf{w} - \eta \hat{\mathbf{m}} / (\sqrt{\hat{\mathbf{v}}} + \epsilon)$. The tangent line still shows the local displayed-surface slope, while the readout color-codes the symbolic Adam equation and the numeric values used for the next update.
- **Paper:** Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). "Visualizing the Loss Landscape of Neural Nets." *NeurIPS*.
- **Also:** Dauphin, Y. et al. (2014). "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization." *NeurIPS*.

## Overfitting and Generalization

- A network with enough parameters can memorize any finite training set, achieving zero training loss.
- **Paper:** Zhang, C. et al. (2017). "Understanding Deep Learning Requires Rethinking Generalization." *ICLR*. Showed that deep networks can fit random labels, demonstrating that model capacity alone does not explain generalization.
- **Regularization techniques (named but not covered deeply):**
  - **Dropout:** Srivastava, N. et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*.
  - **Weight decay (L2 regularization):** Adding $\lambda \|\mathbf{w}\|^2$ to the loss penalizes large weights.

## Local Minima and Saddle Points

- A critical point (zero gradient) is a minimum only if the loss curves upward in every direction, i.e. every eigenvalue of the Hessian is positive. If the sign of each eigenvalue were an independent coin flip, a random critical point would be a minimum with probability $2^{-n}$ for $n$ weights. The heuristic is exact for random Gaussian landscapes and matches what is observed in neural networks: Dauphin, Pascanu, Gulcehre, Cho, Ganguli, and Bengio, "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization," NeurIPS 2014; Choromanska, Henaff, Mathieu, Ben Arous, and LeCun, "The Loss Surfaces of Multilayer Networks," AISTATS 2015. The practical consequence is that in high dimensions the obstacle is saddle points and plateaus, not local minima, and SGD noise plus momentum move through them.
- The right panel draws $L = w_1^2 - w_2^2$, the canonical saddle: the gradient vanishes at the origin, the surface curves up along $w_1$ and down along $w_2$, and the green arrow is the escape direction.

## Notable Figures

### Frank Rosenblatt (1928-1971)
- Psychologist at Cornell. Built the Mark I Perceptron (1958), a machine that could learn to classify visual patterns.
- **Paper:** Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6).
- The Mark I Perceptron was a physical machine with photocells, potentiometers for weights, and motor-driven weight updates.

### Marvin Minsky (1927-2016) and Seymour Papert (1928-2016)
- **Book:** Minsky, M. & Papert, S. (1969). *Perceptrons*. MIT Press.
- Their formal proof that single-layer perceptrons cannot solve XOR was widely interpreted as showing neural networks were dead ends. Minsky was co-founder of MIT's AI Lab and a leading proponent of symbolic AI.

### David Rumelhart (1942-2011), Geoffrey Hinton (b. 1947), and Ronald Williams
- **Paper:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533-536.
- Popularized backpropagation for training multi-layer networks, reviving neural network research after over a decade of winter.
- Hinton went on to pioneer deep learning and received the 2024 Nobel Prize in Physics (shared with John Hopfield) for foundational discoveries enabling machine learning with artificial neural networks.

## Exercise Notes

### Datasets
- **linear_separable.csv:** 150 samples from two Gaussian clusters centered at $(-1.5, -1.0)$ and $(1.5, 1.0)$ with standard deviations $\sigma_x = 0.8, \sigma_y = 0.7$. Generated with numpy seed 42.
- **non_linear_separable.csv:** 160 samples in an XOR-like pattern. Four Gaussian clusters at $(\pm 1.5, \pm 1.5)$ with $\sigma = 0.5$. Diagonal pairs share labels (top-left/bottom-right = class 0, top-right/bottom-left = class 1). Generated with numpy seed 42.

### Gradient Derivation for Sigmoid + BCE
For a single neuron with sigmoid activation and binary cross-entropy loss:

$$L = -[y \ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$$

where $\hat{y} = \sigma(z)$ and $z = \mathbf{w} \cdot \mathbf{x} + b$.

Using the chain rule and the fact that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

For a **single sample**, the simplification is:

$$\frac{\partial L}{\partial w_j} = (\hat{y} - y) \, x_j$$

$$\frac{\partial L}{\partial b} = \hat{y} - y$$

For a **batch** of $n$ samples, the loss and gradients are averaged:

$$\mathbf{dw} = \frac{1}{n} X^{\top}(\hat{\mathbf{y}} - \mathbf{y})$$

$$db = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

This simplification is why sigmoid + BCE is a natural pairing for binary classification. It covers steps 1&ndash;4 (the single neuron with full-batch gradient descent).

### Exercise Structure
- The single student-facing file is `exercises/module_02_perceptrons/exercise.py`. Steps: (1) `forward` single neuron (vectorized over a batch), (2) `binary_cross_entropy`, (3) `compute_gradients` (averaged over the batch), (4) `update_parameters`, (5) observe the single neuron fail on XOR (no code), (6) `relu`, (7) `MLP.forward`, extra credit `SGD.step`.
- The runner (`src/main.py`) tags each step CORRECT, INCORRECT, or INCOMPLETE on its header line, then prints the step's output and one line per test. The tests live in `tests/`, one file per step, and call the student's function on small tensors with a hand-computable answer: for example, `forward` on a 2-sample batch with $w = (1, -1)$, $b = 0.5$ gives $\sigma(-0.5) = 0.3775$ and $\sigma(-1) = 0.2689$; `binary_cross_entropy` at $p = 0.5$ must equal $\ln 2$; `compute_gradients` is checked against autograd on a random 20-sample batch; `update_parameters` with $w = (1, 1)$, $dw = (1, -1)$, $\eta = 0.5$ must give $(0.5, 1.5)$.
- Step 5 has no new code. It trains the neuron from steps 1&ndash;4 on both datasets and its tests confirm the expected outcomes: at least 95% on the linear data (actual: 149/150, 99.3%), a loss that drops to under a quarter of its starting value, and on the XOR data a final loss within 0.01 of $\ln 2 \approx 0.693$ with accuracy no better than 60% (actual: 80/160, 50.0%). The XOR checks are deliberate: a correct single neuron *must* fail here, and a student whose neuron "succeeds" on XOR has a bug.
- Step 7's tests recompute `sigmoid(output(relu(hidden(x))))` from the model's own `nn.Linear` layers with `torch.relu` and `torch.sigmoid`, so they do not depend on the student's `relu`. The extra-credit tests compare one `SGD.step()` against `torch.optim.SGD` on an `nn.Linear` layer with identical seed and gradients.
- Slide terminal outputs are captured from the runner with the exercise filled in cumulatively (steps 1&ndash;4, then through step 7, then with the extra credit). Training is seeded (`torch.manual_seed(42)`), so the numbers are reproducible.

### MLP with ReLU Hidden Layer
- `MLP.forward`: `h = relu(self.hidden(x))`, then `sigmoid(self.output(h))`. ReLU on the hidden layer, sigmoid on the output so the result is a probability for BCE.
- PyTorch autograd computes the MLP gradients. The extra credit asks students to implement the optimizer step that uses those gradients: inside `torch.no_grad()`, update each parameter with `p -= self.lr * p.grad`.
- With `hidden_size = 8`, learning rate $1.0$, 500 epochs (torch seed 42), the ReLU MLP reaches 160/160 (100%) on the XOR-like dataset, with training loss dropping to roughly 0.0024.
