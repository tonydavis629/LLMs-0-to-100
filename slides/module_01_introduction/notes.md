# Module 1: Course Introduction — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

## Information Theory

### Shannon's "A Mathematical Theory of Communication" (1948)

- **Paper:** Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27, 379-423 and 623-656.
- **Claim:** "Shannon proved communication could be quantified in bits."
  - Shannon defined information entropy as:
    $$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$
  - In source coding, this is the lower bound on the average number of bits per symbol for an optimal code. A uniform distribution over N symbols gives H = log2(N) bits (maximum entropy). A perfectly predictable source has H = 0.
  - Shannon also proved the noisy-channel coding theorem: reliable communication is possible over a noisy channel at any rate below channel capacity, with arbitrarily small error probability.
  - Shannon explicitly says base-2 logarithms give units that "may be called binary digits, or more briefly bits," noting that the word "bit" was suggested by J. W. Tukey.

### Shannon's Prediction Experiment

- **Paper:** Shannon, C. E. (1951). "Prediction and Entropy of Printed English." *Bell System Technical Journal*, 30(1), 50-64.
- Shannon asked human subjects to guess the next character in a text. Using those prediction experiments, he estimated English entropy to be of the order of 1 bit/letter, with experimental lower and upper bounds reaching roughly 0.6-1.3 bits/letter for long contexts.
- In the same paper, Shannon reports the classic 26-letter approximations (spaces and punctuation ignored): zero-order = log2(26) = 4.70 bits/letter, unigram = 4.14, bigram = 3.56, trigram = about 3.3.
- The "PROBABLY _" bar chart in the animation is a pedagogical, hand-chosen distribution to illustrate how context sharpens prediction. It should not be cited as a published Shannon table or as a Norvig-derived empirical result.

### Entropy Equation

The Shannon entropy formula used in the "Language as Compression" slide:

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

- **Uniform encoding:** The slide uses 5 bits per character (since ceil(log2(27)) = 5 for a fixed-width code), giving 22 * 5 = 110 bits total.
- **Variable-length prefix code:** The slide uses a pedagogical prefix code based on frequency counts in "the cat sat on the mat" (22 chars total, 10 unique). The code assigns: ' '=00, t=01, a=100, e=101, h=110, n=1110, o=11110, s=111110, c=1111110, m=1111111. Total bits: 70. This is a valid prefix code (no code is a prefix of another) but not the optimal Huffman code for these frequencies. A true Huffman tree would produce roughly 66 bits. The unary-like structure was chosen because it clearly illustrates the principle (short codes for common symbols, long codes for rare ones) and the actual bit strings are easy to read.
- **Why variable-length blocks are smaller in total:** Each character's block width is proportional to its code length. Common characters (' ' and 't', 2 bits each) have narrow blocks; rare characters ('c' and 'm', 7 bits each) have wide blocks. But because common characters appear many more times, the total width (bits) is dominated by the short codes. The overall savings: 110 bits → 70 bits (36% reduction).

### Cross-entropy and Perplexity

- **Cross-entropy** (theoretical definition): $H(p, q) = -\sum_x p(x) \log_2 q(x)$, measuring how many bits model $q$ needs on average to encode symbols from the true distribution $p$.
- **Empirical cross-entropy** (what we compute in practice): $\hat{H}(p, q) = -\frac{1}{N} \sum_{i=1}^{N} \log_2 q(c_i \mid \text{context}_i)$. This averages the negative log-likelihood over the actual observed characters. By the law of large numbers, this converges to the true cross-entropy as $N \to \infty$. The true distribution $p$ does not appear explicitly because we are summing over the actual data (samples from $p$).
- **Perplexity** = 2^H(p,q). Interpretation: the model is as uncertain as if it were choosing uniformly from this many options.
- The current bar-chart labels should be treated as **illustrative**, not as one consistent measured experiment. In particular, they mix a 27-symbol uniform baseline with Shannon's 1951 26-letter published values, and the trigram value does not match Shannon's published trigram estimate.
- If you want **published Shannon values** (26 letters, no spaces/punctuation), use: 4.70 / 4.14 / 3.56 / about 3.3 bits per letter for zero-order / unigram / bigram / trigram.
- If you want **repo-specific measured values** from the current solution on the raw Gutenberg-stripped `alice.txt`, the current `cross_entropy()` implementation gives about 3.45 bits/char for the bigram model and 2.56 bits/char for the trigram model. A unigram baseline would need a separate calculation because the current helper expects a context-conditioned model.

## N-gram Language Models

### Chain Rule of Probability

The chain rule decomposes the joint probability of a sequence into a product of conditional probabilities:

$$P(w_1 w_2 \ldots w_n) = \prod_{k=1}^{n} P(w_k \mid w_1 \ldots w_{k-1})$$

This is an exact identity, not an approximation. The problem is that estimating $P(w_k \mid w_1 \ldots w_{k-1})$ requires observing every possible history, which is infeasible for long sequences.

- **Reference:** Jurafsky & Martin, *Speech and Language Processing*, 3rd ed., Ch. 3, Eq. 3.3-3.4.

### The Markov Assumption

The n-gram model approximates the full history with just the last $(n-1)$ symbols:

$$P(w_k \mid w_1 \ldots w_{k-1}) \approx P(w_k \mid w_{k-n+1} \ldots w_{k-1})$$

A bigram (n=2) conditions only on the immediately preceding word. A trigram (n=3) conditions on the two preceding words. This is called the Markov assumption: the future depends only on a fixed-length window of the past, not the entire history.

- **Named after:** Andrei Markov, who studied sequences with this property in the early 1900s.
- **Reference:** Jurafsky & Martin, Ch. 3, Section 3.1.1.
- **The Walden Pond example** in the slides is adapted from Jurafsky & Martin: instead of $P(\text{blue} \mid \text{The water of Walden Pond is so beautifully})$, a bigram approximates this as $P(\text{blue} \mid \text{beautifully})$.

### Definition

An n-gram model estimates P(next_symbol | context) where the context is the preceding (n-1) symbols:

$$P(w_t \mid w_{t-n+1}, \ldots, w_{t-1}) = \frac{\text{count}(w_{t-n+1} \ldots w_t)}{\text{count}(w_{t-n+1} \ldots w_{t-1})}$$

- **Reference:** Jurafsky, D. & Martin, J. H. (2024). *Speech and Language Processing*, 3rd ed., Chapter 3. Available at https://web.stanford.edu/~jurafsky/slp3/
- Shannon's 1948 paper describes this exact procedure for generating random text by order of approximation (Section 3, pp. 388-389).

### Sample Outputs

The sample outputs in the exercise slides were generated from the reference solution with `random.seed(42)` on `data/alice.txt` (144,603 characters). Some slide snippets are cropped/truncated slightly for readability, but they do come from the real generated output.

## GOFAI ("Good Old-Fashioned AI")

- **Term associated with / popularized by:** Haugeland, J. (1985). *Artificial Intelligence: The Very Idea*. MIT Press.
- **Expert systems era:** Feigenbaum, E. A. (1977). "The Art of Artificial Intelligence." IJCAI-77.
- **Lighthill Report (1973):** Lighthill, J. (1973). "Artificial Intelligence: A General Survey." UK Science Research Council. Triggered the first major AI funding cut in the UK.

## ELIZA

- **Paper:** Weizenbaum, J. (1966). "ELIZA -- A Computer Program for the Study of Natural Language Communication Between Man and Machine." *Communications of the ACM*, 9(1), 36-45.
- **The ELIZA effect:** People's tendency to attribute human-like understanding to computer programs. The term itself is later than the 1966 paper; the paper supports the broader point that users readily engaged with ELIZA as if it understood them.
- **Try it:** https://www.masswerk.at/elizabot/ (a faithful web recreation by Norbert Landsteiner)

## Notable Figures

### Grace Hopper (1906-1992)
- Created the A-0 System (1952), often described as the first compiler; more precisely, it automated subroutine selection/linking and is a direct ancestor of later compilers. See: Beyer, K. W. (2009). *Grace Hopper and the Invention of the Information Age*. MIT Press.
- COBOL development: Conference on Data Systems Languages (CODASYL), 1959.

### Claude Shannon (1916-2001)
- See papers cited above.
- Also: Gleick, J. (2011). *The Information: A History, a Theory, a Flood*. Pantheon Books.

### John von Neumann (1903-1957)
- Von Neumann architecture: Von Neumann, J. (1945). "First Draft of a Report on the EDVAC."
- Goldstine, H. H. (1972). *The Computer from Pascal to von Neumann*. Princeton University Press.

### Jensen Huang (b. 1963)
- CUDA: Nickolls, J. et al. (2008). "Scalable Parallel Programming with CUDA." *ACM Queue*, 6(2).
- NVIDIA's role in deep learning: Raina, R. et al. (2009). "Large-scale Deep Unsupervised Learning using Graphics Processors." ICML.

## Deep Learning

### AlexNet
- **Paper:** Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.
- Error rate drop: 26.2% (second-best entry) to 15.3% (AlexNet ensemble) on ImageNet LSVRC-2012 top-5 error.

### Backpropagation
- **Paper:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533-536.

## Transformers and LLMs

### "Attention Is All You Need"
- **Paper:** Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.

### BERT and GPT
- **BERT:** Devlin, J. et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT 2019*.
- **GPT:** Radford, A. et al. (2018). "Improving Language Understanding by Generative Pre-Training." OpenAI.

### InstructGPT and RLHF
- **Paper:** Ouyang, L. et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*.

## AI Summers and Winters Chart

The chart is illustrative (not based on a specific dataset) but reflects documented funding cycles:
- **Dartmouth Summer (1956):** McCarthy, J. et al. (1955). "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence."
- **First AI Winter (~1974):** Triggered by the Lighthill Report (1973) and DARPA funding cuts.
- **Expert Systems Boom (1980s):** Feigenbaum & McCorduck (1983). *The Fifth Generation*.
- **Second AI Winter (~1988-1993):** Collapse of the Lisp machine market and Japanese Fifth Generation project.
- **Deep Learning Revival (2012+):** Triggered by AlexNet (see above).
- **LLM Era (2022+):** ChatGPT was released on November 30, 2022; Reuters reported an estimate that it reached about 100 million monthly active users by January 2023.
