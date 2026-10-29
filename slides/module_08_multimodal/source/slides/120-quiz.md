:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-position" title="Why Patches Need Positions"
Why does a transformer need positional information **after** an image is split into patches?
+++
**Short answer: flattening a 2-D grid into a 1-D sequence discards where each patch sat, and self-attention is permutation-invariant, so without positions the model cannot tell top from bottom.**

- Attention treats its inputs as an unordered set: same result in any patch order
- Flattening removes the 2-D layout (which patch is above which) from the values
- Position embeddings put it back: "the triangle is on top," not just "a triangle and a square are present"
:::

---

:::quiz id="quiz-token-types" title="Three Things Called Tokens"
What is the difference between a **text token**, a **patch embedding**, and a **discrete image token**?
+++
**Short answer: a text token and a discrete image token are indices into a fixed vocabulary/codebook; a patch embedding is a continuous vector, not an index.**

- **Text token**: a discrete id looked up in an embedding table
- **Discrete image token** (VQ-VAE style): an index into a learned **codebook** of visual patterns
- **Patch embedding**: a **continuous vector** projected from raw pixels; no vocabulary, no lookup
- "Visual token" is ambiguous; only the discrete code is a token in the word sense
:::

---

:::quiz id="quiz-vq" title="Continuous or Discrete?"
You want one model that can both describe a photo and draw a new one. Why would you tokenize images with a VQ-VAE instead of feeding it projected ViT patch embeddings, and what do you give up?
+++
**Short answer: generating an image means predicting tokens with a softmax, which needs a finite vocabulary; VQ codebook ids provide one. The price is detail lost to quantization.**

- A projected patch embedding can be read by the LLM, but there is no list of possible patch vectors to choose from at the output
- VQ ids join the text vocabulary, so the same next-token loss trains both reading and writing images
- Each region is forced onto one of $K$ codes, so fine detail such as small text can be lost
- For understanding-only tasks, continuous features usually keep more of the image
:::

---

:::quiz id="quiz-clip-diagonal" title="CLIP's Diagonal"
In CLIP, why does the similarity matrix have exactly **one correct caption per image** and **one correct image per caption**?
+++
**Short answer: the batch is constructed from matched pairs, so image i belongs with caption i and nothing else in the batch; the correct answers are the diagonal.**

- The batch is $B$ **matched** pairs: image $i$'s caption is caption $i$; every other caption is a presumed negative
- **Rows**: image-to-text classification, correct class on the diagonal. **Columns**: text-to-image, same diagonal
- So the targets are $0..B{-}1$ and the loss is ordinary cross-entropy in both directions
- Caveat: two images sharing a caption create a false negative
:::

---

:::quiz id="quiz-zeroshot" title="Zero-Shot Without a Head"
Why can CLIP do zero-shot classification even though it was **never trained with a fixed classifier head**?
+++
**Short answer: its "classifier" is text, so any set of class names becomes a classifier by embedding the names and comparing to the image embedding.**

- CLIP learned a **shared space** where images sit near their descriptions
- Write one prompt per class, embed each, embed the image, pick the nearest prompt
- Supervision came from **natural language**, so classes are open-ended: define them at inference time, swap label sets without touching the weights
:::

---

:::quiz id="quiz-projector" title="What the Projector Learns"
What does the **projector** between a vision encoder and an LLM actually learn?
+++
**Short answer: a coordinate transformation from the vision encoder's feature space into the LLM's token-embedding space, so a visual vector "looks like" something the LLM can attend to.**

- The encoder and LLM were (often) trained separately, so their spaces are unrelated: a direction meaning "red" to the encoder means nothing to the LLM
- The projector learns the **mapping between the two spaces**, placing each visual feature where the LLM expects meaning, at the LLM's width
- Small in parameters but decisive: a bad projector lands visual vectors in nonsense directions, leaving the encoder **unreadable**
:::

---

:::quiz id="quiz-nexttoken" title="Still Next-Token"
Why is image-conditioned captioning still trained with a **next-token loss over text**?
+++
**Short answer: the output is still a text sequence, so the objective is unchanged; the image only enters as extra context in the prefix.**

- The model still produces **words** one after another, the setup of pretraining and SFT
- The image changes what the model **conditions on**, not what it predicts
- The visual prefix sits in the context like extra tokens; attention mixes it into the hidden states
- Same masked cross-entropy over response tokens: $-\sum_{t\in R}\log p_\theta(x_t\mid \text{image}, x_{<t})$
:::

---

:::quiz id="quiz-priors" title="Answering From Priors"
What failure would suggest the model is answering from **language priors** instead of using the image?
+++
**Short answer: the answer stays the same (or tracks the question's wording) when you change the image, or it matches the dataset's most common answer rather than the picture.**

- Swap the image and "describe the image" returns the **same** caption: the image is not being used
- Answering "yellow" for a green banana exploits what bananas **usually** are in training data
- The clean test is **controlled, balanced** data: every color and shape equally likely, so only reading the image can be right
- Then a correct answer is evidence of grounding; a constant answer exposes the shortcut
:::

---

:::quiz id="quiz-perception" title="Perception or Reasoning?"
A model looks at a photo of a parking lot and answers "How many red cars are there?" with **3**. There are **4**. Describe one experiment that tells you whether it failed at **perception** or at **reasoning**.
+++
**Short answer: remove the perception step. Give the model the same question with the scene described in text (or ask it to list every car and its color first). If it now counts 4, the miss was perception; if it still says 3, the counting itself is wrong.**

- Counting has two steps: find each red car (perception), then count them (reasoning)
- A text description or a listed inventory isolates the second step
- Also try a higher-resolution crop: if the fourth car was tiny or cut off, the pixels never reached the encoder
- The two failures need different fixes, so a single accuracy number cannot tell you what to change
:::
