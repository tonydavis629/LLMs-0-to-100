:::divider id="divider-exercise" title="Exercise" sub="Serve a model with vLLM, then write an API client to it"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Start a server first. Any OpenAI-compatible server works; the client cannot tell the difference. That is the lesson: <!-- .element: class="text-lg" -->

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct        # Linux + NVIDIA GPU
OLLAMA_NUM_PARALLEL=8 ollama serve           # macOS/Windows; then: ollama pull qwen2.5:0.5b-instruct
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf   # any machine, quantized GGUF
```

Then fill in the eight `NotImplementedError` lines in `module_10_deployment/exercise.py` and run after each one: <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_10_deployment/src/main.py

# Run a single step (1-8)
uv run python module_10_deployment/src/main.py --step 6
```

The runner probes ports 8000 (vLLM), 11434 (Ollama), and 8080 (llama.cpp) and uses the first server it finds. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code produced (a table, the request body, or a live demo), then runs the step's **tests** from `tests/`. Each test calls your function on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`

The tests feed your client hand-made dictionaries, JSON strings, stream lines, and timestamps, so none of them needs a server. With no server running, Steps 5-8 print one `live demo skipped` line and are still tagged by their tests. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_10_deployment/src/main.py --step 7" maxw="1060px" caption="Here <code>latency_stats()</code> counted all five tokens, including the first one, which belongs to prefill. The live rate still looks believable, but the tests catch the mistake: they expected 10 and got 12.5, and they point to <code>len(token_times) - 1</code>."
<span class="header">=== Step 7: latency_stats() ===</span> <span class="t-fail">INCORRECT</span>
stopwatch on the streamed reply (45 tokens):
time to first token : 351 ms   (prefill)
decode rate         : 104.1 tokens/sec   (decode)
compare the decode rate with the Step 3 speed limit for your machine
  <span class="success">CORRECT</span>    returns a dict with both keys, ttft and tokens_per_second
  <span class="t-fail">INCORRECT</span>  sent at 10.0, 5 tokens at 10.5, 10.6, ..., 10.9: 4 tokens in 0.4 s is 10 tokens/s
             expected 10 tokens/s
             got      12.5 tokens/s (token 1 belongs to prefill: count len(token_times) - 1)
  <span class="t-fail">INCORRECT</span>  a 2 s wait for the first token, then 11 tokens 0.05 s apart, still decodes at 20 tokens/s
             expected 20 tokens/s
             got      22 tokens/s (count only the len(token_times) - 1 tokens after the first)
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: The Lecture, Measured

:::columns cols="2" gap="30px"
**You write**

- The napkin math: weight memory, KV cache growth, the speed limit
- The client: request body, reply extraction, stream parser, stopwatch, throughput
- Each blank is one line or one short expression, and every step has its own tests in `tests/`
+++
**The payoff**

- Step 8 fires 1, 2, 4, and 8 concurrent requests at your server
- Total throughput printed next to per-stream speed
- The machine speeds up while each stream barely pays: batching, measured on your own laptop
:::

---

:::step id="exercise-step1" title="Step 1: model_weight_bytes()"
```python
def model_weight_bytes(n_params: int, bits_per_weight: int) -> float:
    """How many bytes of memory a model's weights occupy.

    This one multiplication decides which machines a model fits on. A 7B model
    at 16 bits per weight is 14 GB; the same model quantized to 4 bits is 3.5 GB
    and suddenly runs on a laptop. It is also the number the decode speed limit
    in step 3 divides by, because every decode step reads every weight.

    Args:
        n_params: Number of parameters (e.g. 7_000_000_000).
        bits_per_weight: Bits used to store each weight (16, 8, or 4).

    Returns:
        The size of the weights in bytes.
    """
    # TODO: Return the total bytes: parameters times bits per weight, in bytes.
    raise NotImplementedError("TODO: compute the weight memory in bytes")
```
+++
**Hint:** there are 8 bits in a byte, so divide the total bits by 8.
+++
**Answer:**

```python
return n_params * bits_per_weight / 8
```
:::

---

:::step id="exercise-step2" title="Step 2: kv_cache_bytes_per_token()"
```python
def kv_cache_bytes_per_token(
    n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_value: int
) -> float:
    """How many bytes the KV cache grows for every token in the context.

    Each layer stores one key vector and one value vector per KV head, and each
    of those vectors has head_dim entries. This number times the context length
    times the number of concurrent users is the memory the conversations take
    up, on top of the weights. It is why grouped-query attention (fewer KV
    heads) exists.

    Args:
        n_layers: Number of transformer layers.
        n_kv_heads: Number of key/value heads (32 for classic multi-head
            attention, 8 for a grouped-query model like Llama 3 8B).
        head_dim: Dimension of each head (typically 128).
        bytes_per_value: Bytes per stored number (2 for fp16).

    Returns:
        Bytes of cache per token of context.
    """
    # TODO: Return the bytes per token: keys and values, for every layer, for
    #       every KV head, head_dim numbers each.
    raise NotImplementedError("TODO: compute the KV cache bytes per token")
```
+++
**Hint:** multiply all four arguments together, then double it (one key AND one value).
+++
**Answer:**

```python
return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value
```
:::

---

:::step id="exercise-step3" title="Step 3: decode_tokens_per_second()"
```python
def decode_tokens_per_second(
    bandwidth_bytes_per_s: float, bytes_per_token: float
) -> float:
    """The lecture's speed-limit formula for single-user decoding.

    During decode the GPU must read every byte of the weights from memory to
    produce each token, so memory bandwidth divided by bytes read per token is
    the ceiling on tokens per second. No amount of extra compute raises it.
    This formula predicts real single-user speeds surprisingly well, and the
    runner will put your measured speed next to it.

    Args:
        bandwidth_bytes_per_s: Memory bandwidth (e.g. 3.35e12 for an H100).
        bytes_per_token: Bytes read per generated token (the step 1 number).

    Returns:
        The predicted maximum tokens per second.
    """
    # TODO: Return the speed limit in tokens per second.
    raise NotImplementedError("TODO: compute the decode speed limit")
```
+++
**Hint:** it is one division. Bandwidth on top.
+++
**Answer:**

```python
return bandwidth_bytes_per_s / bytes_per_token
```
:::

---

:::terminal id="exercise-output-1" title="After Step 3: The Hardware Tables" cmd="uv run python module_10_deployment/src/main.py" maxw="1060px" caption="Actual output, no server needed: what fits where, how the cache grows, each machine's ceiling."
<span class="header">=== Step 1: model_weight_bytes() ===</span> <span class="success">CORRECT</span>
Weight memory by precision (GB):
  model             fp16    int8    int4
  Qwen2.5-0.5B       1.0     0.5     0.2
  Llama-3-8B        16.1     8.0     4.0
  Llama-3-70B      141.2    70.6    35.3
  <span class="skipped">...</span>
<span class="header">=== Step 2: kv_cache_bytes_per_token() ===</span> <span class="success">CORRECT</span>
KV cache at fp16 (per token, and for one 8,192-token user):
  model           KB/token   GB @ 8K
  Qwen2.5-0.5B        12.3      0.10
  Llama-3-8B         131.1      1.07
  Llama-3-70B        327.7      2.68
  <span class="skipped">...</span>
<span class="header">=== Step 3: decode_tokens_per_second() ===</span> <span class="success">CORRECT</span>
Decode speed limit for Llama-3-8B, tokens/sec (bandwidth / bytes):
  machine                  GB/s    fp16    int4
  NVIDIA H100 SXM          3350     209     834
  NVIDIA RTX 4090          1008      63     251
  Apple M4 Max              546      34     136
  MacBook Air (M2)          100   (n/a)      25
  (n/a: the fp16 weights alone do not fit in that machine's memory)
  <span class="success">CORRECT</span>    100 GB/s of bandwidth reading 4 GB per token allows 25 tokens/s
  <span class="success">CORRECT</span>    an H100 (3,350 GB/s) reading fp16 Llama-3-8B (16.06 GB) allows about 208.6 tokens/s
  <span class="success">CORRECT</span>    twice the bandwidth doubles the speed, twice the bytes per token halves it

<span class="header">=== Step 4: build_chat_request() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: add the messages list to the request body</span>
<span class="skipped">...</span>
:::

---

:::step id="exercise-step4" title="Step 4: build_chat_request()"
```python
def build_chat_request(
    model: str, prompt: str, temperature: float, max_tokens: int
) -> dict:
    """Build the JSON body for an OpenAI-compatible chat completion request.

    This is the wire format nearly every inference server speaks: vLLM, Ollama,
    and llama.cpp all accept exactly this dictionary at /v1/chat/completions.
    The `messages` list is the chat template from Module 6 in JSON form; the
    server turns it into role tokens before the model sees it.

    Args:
        model: The model name the server exposes.
        prompt: The user's message text.
        temperature: Sampling temperature (Module 4).
        max_tokens: Cap on the decode loop.

    Returns:
        The request body as a dictionary, ready to be JSON-encoded.
    """
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # TODO: Set body["messages"] below to a list containing one message:
        #       a dictionary with role "user" and the prompt as its content.
    }
    raise NotImplementedError("TODO: add the messages list to the request body")
```
+++
**Hint:** the OpenAI format is `[{"role": ..., "content": ...}]`.
+++
**Answer:**

```python
body["messages"] = [{"role": "user", "content": prompt}]
return body
```
:::

---

:::step id="exercise-step5" title="Step 5: extract_reply()"
```python
def extract_reply(response: dict) -> str:
    """Pull the generated text out of a (non-streaming) chat completion response.

    The server replies with JSON: a list of `choices` (one unless you asked for
    more), each holding a `message` with a `role` and the generated `content`.
    The runner has already parsed the JSON into nested dictionaries and lists
    for you.

    Args:
        response: The parsed response dictionary.

    Returns:
        The reply text.
    """
    # TODO: Return the content of the message inside the first choice.
    raise NotImplementedError("TODO: extract the reply text from the response")
```
+++
**Hint:** `response["choices"]` is a list; take element 0, then its `"message"`, then that message's `"content"`.
+++
**Answer:**

```python
return response["choices"][0]["message"]["content"]
```
:::

---

:::terminal id="exercise-output-2" title="After Step 5: The First Real Request" cmd="uv run python module_10_deployment/src/main.py" maxw="1060px" caption="Actual output, with Steps 1-4 folded. A small local model answering over the same wire protocol the frontier labs sell. The usage line is the billing meter from the lecture."
<span class="header">=== Step 1: model_weight_bytes() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: kv_cache_bytes_per_token() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: decode_tokens_per_second() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_chat_request() ===</span> <span class="success">CORRECT</span>
JSON body for POST /v1/chat/completions:
  <span class="skipped">...</span>

<span class="header">=== Step 5: extract_reply() ===</span> <span class="success">CORRECT</span>
POST http://localhost:11434/v1/chat/completions  (model: qwen2.5:0.5b-instruct)
prompt: Explain in two sentences why GPUs are faster than CPUs for matrix math.

reply:
GPGPUs (Graphics Processing Units) provide many advantages over traditional
central processing units (CPUs), allowing them to perform matrix operations
much more efficiently and quickly compared to CPUs. GPUs have specialized
hardware designed for parallel processing, enabling them to handle large
<span class="skipped">...</span>

usage: 20 prompt tokens in, 93 completion tokens out (this is the billing meter)
  <span class="success">CORRECT</span>    a one-choice response with content "hi" gives "hi"
  <span class="success">CORRECT</span>    a full server response (with id, usage, finish_reason) gives just the reply text
  <span class="success">CORRECT</span>    with two choices it reads the first one: "first", not "second"

<span class="header">=== Step 6: parse_stream_line() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: extract the token text from the chunk</span>
<span class="skipped">...</span>
:::

---

:::step id="exercise-step6" title="Step 6: parse_stream_line()"
```python
def parse_stream_line(line: str) -> str | None:
    """Turn one line of a streaming response into its piece of text, or None.

    With "stream": true the server sends server-sent events: each line looks
    like `data: {...json...}`, and the stream ends with `data: [DONE]`. Each
    JSON chunk carries a `delta` (what changed since the last chunk) instead of
    a full `message`, and the delta's `content` is the newly generated text,
    usually one token. The boilerplate below strips the prefix and handles the
    end marker; you extract the text from the chunk.

    Args:
        line: One decoded line from the response stream.

    Returns:
        The new text in this chunk, or None if the line carries none.
    """
    if not line.startswith("data: "):
        return None                      # keep-alives and blank lines
    payload = line[len("data: "):].strip()
    if payload == "[DONE]":
        return None                      # end-of-stream marker, no text
    chunk = json.loads(payload)
    # TODO: Return the content string of the delta inside the first choice
    #       (and None if the delta has no "content" key).
    raise NotImplementedError("TODO: extract the token text from the chunk")
```
+++
**Hint:** walk into `"choices"`, element 0, then `"delta"`, which is a dict; its `.get` method returns None on its own when the key is missing.
+++
**Answer:**

```python
return chunk["choices"][0]["delta"].get("content")
```
:::

---

:::step id="exercise-step7" title="Step 7: latency_stats()"
```python
def latency_stats(start_time: float, token_times: list[float]) -> dict:
    """Compute TTFT and the decode rate from per-token arrival timestamps.

    The runner records the clock when it sends the request and again as each
    streamed token arrives. Two numbers summarize the user experience: time to
    first token (how long the user stares at nothing, the prefill phase), and
    tokens per second after the first token (how fast the reply streams, the
    decode phase). The TTFT line is written for you.

    Args:
        start_time: Clock reading when the request was sent.
        token_times: Clock readings when each token arrived (at least two).

    Returns:
        {"ttft": seconds, "tokens_per_second": rate after the first token}.
    """
    ttft = token_times[0] - start_time
    # TODO: Set tokens_per_second to the number of tokens generated after the
    #       first one, divided by the time those tokens took to arrive.
    raise NotImplementedError("TODO: compute the decode rate in tokens per second")
```
+++
**Hint:** `len(token_times) - 1` tokens arrived between `token_times[0]` and `token_times[-1]`.
+++
**Answer:**

```python
tokens_per_second = (len(token_times) - 1) / (token_times[-1] - token_times[0])
return {"ttft": ttft, "tokens_per_second": tokens_per_second}
```
:::

---

:::terminal id="exercise-output-3" title="After Step 7: The Two Phases, Timed" cmd="uv run python module_10_deployment/src/main.py" maxw="1060px" caption="Actual output. TTFT is prefill; the decode rate is the bandwidth-bound decode loop."
<span class="header">=== Step 1: model_weight_bytes() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: kv_cache_bytes_per_token() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: decode_tokens_per_second() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_chat_request() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: extract_reply() ===</span> <span class="success">CORRECT</span>
  <span class="skipped">...</span>

<span class="header">=== Step 6: parse_stream_line() ===</span> <span class="success">CORRECT</span>
streamed reply, reassembled from the text in each data: line:
GPU (Graphics Processing Unit) performance is typically much better than
that of CPUs (Central Processing Unit) when it comes to performing
<span class="skipped">...</span>
85 chunks carried text, one per generated token
  <span class="success">CORRECT</span>    a token chunk with delta {"content": " Paris"} gives " Paris", leading space kept
  <span class="success">CORRECT</span>    the closing chunk with an empty delta {} gives None
  <span class="success">CORRECT</span>    a whole 11-line stream (role chunk, 3 tokens, [DONE]) reassembles to "Hello!"

<span class="header">=== Step 7: latency_stats() ===</span> <span class="success">CORRECT</span>
stopwatch on the streamed reply (85 tokens):
time to first token : 43 ms   (prefill)
decode rate         : 92.4 tokens/sec   (decode)
compare the decode rate with the Step 3 speed limit for your machine
  <span class="success">CORRECT</span>    returns a dict with both keys, ttft and tokens_per_second
  <span class="success">CORRECT</span>    sent at 10.0, 5 tokens at 10.5, 10.6, ..., 10.9: 4 tokens in 0.4 s is 10 tokens/s
  <span class="success">CORRECT</span>    a 2 s wait for the first token, then 11 tokens 0.05 s apart, still decodes at 20 tokens/s

<span class="header">=== Step 8: total_throughput() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: compute the combined tokens per second</span>
:::

---

:::step id="exercise-step8" title="Step 8: total_throughput()"
```python
def total_throughput(token_counts: list[int], wall_seconds: float) -> float:
    """The operator's metric: tokens per second across the whole machine.

    The runner fires N requests at once, waits for all of them, and hands you
    each reply's token count plus the wall-clock time the whole batch took.
    Dividing gives the machine's combined output rate. Watching this number
    climb with N while each individual stream barely slows down is batching
    (lecture section d) happening in front of you.

    Args:
        token_counts: Generated-token count of each concurrent request.
        wall_seconds: Wall-clock time from first send to last finish.

    Returns:
        Combined tokens per second.
    """
    # TODO: Return the total number of generated tokens divided by the wall time.
    raise NotImplementedError("TODO: compute the combined tokens per second")
```
+++
**Hint:** `sum()` the counts.
+++
**Answer:**

```python
return sum(token_counts) / wall_seconds
```
:::

---

:::terminal id="exercise-output-4" title="After Step 8: Batching, Live" cmd="uv run python module_10_deployment/src/main.py" maxw="1060px" caption="Actual output on a MacBook Pro (M2 Pro) running Ollama with OLLAMA_NUM_PARALLEL=8. Eight concurrent users slow each stream about 3.7x, not the 8x a queue would charge, and the machine produces 2.2x the tokens per second."
<span class="header">=== Step 1: model_weight_bytes() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: kv_cache_bytes_per_token() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: decode_tokens_per_second() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_chat_request() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: extract_reply() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: parse_stream_line() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 7: latency_stats() ===</span> <span class="success">CORRECT</span>
  <span class="skipped">...</span>

<span class="header">=== Step 8: total_throughput() ===</span> <span class="success">CORRECT</span>
80 token budget per request
concurrent  total tok/s  per-stream tok/s  wall (s)
         1        114.9             121.3       0.7
         2        <span class="success">155.0</span>              87.1       0.9
         4        <span class="success">203.2</span>              53.6       1.6
         8        <span class="success">252.6</span>              33.1       2.5

saved figure: output/throughput.png
the same hardware now produces 2.2x the tokens per second.
each stream slows, but far less than a queue would cost it: one weight
read now feeds every request in the batch
  <span class="success">CORRECT</span>    4 requests of 80 tokens in 2.0 s is 320 / 2.0 = 160 tokens/s
  <span class="success">CORRECT</span>    uneven replies of 10, 30, and 60 tokens in 4.0 s give 100 / 4.0 = 25 tokens/s
  <span class="success">CORRECT</span>    a single request of 120 tokens in 1.5 s is 80 tokens/s
:::

---

<!-- .slide: id="exercise-figure" -->

## The Curve the Whole Module Is About

<div class="img-figure">
  <img src="images/throughput.png" alt="Total throughput rising with concurrency while per-stream rate falls gently">
</div>

The blue line is what the operator sells. The orange line is what one user feels. The gap between them, divided by the hardware bill, is the price of a token. **Which point on the x-axis would you run your service at?** <!-- .element: class="text-lg" -->
