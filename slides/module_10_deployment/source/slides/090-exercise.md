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

Then fill in the eight `NotImplementedError` lines in `module_10_deployment/exercise.py` and run after each one; unfinished steps are skipped automatically: <!-- .element: class="text-lg" -->

```bash
cd exercises
uv run python module_10_deployment/src/main.py
```

The runner probes ports 8000 (vLLM), 11434 (Ollama), and 8080 (llama.cpp) and uses the first server it finds. Steps 1-3 are pure arithmetic and run with no server at all. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: The Lecture, Measured

:::columns cols="2" gap="30px"
**You write**

- The napkin math: weight memory, KV cache growth, the speed limit
- The client: request body, reply extraction, stream parser, stopwatch, throughput
- Each blank is one line or one short expression
+++
**The payoff**

- Stage 4 fires 1, 2, 4, and 8 concurrent requests at your server
- Total throughput printed next to per-stream speed
- The machine speeds up while each stream barely pays: batching, measured on your own laptop
:::

---

:::step id="exercise-step1" title="Step 1: model_weight_bytes()"
```python
def model_weight_bytes(n_params: int, bits_per_weight: int) -> float:
    """How many bytes of memory a model's weights occupy."""
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
    """How many bytes the KV cache grows for every token in the context."""
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
    """The lecture's speed-limit formula for single-user decoding."""
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

:::terminal id="exercise-output-1" title="After Step 3: The Hardware Tables" cmd="uv run python module_10_deployment/src/main.py" caption="Actual output, no server needed: what fits where, how the cache grows, each machine's ceiling."
<span class="header">Stage 1: napkin math (no server needed)</span>
Weight memory by precision (GB):
  model             fp16    int8    int4
  Qwen2.5-0.5B       1.0     0.5     0.2
  Llama-3-8B        16.1     8.0     4.0
  Llama-3-70B      141.2    70.6    35.3

KV cache at fp16 (per token, and for one 8,192-token user):
  model           KB/token   GB @ 8K
  Qwen2.5-0.5B        12.3      0.10
  Llama-3-8B         131.1      1.07
  Llama-3-70B        327.7      2.68

Decode speed limit for Llama-3-8B, tokens/sec (bandwidth / bytes):
  machine                  GB/s    fp16    int4
  NVIDIA H100 SXM          3350     209     834
  NVIDIA RTX 4090          1008      63     251
  Apple M4 Max              546      34     136
  MacBook Air (M2)          100   (n/a)      25
server found: http://localhost:11434/v1  (model: qwen2.5:0.5b-instruct)
<span class="skipped">Stage 2: one chat completion request  [skipped: implement build_chat_request, extract_reply]</span>
<span class="skipped">Stages 3-4: streaming and the benchmark  [skipped: steps 6-8]</span>
:::

---

:::step id="exercise-step4" title="Step 4: build_chat_request()"
```python
def build_chat_request(
    model: str, prompt: str, temperature: float, max_tokens: int
) -> dict:
    """Build the JSON body for an OpenAI-compatible chat completion request."""
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
    """Pull the generated text out of a (non-streaming) chat completion response."""
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

:::terminal id="exercise-output-2" title="After Step 5: The First Real Request" cmd="uv run python module_10_deployment/src/main.py" caption="Actual output, appended to the tables above. A small local model answering over the same wire protocol the frontier labs sell. The usage line is the billing meter from the lecture."
<span class="header">Stage 2: one chat completion request</span>
POST http://localhost:11434/v1/chat/completions
prompt: Explain in two sentences why GPUs are faster than CPUs for matrix math.

reply: GPGPU (General-purpose GPU Computing) is designed to accelerate matrix
computations involving large-scale linear algebra tasks using NVIDIA's GPUs.
This makes it superior compared to traditional CPU-based algorithms, as these
machines offer an edge due to their optimized hardware acceleration
capabilities for specific applications like numerical linear algebra. [...]

usage: 18 prompt tokens in, 83 completion tokens out (this is the billing meter)

<span class="skipped">Stage 3: streaming  [skipped: implement parse_stream_line, latency_stats]</span>
<span class="skipped">Stage 4: concurrency benchmark  [skipped: implement total_throughput]</span>
:::

---

:::step id="exercise-step6" title="Step 6: parse_stream_line()"
```python
def parse_stream_line(line: str) -> str | None:
    """Turn one line of a streaming response into its piece of text, or None."""
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
**Hint:** `chunk["choices"][0]["delta"]` is a dict; `.get("content")` returns None on its own when the key is missing.
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
    """Compute TTFT and the decode rate from per-token arrival timestamps."""
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

:::terminal id="exercise-output-3" title="After Step 7: The Two Phases, Timed" cmd="uv run python module_10_deployment/src/main.py" caption="Actual output. The two numbers are the lecture's two phases: TTFT is prefill, the decode rate is the bandwidth-bound loop. Put the decode rate next to your stage 1 prediction for this machine."
<span class="header">Stage 3: streaming, with the stopwatch running</span>
streaming reply:

GPGs have higher processing power and memory capacity, which allows them to
handle and multiply large matrices more efficiently compared to CPUs that use
traditional double-precision arithmetic with limited instruction-size units
(ISAs). GPUs are also equipped with optimized hardware architectures optimized
for matrix operations, leading to faster computation speeds.

time to first token : 28 ms   (prefill)
decode rate         : 112.6 tokens/sec   (decode)
compare the decode rate with your stage 1 speed-limit prediction

<span class="skipped">Stage 4: concurrency benchmark  [skipped: implement total_throughput]</span>
:::

---

:::step id="exercise-step8" title="Step 8: total_throughput()"
```python
def total_throughput(token_counts: list[int], wall_seconds: float) -> float:
    """The operator's metric: tokens per second across the whole machine."""
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

:::terminal id="exercise-output-4" title="After Step 8: Batching, Live" cmd="uv run python module_10_deployment/src/main.py" caption="Actual output on a MacBook running Ollama with OLLAMA_NUM_PARALLEL=8. Eight users cost each stream a 3x slowdown, not the 8x a queue would charge, and the machine produces 2.4x the tokens per second."
<span class="header">Stage 4: concurrency benchmark (batching, live)</span>
120 token budget per request; 80 in the benchmark
concurrent  total tok/s  per-stream tok/s  wall (s)
         1        110.0             114.6       0.7
         2        <span class="success">159.2</span>              83.1       1.0
         4        <span class="success">216.2</span>              55.2       1.5
         8        <span class="success">269.3</span>              35.4       2.3

saved figure: module_10_deployment/output/throughput.png
the same hardware now produces 2.4x the tokens per second.
each stream slows, but far less than a queue would cost it: one weight
read now feeds every request in the batch
:::

---

<!-- .slide: id="exercise-figure" -->

## The Curve the Whole Module Is About

<div class="img-figure">
  <img src="images/throughput.png" alt="Total throughput rising with concurrency while per-stream rate falls gently">
</div>

The blue line is what the operator sells. The orange line is what one user feels. The gap between them, divided by the hardware bill, is the price of a token. **Which point on the x-axis would you run your service at?** <!-- .element: class="text-lg" -->
