# Module 10 Exercise: Serve a model, then write an API client to it

You stand up a real inference server, then build the client side yourself: the
lecture's napkin math as code, an HTTP client that speaks the OpenAI-compatible
chat completions API, streaming with a stopwatch on it, and a concurrency
benchmark that makes batching visible.

You edit exactly one file: `exercise.py`. Everything in `src/` is provided
plumbing (the runner, the HTTP code, the threading, the plotting).

## Running

```
uv run python module_10_deployment/src/main.py
```

The runner goes through the eight steps in order. Each step's header line
carries a tag, and the step's output follows: the table or live demo your code
produced, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

The tests call your function on small hand-made inputs (numbers, dictionaries,
JSON strings, stream lines, timestamps), so none of them needs a server. Steps
1-4 print the hardware tables and the request body. Steps 5-8 also run a live
demo against a local server. With no server running, each of those steps
prints one `live demo skipped` line and is still tagged by its tests:

```
=== Step 7: latency_stats() === CORRECT
live demo skipped: no server found on port 8000, 11434, or 8080 (see README.md)
  CORRECT    returns a dict with both keys, ttft and tokens_per_second
  CORRECT    sent at 10.0, 5 tokens at 10.5, 10.6, ..., 10.9: 4 tokens in 0.4 s is 10 tokens/s
  CORRECT    a 2 s wait for the first token, then 11 tokens 0.05 s apart, still decodes at 20 tokens/s
```

Run a single step with `--step` (1 to 8):

```
uv run python module_10_deployment/src/main.py --step 6
```

`exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`. Run the finished answers with `--solution`:

```
uv run python module_10_deployment/src/main.py --solution
```

## Starting a server (for the Step 5-8 demos)

The live demos talk to a server on localhost. Start whichever one fits your
machine; they all expose the same API, which is why the client code does not
care.

**vLLM** (Linux with an NVIDIA GPU; the engine from the lecture):

```
uv tool install vllm
vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

**Ollama** (macOS, Windows, or Linux; easiest install):

```
ollama serve
ollama pull qwen2.5:0.5b-instruct
```

**llama.cpp** (any machine; the quantized-GGUF route from the lecture):

```
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The runner probes ports 8000 (vLLM), 11434 (Ollama), and 8080 (llama.cpp) and
uses the first server it finds. To point it somewhere else:

```
LLM_SERVER_URL=http://myhost:8000/v1 uv run python module_10_deployment/src/main.py
```

## The steps

| Step | Function | What it computes |
| ---- | -------- | ---------------- |
| 1 | `model_weight_bytes` | weight memory from parameter count and precision |
| 2 | `kv_cache_bytes_per_token` | how fast the cache grows with context |
| 3 | `decode_tokens_per_second` | the bandwidth speed limit on decoding |
| 4 | `build_chat_request` | the JSON body of a chat completion request |
| 5 | `extract_reply` | the generated text from a response |
| 6 | `parse_stream_line` | one token from one server-sent-event line |
| 7 | `latency_stats` | TTFT and decode rate from token timestamps |
| 8 | `total_throughput` | tokens/sec across concurrent requests |

Each function raises `NotImplementedError` until you fill in the marked line.
The runner tags that step INCOMPLETE and moves on to the next, so run it after
every step. The tests live in `tests/`, one file per step
(`test_step1_weight_bytes.py` through `test_step8_throughput.py`). Each one
calls your function on inputs with a known answer, so you can read the test for
the step you are on to see exactly what is expected.

## What you should see

Steps 1-3 print the hardware tables: model sizes at three precisions, KV cache
growth, and predicted tokens per second for four real machines. Step 4 prints
the JSON body your client will send. Step 5 sends one request and prints the
reply with its token counts. Step 6 streams the same prompt and rebuilds the
reply from the chunks, and Step 7 puts a stopwatch on that stream: time to
first token and the decode rate. Step 8 fires 1, 2, 4, and 8 concurrent requests:
total throughput climbs while each individual stream slows far less than an
8-way queue would cost it, and the figure lands in `output/throughput.png`.
That curve is batching, the economics of the entire serving business, measured
on your own machine. (Ollama users: start the server with
`OLLAMA_NUM_PARALLEL=8 ollama serve`, or it queues requests one at a time and
the curve stays flat, which is itself instructive.)

## Extra credit

- Run a quantized variant of the same model (Ollama tags like
  `qwen2.5:0.5b-instruct-q4_0`, or a smaller GGUF) and compare measured decode
  speed against the step 1 + step 3 prediction
- Send two requests that share a long common prefix and compare their TTFTs.
  Does your server do prefix caching?
- Add a system prompt (a `"system"` message before the user message) and a
  `stop` sequence to `build_chat_request` and watch the server apply the chat
  template for you. The Step 4 tests still pass with a system message in front.
- Compute a cost per million output tokens: pick a GPU rental price per hour,
  divide by your measured Step 8 throughput. How does it compare with a
  public API's price for a small model?
- Keep raising the concurrency until total throughput stops climbing. Where is
  the knee, and which resource ran out?

The answers to the eight steps are in `solution/exercise.py`. The extra credit
items are open-ended experiments, so they have no answer key.
