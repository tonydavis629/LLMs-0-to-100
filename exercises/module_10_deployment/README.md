# Module 10 Exercise: Serve a model, then write an API client to it

You stand up a real inference server, then build the client side yourself: the
lecture's napkin math as code, an HTTP client that speaks the OpenAI-compatible
chat completions API, streaming with a stopwatch on it, and a concurrency
benchmark that makes batching visible.

You edit exactly one file: `exercise.py`. Everything in `src/` is provided
plumbing (the runner, the HTTP code, the threading, the plotting).

## Running

Steps 1-3 are pure arithmetic and need no server:

```
uv run python module_10_deployment/src/main.py
```

Steps 4-8 talk to a server on localhost. Start whichever one fits your machine;
they all expose the same API, which is why the client code does not care.

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

Each function raises `NotImplementedError` until you fill in the marked line;
the runner detects unfinished steps and skips the stages that need them, so
run it after every step.

## What you should see

Stage 1 prints the hardware tables: model sizes at three precisions, KV cache
growth, and predicted tokens per second for four real machines. Stage 2 sends
one request and prints the reply with its token counts. Stage 3 streams a
reply token by token with the stopwatch running and reports time to first
token and the decode rate. Stage 4 fires 1, 2, 4, and 8 concurrent requests:
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
- Add a system prompt and a `stop` sequence to `build_chat_request` and watch
  the server apply the chat template for you
- Compute a cost per million output tokens: pick a GPU rental price per hour,
  divide by your measured stage 4 throughput. How does it compare with a
  public API's price for a small model?
- Keep raising the concurrency until total throughput stops climbing. Where is
  the knee, and which resource ran out?

Answers, including the extra credit, are in `solution/`.
