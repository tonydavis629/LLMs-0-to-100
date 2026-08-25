"""
Module 10 Exercise runner: serve a model, then write an API client to it

Run with:
    uv run python module_10_deployment/src/main.py

Stage 1 (no server needed) prints the napkin-math tables from steps 1-3: weight
memory, KV cache growth, and the decode speed limit for real machines. Stages
2-4 need an OpenAI-compatible server running locally (vLLM, Ollama, or
llama.cpp; see the README): one request, then a streamed request with latency
measurement, then a concurrency benchmark that shows batching working.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one function at a time and re-run immediately.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided plotting helper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (  # noqa: E402  (import after sys.path edits)
    model_weight_bytes,
    kv_cache_bytes_per_token,
    decode_tokens_per_second,
    build_chat_request,
    extract_reply,
    parse_stream_line,
    latency_stats,
    total_throughput,
)
from visualization import save_throughput_figure  # noqa: E402


# ---------------------------------------------------------------------------
# The protocol. Every one of these knobs moves the measurements, which is why
# a serving benchmark states them before it reports any number.
# ---------------------------------------------------------------------------
TEMPERATURE = 0.7        # sampling temperature for every request
MAX_TOKENS = 120         # decode budget per request
BENCH_MAX_TOKENS = 80    # decode budget per benchmark request
CONCURRENCY_LEVELS = [1, 2, 4, 8]
PROMPT = "Explain in two sentences why GPUs are faster than CPUs for matrix math."
BENCH_PROMPTS = [
    "Describe the water cycle in three sentences.",
    "Explain what a hash table is in three sentences.",
    "Summarize how photosynthesis works in three sentences.",
    "Explain what latency means in networking, in three sentences.",
    "Describe how a compiler differs from an interpreter, in three sentences.",
    "Explain why the sky is blue in three sentences.",
    "Describe what version control is for, in three sentences.",
    "Explain what a prime number is in three sentences.",
]

# Where to look for a server. The first URL that answers GET /models wins.
# vLLM serves on 8000, Ollama on 11434, llama.cpp's llama-server on 8080;
# all three speak the same OpenAI-compatible API, which is the point.
SERVER_CANDIDATES = [
    os.environ.get("LLM_SERVER_URL"),
    "http://localhost:8000/v1",
    "http://localhost:11434/v1",
    "http://localhost:8080/v1",
]

_THIS_DIR = Path(__file__).resolve().parent
GB = 1e9


def _find_data_file(name: str) -> Path:
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def _is_implemented(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception still means the student wrote *something*.
        return True


_PROBE_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
_PROBE_LINE = 'data: {"choices": [{"delta": {"content": "hi"}}]}'


def _probe_steps() -> dict[str, bool]:
    """Detect which exercise.py steps are implemented, using throwaway inputs."""
    return {
        "model_weight_bytes": _is_implemented(model_weight_bytes, 1000, 16),
        "kv_cache_bytes_per_token": _is_implemented(kv_cache_bytes_per_token, 2, 2, 4, 2),
        "decode_tokens_per_second": _is_implemented(decode_tokens_per_second, 1e12, 1e9),
        "build_chat_request": _is_implemented(build_chat_request, "m", "hi", 0.7, 8),
        "extract_reply": _is_implemented(extract_reply, _PROBE_RESPONSE),
        "parse_stream_line": _is_implemented(parse_stream_line, _PROBE_LINE),
        "latency_stats": _is_implemented(latency_stats, 0.0, [0.5, 0.6, 0.7]),
        "total_throughput": _is_implemented(total_throughput, [10, 10], 2.0),
    }


def _heading(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def _skip(stage: str, missing: list[str]) -> None:
    print(f"[skipped] {stage}: implement {', '.join(missing)} in exercise.py first\n")


# ---------------------------------------------------------------------------
# HTTP plumbing (provided). Only the standard library, so there is nothing to
# install: urllib POSTs the JSON body and hands back the raw response.
# ---------------------------------------------------------------------------


def _get_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, body: dict, timeout: float = 300.0):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _discover_server() -> tuple[str, str] | None:
    """Return (base_url, model_name) for the first reachable server, else None."""
    for base in SERVER_CANDIDATES:
        if not base:
            continue
        base = base.rstrip("/")
        try:
            listing = _get_json(base + "/models")
            return base, listing["data"][0]["id"]
        except Exception:
            pass
        # Older Ollama versions answer chat completions but not /v1/models;
        # their native /api/tags endpoint lists the pulled models instead.
        try:
            root = base.rsplit("/v1", 1)[0]
            tags = _get_json(root + "/api/tags")
            return base, tags["models"][0]["name"]
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Stage 1: napkin math (steps 1-3, no server)
# ---------------------------------------------------------------------------


def stage_napkin_math(steps: dict[str, bool]) -> None:
    _heading("Stage 1: napkin math (no server needed)")
    models = json.loads(_find_data_file("models.json").read_text())
    machines = json.loads(_find_data_file("machines.json").read_text())

    if not steps["model_weight_bytes"]:
        _skip("weight memory table", ["model_weight_bytes"])
    else:
        print("Weight memory by precision (GB):")
        print(f"  {'model':<14}{'fp16':>8}{'int8':>8}{'int4':>8}")
        for m in models:
            sizes = [model_weight_bytes(m["params"], bits) / GB for bits in (16, 8, 4)]
            print(f"  {m['name']:<14}" + "".join(f"{s:>8.1f}" for s in sizes))
        print()

    if not steps["kv_cache_bytes_per_token"]:
        _skip("KV cache table", ["kv_cache_bytes_per_token"])
    else:
        print("KV cache at fp16 (per token, and for one 8,192-token user):")
        print(f"  {'model':<14}{'KB/token':>10}{'GB @ 8K':>10}")
        for m in models:
            per_token = kv_cache_bytes_per_token(
                m["layers"], m["kv_heads"], m["head_dim"], 2
            )
            print(f"  {m['name']:<14}{per_token / 1e3:>10.1f}{per_token * 8192 / GB:>10.2f}")
        print()

    needed = ["model_weight_bytes", "decode_tokens_per_second"]
    if not all(steps[s] for s in needed):
        _skip("speed-limit table", [s for s in needed if not steps[s]])
    else:
        print("Decode speed limit for Llama-3-8B, tokens/sec (bandwidth / bytes):")
        print(f"  {'machine':<22}{'GB/s':>7}{'fp16':>8}{'int4':>8}")
        llama = next(m for m in models if m["name"] == "Llama-3-8B")
        for mach in machines:
            bw = mach["bandwidth_gb_s"] * GB
            fp16 = decode_tokens_per_second(bw, model_weight_bytes(llama["params"], 16))
            int4 = decode_tokens_per_second(bw, model_weight_bytes(llama["params"], 4))
            fits16 = model_weight_bytes(llama["params"], 16) / GB <= mach["memory_gb"]
            fp16_cell = f"{fp16:>8.0f}" if fits16 else "   (n/a)"
            print(f"  {mach['name']:<22}{mach['bandwidth_gb_s']:>7.0f}{fp16_cell}{int4:>8.0f}")
        print("  (n/a: the fp16 weights alone do not fit in that machine's memory)")
        print()


# ---------------------------------------------------------------------------
# Stage 2: one request (steps 4-5)
# ---------------------------------------------------------------------------


def stage_single_request(steps: dict[str, bool], base: str, model: str) -> None:
    _heading("Stage 2: one chat completion request")
    needed = ["build_chat_request", "extract_reply"]
    if not all(steps[s] for s in needed):
        _skip("single request", [s for s in needed if not steps[s]])
        return
    body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
    print(f"POST {base}/chat/completions")
    print(f"prompt: {PROMPT}")
    with _post_json(base + "/chat/completions", body) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    print(f"\nreply: {extract_reply(response).strip()}")
    usage = response.get("usage") or {}
    print(
        f"\nusage: {usage.get('prompt_tokens', '?')} prompt tokens in, "
        f"{usage.get('completion_tokens', '?')} completion tokens out "
        "(this is the billing meter)\n"
    )


# ---------------------------------------------------------------------------
# Stage 3: streaming with latency measurement (steps 4, 6, 7)
# ---------------------------------------------------------------------------


def _stream_request(base: str, body: dict, echo: bool = False) -> tuple[float, list[float], int]:
    """Send a streaming request; return (start_time, token arrival times, token count)."""
    body = dict(body)
    body["stream"] = True
    start = time.perf_counter()
    token_times: list[float] = []
    n_tokens = 0
    with _post_json(base + "/chat/completions", body) as resp:
        for raw in resp:
            text = parse_stream_line(raw.decode("utf-8"))
            if text is None:
                continue
            token_times.append(time.perf_counter())
            n_tokens += 1
            if echo:
                print(text, end="", flush=True)
    return start, token_times, n_tokens


def stage_streaming(steps: dict[str, bool], base: str, model: str) -> None:
    _heading("Stage 3: streaming, with the stopwatch running")
    needed = ["build_chat_request", "parse_stream_line", "latency_stats"]
    if not all(steps[s] for s in needed):
        _skip("streaming", [s for s in needed if not steps[s]])
        return
    body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
    print("streaming reply:\n")
    start, token_times, _ = _stream_request(base, body, echo=True)
    print("\n")
    if len(token_times) < 2:
        print("(too few tokens streamed to measure anything)\n")
        return
    stats = latency_stats(start, token_times)
    print(f"time to first token : {stats['ttft'] * 1000:.0f} ms   (prefill)")
    print(f"decode rate         : {stats['tokens_per_second']:.1f} tokens/sec   (decode)")
    print("compare the decode rate with your stage 1 speed-limit prediction\n")


# ---------------------------------------------------------------------------
# Stage 4: the concurrency benchmark (steps 4, 6, 7, 8)
# ---------------------------------------------------------------------------


def stage_benchmark(steps: dict[str, bool], base: str, model: str) -> None:
    _heading("Stage 4: concurrency benchmark (batching, live)")
    needed = ["build_chat_request", "parse_stream_line", "latency_stats", "total_throughput"]
    if not all(steps[s] for s in needed):
        _skip("benchmark", [s for s in needed if not steps[s]])
        return

    print(f"{MAX_TOKENS} token budget per request; {BENCH_MAX_TOKENS} in the benchmark")
    print(f"{'concurrent':>10}{'total tok/s':>13}{'per-stream tok/s':>18}{'wall (s)':>10}")
    results = []
    for n in CONCURRENCY_LEVELS:
        bodies = [
            build_chat_request(model, BENCH_PROMPTS[i % len(BENCH_PROMPTS)],
                               TEMPERATURE, BENCH_MAX_TOKENS)
            for i in range(n)
        ]
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            runs = list(pool.map(lambda b: _stream_request(base, b), bodies))
        wall = time.perf_counter() - wall_start

        counts = [count for (_, _, count) in runs]
        throughput = total_throughput(counts, wall)
        per_stream = [
            latency_stats(start, times)["tokens_per_second"]
            for (start, times, _) in runs
            if len(times) >= 2
        ]
        mean_stream = sum(per_stream) / len(per_stream) if per_stream else 0.0
        results.append({"n": n, "total": throughput, "per_stream": mean_stream})
        print(f"{n:>10}{throughput:>13.1f}{mean_stream:>18.1f}{wall:>10.1f}")

    out_dir = _THIS_DIR.parent / "output"
    out_dir.mkdir(exist_ok=True)
    figure_path = out_dir / "throughput.png"
    save_throughput_figure(results, figure_path)
    print(f"\nsaved figure: {figure_path}")
    single = results[0]["total"]
    best = max(r["total"] for r in results)
    if single > 0:
        print(f"the same hardware now produces {best / single:.1f}x the tokens per second.")
    print("each stream slows, but far less than a queue would cost it: one weight")
    print("read now feeds every request in the batch\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    steps = _probe_steps()
    done = sum(steps.values())
    print(f"\nimplemented: {done}/8 steps "
          f"({', '.join(name for name, ok in steps.items() if ok) or 'none yet'})\n")

    stage_napkin_math(steps)

    server = _discover_server()
    if server is None:
        _heading("Stages 2-4 need a running server")
        print("No OpenAI-compatible server found. Start one (see README.md), e.g.:")
        print("  vllm serve Qwen/Qwen2.5-0.5B-Instruct        (Linux + GPU)")
        print("  ollama serve   +   ollama pull qwen2.5:0.5b-instruct")
        print("  llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf")
        print("then re-run this script.\n")
        return

    base, model = server
    print(f"server found: {base}  (model: {model})\n")
    stage_single_request(steps, base, model)
    stage_streaming(steps, base, model)
    stage_benchmark(steps, base, model)


if __name__ == "__main__":
    main()
