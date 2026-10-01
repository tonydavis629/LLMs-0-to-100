"""
Module 10 Exercise runner: serve a model, then write an API client to it

Run with:
    uv run python module_10_deployment/src/main.py

Each step below runs one function you write in exercise.py, then tests it.
Read top to bottom, the steps go from napkin math to a working client:

    1. how much memory the weights take at each precision
    2. how much memory the KV cache adds per token
    3. the decode speed limit set by memory bandwidth
    4. build the JSON body of a chat request
    5. send it and pull the reply out of the response
    6. stream the reply one token at a time
    7. time the stream: time to first token and decode rate
    8. send many requests at once and measure total throughput

Steps 1-4 need no server. Steps 5-8 also run a live demo against an
OpenAI-compatible server on localhost (vLLM, Ollama, or llama.cpp; see the
README). The tests never need the server: without one, each of those steps
prints a single "live demo skipped" line and is still tagged by its tests.

Add --step N to run one step (1-8).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    build_chat_request,
    decode_tokens_per_second,
    extract_reply,
    kv_cache_bytes_per_token,
    latency_stats,
    model_weight_bytes,
    parse_stream_line,
    total_throughput,
)
from src.client import find_server, send_request, stream_request
from src.prerequisites import missing_steps, require
from src.reporting import demo, run_step, wrap
from src.visualization import save_throughput_figure

# One test file per step lives in tests/
from tests.test_step1_weight_bytes import check_model_weight_bytes
from tests.test_step2_kv_cache import check_kv_cache_bytes_per_token
from tests.test_step3_speed_limit import check_decode_tokens_per_second
from tests.test_step4_chat_request import check_build_chat_request
from tests.test_step5_extract_reply import check_extract_reply
from tests.test_step6_stream_line import check_parse_stream_line
from tests.test_step7_latency import check_latency_stats
from tests.test_step8_throughput import check_total_throughput

MODULE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = MODULE_DIR / "output"
GB = 1e9

# The hardware and model specs behind the napkin-math tables
MODELS = json.loads((MODULE_DIR / "data" / "models.json").read_text())
MACHINES = json.loads((MODULE_DIR / "data" / "machines.json").read_text())

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

# The model name to show in Step 4 when no server is running yet
DEFAULT_MODEL = "qwen2.5:0.5b-instruct"

# The one line Steps 5-8 print in place of their live demo when no server answered
NO_SERVER = "live demo skipped: no server found on port 8000, 11434, or 8080 (see README.md)"


# ---------------------------------------------------------------------------
# Step 1: weight memory at three precisions (no server)
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        # Compute every cell first, so an unfinished function prints no half table
        rows = [(m["name"], [model_weight_bytes(m["params"], bits) / GB for bits in (16, 8, 4)])
                for m in MODELS]
        print("Weight memory by precision (GB):")
        print(f"  {'model':<14}{'fp16':>8}{'int8':>8}{'int4':>8}")
        for name, sizes in rows:
            print(f"  {name:<14}" + "".join(f"{s:>8.1f}" for s in sizes))

    return run_step("Step 1: model_weight_bytes()", demo(show),
                    lambda: check_model_weight_bytes(model_weight_bytes))


# ---------------------------------------------------------------------------
# Step 2: KV cache memory per token (no server)
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        # fp16 cache: 2 bytes per stored number
        rows = [(m["name"], kv_cache_bytes_per_token(m["layers"], m["kv_heads"], m["head_dim"], 2))
                for m in MODELS]
        print("KV cache at fp16 (per token, and for one 8,192-token user):")
        print(f"  {'model':<14}{'KB/token':>10}{'GB @ 8K':>10}")
        for name, per_token in rows:
            print(f"  {name:<14}{per_token / 1e3:>10.1f}{per_token * 8192 / GB:>10.2f}")

    return run_step("Step 2: kv_cache_bytes_per_token()", demo(show),
                    lambda: check_kv_cache_bytes_per_token(kv_cache_bytes_per_token))


# ---------------------------------------------------------------------------
# Step 3: the bandwidth speed limit for Llama 3 8B (no server)
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        require("3")
        # The table divides by the weight size, which is Step 1's job
        missing = missing_steps("1")
        if missing:
            print(f"table skipped: {missing}")
            return

        llama = next(m for m in MODELS if m["name"] == "Llama-3-8B")
        fp16_bytes = model_weight_bytes(llama["params"], 16)
        int4_bytes = model_weight_bytes(llama["params"], 4)
        print("Decode speed limit for Llama-3-8B, tokens/sec (bandwidth / bytes):")
        print(f"  {'machine':<22}{'GB/s':>7}{'fp16':>8}{'int4':>8}")
        for mach in MACHINES:
            bw = mach["bandwidth_gb_s"] * GB
            fp16 = decode_tokens_per_second(bw, fp16_bytes)
            int4 = decode_tokens_per_second(bw, int4_bytes)
            # A model whose weights do not fit in memory has no speed at all
            fp16_cell = f"{fp16:>8.0f}" if fp16_bytes / GB <= mach["memory_gb"] else "   (n/a)"
            print(f"  {mach['name']:<22}{mach['bandwidth_gb_s']:>7.0f}{fp16_cell}{int4:>8.0f}")
        print("  (n/a: the fp16 weights alone do not fit in that machine's memory)")

    return run_step("Step 3: decode_tokens_per_second()", demo(show),
                    lambda: check_decode_tokens_per_second(decode_tokens_per_second))


# ---------------------------------------------------------------------------
# Step 4: build the request body
# ---------------------------------------------------------------------------
# From here on the tests use hand-made dicts, JSON lines, and timestamps, so
# they run with no server; only the live demos need one.


def step_4() -> str:
    def show():
        # Use the server's model name if one is running, else the README's model
        server = find_server()
        model = server[1] if server else DEFAULT_MODEL
        body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
        print("JSON body for POST /v1/chat/completions:")
        print(json.dumps(body, indent=2, default=str))

    return run_step("Step 4: build_chat_request()", demo(show),
                    lambda: check_build_chat_request(build_chat_request))


# ---------------------------------------------------------------------------
# Step 5: one complete request, and the reply pulled out of the response
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        require("5")
        server = find_server()
        if server is None:
            print(NO_SERVER)
            return
        missing = missing_steps("4")
        if missing:
            print(f"live demo skipped: {missing}")
            return

        base, model = server
        body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
        print(f"POST {base}/chat/completions  (model: {model})")
        print(f"prompt: {PROMPT}")
        # Send the request and wait for the whole reply to come back as one JSON document
        response = send_request(base, body)
        print()
        print("reply:")
        print(wrap(str(extract_reply(response))))
        print()
        usage = response.get("usage") or {}
        print(f"usage: {usage.get('prompt_tokens', '?')} prompt tokens in, "
              f"{usage.get('completion_tokens', '?')} completion tokens out "
              "(this is the billing meter)")

    return run_step("Step 5: extract_reply()", demo(show),
                    lambda: check_extract_reply(extract_reply))


# ---------------------------------------------------------------------------
# Step 6: the same prompt again, streamed one token at a time
# ---------------------------------------------------------------------------


@cache  # stream once, so Step 7 can time the same reply Step 6 printed
def stream_the_prompt(base: str, model: str) -> tuple[float, list[float], str]:
    """Stream PROMPT through your parse_stream_line; return (start, token times, text)."""
    body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
    return stream_request(base, body, parse_stream_line)


def step_6() -> str:
    def show():
        require("6")
        server = find_server()
        if server is None:
            print(NO_SERVER)
            return
        missing = missing_steps("4")
        if missing:
            print(f"live demo skipped: {missing}")
            return

        start, token_times, text = stream_the_prompt(*server)
        print("streamed reply, reassembled from the text in each data: line:")
        print(wrap(text))
        print()
        print(f"{len(token_times)} chunks carried text, one per generated token")

    return run_step("Step 6: parse_stream_line()", demo(show),
                    lambda: check_parse_stream_line(parse_stream_line))


# ---------------------------------------------------------------------------
# Step 7: time to first token and decode rate for that stream
# ---------------------------------------------------------------------------


def step_7() -> str:
    def show():
        require("7")
        server = find_server()
        if server is None:
            print(NO_SERVER)
            return
        missing = missing_steps("4", "6")
        if missing:
            print(f"live demo skipped: {missing}")
            return

        # The Step 6 stream if it ran; otherwise (--step 7) this streams a new reply
        start, token_times, _ = stream_the_prompt(*server)
        if len(token_times) < 2:
            print("(too few tokens streamed to measure anything)")
            return
        stats = latency_stats(start, token_times)
        ttft_ms, rate = stats["ttft"] * 1000, stats["tokens_per_second"]
        print(f"stopwatch on the streamed reply ({len(token_times)} tokens):")
        print(f"time to first token : {ttft_ms:.0f} ms   (prefill)")
        print(f"decode rate         : {rate:.1f} tokens/sec   (decode)")
        print("compare the decode rate with the Step 3 speed limit for your machine")

    return run_step("Step 7: latency_stats()", demo(show),
                    lambda: check_latency_stats(latency_stats))


# ---------------------------------------------------------------------------
# Step 8: 1, 2, 4, then 8 requests at once. Batching, measured.
# ---------------------------------------------------------------------------


def step_8() -> str:
    def show():
        require("8")
        server = find_server()
        if server is None:
            print(NO_SERVER)
            return
        missing = missing_steps("4", "6", "7")
        if missing:
            print(f"live demo skipped: {missing}")
            return

        base, model = server
        print(f"{BENCH_MAX_TOKENS} token budget per request")
        print(f"{'concurrent':>10}{'total tok/s':>13}{'per-stream tok/s':>18}{'wall (s)':>10}")
        rows = []
        for n in CONCURRENCY_LEVELS:
            # n different prompts, so the server cannot answer one from another
            bodies = [
                build_chat_request(model, BENCH_PROMPTS[i % len(BENCH_PROMPTS)],
                                   TEMPERATURE, BENCH_MAX_TOKENS)
                for i in range(n)
            ]
            # Fire all n at once from n threads and wait until the last one finishes
            wall_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                runs = list(pool.map(lambda b: stream_request(base, b, parse_stream_line), bodies))
            wall = time.perf_counter() - wall_start

            # Operator's view: every token, over the wall time (your Step 8)
            counts = [len(times) for (_, times, _) in runs]
            throughput = total_throughput(counts, wall)
            # One user's view: each stream's own decode rate (your Step 7), averaged
            per_stream = [
                latency_stats(start, times)["tokens_per_second"]
                for (start, times, _) in runs
                if len(times) >= 2
            ]
            mean_stream = sum(per_stream) / len(per_stream) if per_stream else 0.0
            rows.append({"n": n, "total": throughput, "per_stream": mean_stream})
            print(f"{n:>10}{throughput:>13.1f}{mean_stream:>18.1f}{wall:>10.1f}")

        OUTPUT_DIR.mkdir(exist_ok=True)
        save_throughput_figure(rows, OUTPUT_DIR / "throughput.png")
        print()
        print("saved figure: output/throughput.png")
        single = rows[0]["total"]
        best = max(r["total"] for r in rows)
        if single > 0:
            print(f"the same hardware now produces {best / single:.1f}x the tokens per second.")
        print("each stream slows, but far less than a queue would cost it: one weight")
        print("read now feeds every request in the batch")

    return run_step("Step 8: total_throughput()", demo(show),
                    lambda: check_total_throughput(total_throughput))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4,
         "5": step_5, "6": step_6, "7": step_7, "8": step_8}


def main():
    parser = argparse.ArgumentParser(description="Serve a model, then write an API client to it")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
