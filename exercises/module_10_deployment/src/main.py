"""
Module 10 Exercise runner: serve a model, then write an API client to it

Run with:
    uv run python module_10_deployment/src/main.py

Every step is tagged on its header line, then its output follows: the table
or live demo your code produced and the result of each test in tests/.
The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Steps 1-4 need no server. Steps 5-8 also run a live demo against an
OpenAI-compatible server on localhost (vLLM, Ollama, or llama.cpp; see the
README). The tests never need the server: without one, each of those steps
prints a single "live demo skipped" line and is still tagged by its tests.

Add --step N to run one step (1-8).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

# Make the module root (parent of src/) importable so we can `from exercise import ...`
# (and `from tests... import ...`), and src/ importable for the provided plotting helper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

from exercise import (  # noqa: E402  (import after sys.path edits)
    build_chat_request,
    decode_tokens_per_second,
    extract_reply,
    kv_cache_bytes_per_token,
    latency_stats,
    model_weight_bytes,
    parse_stream_line,
    total_throughput,
)

# One test file per step lives in tests/
from tests.test_step1_weight_bytes import check_model_weight_bytes  # noqa: E402
from tests.test_step2_kv_cache import check_kv_cache_bytes_per_token  # noqa: E402
from tests.test_step3_speed_limit import check_decode_tokens_per_second  # noqa: E402
from tests.test_step4_chat_request import check_build_chat_request  # noqa: E402
from tests.test_step5_extract_reply import check_extract_reply  # noqa: E402
from tests.test_step6_stream_line import check_parse_stream_line  # noqa: E402
from tests.test_step7_latency import check_latency_stats  # noqa: E402
from tests.test_step8_throughput import check_total_throughput  # noqa: E402
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

# The model name to show in Step 4 when no server is running yet
DEFAULT_MODEL = "qwen2.5:0.5b-instruct"

# Where to look for a server. The first URL that answers GET /models wins.
# vLLM serves on 8000, Ollama on 11434, llama.cpp's llama-server on 8080;
# all three speak the same OpenAI-compatible API, which is the point.
SERVER_CANDIDATES = [
    os.environ.get("LLM_SERVER_URL"),
    "http://localhost:8000/v1",
    "http://localhost:11434/v1",
    "http://localhost:8080/v1",
]

# The one line Steps 5-8 print in place of their live demo when no server answered
NO_SERVER = "live demo skipped: no server found on port 8000, 11434, or 8080 (see README.md)"

_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
OUTPUT_DIR = _MODULE_DIR / "output"
GB = 1e9


def _find_data_file(name: str) -> Path:
    """Walk up from src/ until we find data/<name>."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def _wrap(text: str) -> str:
    """Wrap a long model reply to terminal width, keeping its paragraphs."""
    paragraphs = [p for p in text.strip().split("\n") if p.strip()]
    return "\n".join(textwrap.fill(p, width=76) for p in paragraphs)


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


def _stream_request(base: str, body: dict) -> tuple[float, list[float], str]:
    """Send a streaming request; return (start_time, token arrival times, full text).

    Every line the server sends goes through YOUR parse_stream_line(); each
    line that carries text counts as one token and gets a timestamp.
    """
    body = dict(body)
    body["stream"] = True
    start = time.perf_counter()
    token_times: list[float] = []
    pieces: list[str] = []
    with _post_json(base + "/chat/completions", body) as resp:
        for raw in resp:
            text = parse_stream_line(raw.decode("utf-8"))
            if not text:
                continue  # role announcements, keep-alives, and [DONE] carry no text
            token_times.append(time.perf_counter())
            pieces.append(text)
    return start, token_times, "".join(pieces)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


# ---------------------------------------------------------------------------
# Demo helpers. The tests decide each step's tag; the demos are the payoff
# you get to watch once a step works.
# ---------------------------------------------------------------------------


def _demo(show):
    """Wrap a step's demo so a crash in it prints one line instead of hiding the tests.

    An unfinished blank (NotImplementedError) still passes through, so the
    step is tagged INCOMPLETE with its own TODO message.
    """

    def run():
        try:
            show()
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001 - the tests below explain what went wrong
            print(f"demo stopped: {type(e).__name__}: {e}")

    return run


def _require(fn, *args) -> None:
    """Call fn on a throwaway input so an unfinished step reports its own TODO first.

    This runs before any request goes to the server, so an unfinished step
    does not wait on the network before being tagged INCOMPLETE.
    """
    try:
        fn(*args)
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001 - a wrong answer is for the tests to explain
        pass


def _unfinished(fn, *args) -> bool:
    """True if fn (an earlier step's function) still raises NotImplementedError."""
    try:
        fn(*args)
    except NotImplementedError:
        return True
    except Exception:  # noqa: BLE001 - it runs, so the demo can try it
        pass
    return False


# Throwaway inputs used only to ask "is this step filled in yet?"
_PROBE_REQUEST = ("m", "hi", 0.7, 8)
_PROBE_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
_PROBE_LINE = 'data: {"choices": [{"delta": {"content": "hi"}}]}'
_PROBE_TIMES = (0.0, [0.5, 0.6, 0.7])


def _missing_steps(*needed: str) -> str | None:
    """Name the first earlier step a live demo needs that is still unfinished."""
    probes = {
        "4": ("Step 4 (build_chat_request) to build the request",
              lambda: _unfinished(build_chat_request, *_PROBE_REQUEST)),
        "6": ("Step 6 (parse_stream_line) to read the token stream",
              lambda: _unfinished(parse_stream_line, _PROBE_LINE)),
        "7": ("Step 7 (latency_stats) to time each stream",
              lambda: _unfinished(latency_stats, *_PROBE_TIMES)),
    }
    for step in needed:
        label, is_unfinished = probes[step]
        if is_unfinished():
            return f"live demo skipped: needs {label}"
    return None


# ---------------------------------------------------------------------------
# Steps 1-3: the napkin math (no server)
# ---------------------------------------------------------------------------


def step_1(models: list[dict]) -> str:
    """Weight memory for three real models at three precisions."""

    def show():
        # Compute every cell first, so an unfinished function prints no half table
        rows = [(m["name"], [model_weight_bytes(m["params"], bits) / GB for bits in (16, 8, 4)])
                for m in models]
        print("Weight memory by precision (GB):")
        print(f"  {'model':<14}{'fp16':>8}{'int8':>8}{'int4':>8}")
        for name, sizes in rows:
            print(f"  {name:<14}" + "".join(f"{s:>8.1f}" for s in sizes))

    return run_step("Step 1: model_weight_bytes()", _demo(show),
                    lambda: check_model_weight_bytes(model_weight_bytes))


def step_2(models: list[dict]) -> str:
    """How much the KV cache grows per token, and for one long conversation."""

    def show():
        # fp16 cache: 2 bytes per stored number
        rows = [(m["name"], kv_cache_bytes_per_token(m["layers"], m["kv_heads"], m["head_dim"], 2))
                for m in models]
        print("KV cache at fp16 (per token, and for one 8,192-token user):")
        print(f"  {'model':<14}{'KB/token':>10}{'GB @ 8K':>10}")
        for name, per_token in rows:
            print(f"  {name:<14}{per_token / 1e3:>10.1f}{per_token * 8192 / GB:>10.2f}")

    return run_step("Step 2: kv_cache_bytes_per_token()", _demo(show),
                    lambda: check_kv_cache_bytes_per_token(kv_cache_bytes_per_token))


def step_3(models: list[dict], machines: list[dict]) -> str:
    """The bandwidth speed limit for Llama 3 8B on four real machines."""

    def show():
        _require(decode_tokens_per_second, 1e12, 1e9)
        # The table divides by the weight size, which is Step 1's job
        if _unfinished(model_weight_bytes, 1000, 16):
            print("table skipped: needs Step 1 (model_weight_bytes) to size the weights")
            return
        llama = next(m for m in models if m["name"] == "Llama-3-8B")
        fp16_bytes = model_weight_bytes(llama["params"], 16)
        int4_bytes = model_weight_bytes(llama["params"], 4)
        print("Decode speed limit for Llama-3-8B, tokens/sec (bandwidth / bytes):")
        print(f"  {'machine':<22}{'GB/s':>7}{'fp16':>8}{'int4':>8}")
        for mach in machines:
            bw = mach["bandwidth_gb_s"] * GB
            fp16 = decode_tokens_per_second(bw, fp16_bytes)
            int4 = decode_tokens_per_second(bw, int4_bytes)
            # A model whose weights do not fit in memory has no speed at all
            fp16_cell = f"{fp16:>8.0f}" if fp16_bytes / GB <= mach["memory_gb"] else "   (n/a)"
            print(f"  {mach['name']:<22}{mach['bandwidth_gb_s']:>7.0f}{fp16_cell}{int4:>8.0f}")
        print("  (n/a: the fp16 weights alone do not fit in that machine's memory)")

    return run_step("Step 3: decode_tokens_per_second()", _demo(show),
                    lambda: check_decode_tokens_per_second(decode_tokens_per_second))


# ---------------------------------------------------------------------------
# Steps 4-8: the client. The tests use hand-made dicts, JSON lines, and
# timestamps, so they run with no server; only the live demos need one.
# ---------------------------------------------------------------------------


def step_4(server) -> str:
    """Build the request body and show the exact JSON that goes over the wire."""

    def show():
        # Use the server's model name if one is running, else the README's model
        model = server[1] if server else DEFAULT_MODEL
        body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
        print("JSON body for POST /v1/chat/completions:")
        print(json.dumps(body, indent=2, default=str))

    return run_step("Step 4: build_chat_request()", _demo(show),
                    lambda: check_build_chat_request(build_chat_request))


def step_5(server) -> str:
    """Live demo: one complete (non-streaming) request and its reply."""

    def show():
        _require(extract_reply, _PROBE_RESPONSE)
        if server is None:
            print(NO_SERVER)
            return
        missing = _missing_steps("4")
        if missing:
            print(missing)
            return

        base, model = server
        body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
        print(f"POST {base}/chat/completions  (model: {model})")
        print(f"prompt: {PROMPT}")
        # Send the request and wait for the whole reply to come back as one JSON document
        with _post_json(base + "/chat/completions", body) as resp:
            response = json.loads(resp.read().decode("utf-8"))
        print()
        print("reply:")
        print(_wrap(str(extract_reply(response))))
        print()
        usage = response.get("usage") or {}
        print(f"usage: {usage.get('prompt_tokens', '?')} prompt tokens in, "
              f"{usage.get('completion_tokens', '?')} completion tokens out "
              "(this is the billing meter)")

    return run_step("Step 5: extract_reply()", _demo(show),
                    lambda: check_extract_reply(extract_reply))


def step_6(server, results: dict) -> str:
    """Live demo: the same prompt again, streamed one chunk at a time."""

    def show():
        _require(parse_stream_line, _PROBE_LINE)
        if server is None:
            print(NO_SERVER)
            return
        missing = _missing_steps("4")
        if missing:
            print(missing)
            return

        base, model = server
        body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
        start, token_times, text = _stream_request(base, body)
        results["stream"] = (start, token_times)  # Step 7 times this same stream
        print("streamed reply, reassembled from the text in each data: line:")
        print(_wrap(text))
        print()
        print(f"{len(token_times)} chunks carried text, one per generated token")

    return run_step("Step 6: parse_stream_line()", _demo(show),
                    lambda: check_parse_stream_line(parse_stream_line))


def step_7(server, results: dict) -> str:
    """Live demo: TTFT and decode rate for the Step 6 stream."""

    def show():
        _require(latency_stats, *_PROBE_TIMES)
        if server is None:
            print(NO_SERVER)
            return
        missing = _missing_steps("4", "6")
        if missing:
            print(missing)
            return

        # Reuse the Step 6 stream if it ran; otherwise (--step 7) stream a new reply
        if "stream" not in results:
            base, model = server
            body = build_chat_request(model, PROMPT, TEMPERATURE, MAX_TOKENS)
            start, token_times, _ = _stream_request(base, body)
            results["stream"] = (start, token_times)
        start, token_times = results["stream"]
        if len(token_times) < 2:
            print("(too few tokens streamed to measure anything)")
            return
        stats = latency_stats(start, token_times)
        ttft_ms, rate = stats["ttft"] * 1000, stats["tokens_per_second"]
        print(f"stopwatch on the streamed reply ({len(token_times)} tokens):")
        print(f"time to first token : {ttft_ms:.0f} ms   (prefill)")
        print(f"decode rate         : {rate:.1f} tokens/sec   (decode)")
        print("compare the decode rate with the Step 3 speed limit for your machine")

    return run_step("Step 7: latency_stats()", _demo(show),
                    lambda: check_latency_stats(latency_stats))


def step_8(server) -> str:
    """Live demo: 1, 2, 4, then 8 requests at once. Batching, measured."""

    def show():
        _require(total_throughput, [10, 10], 2.0)
        if server is None:
            print(NO_SERVER)
            return
        missing = _missing_steps("4", "6", "7")
        if missing:
            print(missing)
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
                runs = list(pool.map(lambda b: _stream_request(base, b), bodies))
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

    return run_step("Step 8: total_throughput()", _demo(show),
                    lambda: check_total_throughput(total_throughput))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_CHOICES = ["1", "2", "3", "4", "5", "6", "7", "8"]


def main():
    parser = argparse.ArgumentParser(description="Serve a model, then write an API client to it")
    parser.add_argument("--step", choices=[*STEP_CHOICES, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = STEP_CHOICES if args.step == "all" else [args.step]

    # The hardware and model specs behind the napkin-math tables
    models = json.loads(_find_data_file("models.json").read_text())
    machines = json.loads(_find_data_file("machines.json").read_text())

    # Look for a server once, and only if a step that can use one will run
    server = _discover_server() if any(s in "45678" for s in steps) else None
    results: dict = {}  # the Step 6 stream, shared with Step 7

    for step in steps:
        if step == "1":
            step_1(models)
        elif step == "2":
            step_2(models)
        elif step == "3":
            step_3(models, machines)
        elif step == "4":
            step_4(server)
        elif step == "5":
            step_5(server)
        elif step == "6":
            step_6(server, results)
        elif step == "7":
            step_7(server, results)
        elif step == "8":
            step_8(server)


if __name__ == "__main__":
    main()
