"""
Module 10 Exercise: Serve a model, then write an API client to it

You implement the eight functions below. Everything else (the server, the HTTP
plumbing, the threading, the hardware data, and the plotting) is provided. Each
blank is one line or one short expression.

Steps 1-3 are the lecture's napkin math as code and run with no server at all.
Steps 4-8 build a real client for an OpenAI-compatible inference server: one
request, then a streaming request with latency measurement, then a concurrency
benchmark that makes batching visible.

Run after each step; unfinished steps are skipped automatically:
    uv run python module_10_deployment/src/main.py
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Step 1: Model weight memory
# ---------------------------------------------------------------------------


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
    # HINT: There are 8 bits in a byte, so divide the total bits by 8.
    raise NotImplementedError("TODO: compute the weight memory in bytes")


# ---------------------------------------------------------------------------
# Step 2: KV cache size per token
# ---------------------------------------------------------------------------


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
    # HINT: Multiply all four arguments together, then double it (one key AND
    #       one value).
    raise NotImplementedError("TODO: compute the KV cache bytes per token")


# ---------------------------------------------------------------------------
# Step 3: The decode speed limit
# ---------------------------------------------------------------------------


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
    # HINT: It is one division. Bandwidth on top.
    raise NotImplementedError("TODO: compute the decode speed limit")


# ---------------------------------------------------------------------------
# Step 4: Build a chat completion request
# ---------------------------------------------------------------------------


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
    # HINT: The OpenAI format is [{"role": ..., "content": ...}].
    raise NotImplementedError("TODO: add the messages list to the request body")


# ---------------------------------------------------------------------------
# Step 5: Extract the reply
# ---------------------------------------------------------------------------


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
    # HINT: response["choices"] is a list; take element 0, then its "message",
    #       then that message's "content".
    raise NotImplementedError("TODO: extract the reply text from the response")


# ---------------------------------------------------------------------------
# Step 6: Parse one line of the token stream
# ---------------------------------------------------------------------------


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
    # HINT: chunk["choices"][0]["delta"] is a dict; .get("content") returns
    #       None on its own when the key is missing.
    raise NotImplementedError("TODO: extract the token text from the chunk")


# ---------------------------------------------------------------------------
# Step 7: Latency statistics from token timestamps
# ---------------------------------------------------------------------------


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
    # HINT: len(token_times) - 1 tokens arrived between token_times[0] and
    #       token_times[-1].
    raise NotImplementedError("TODO: compute the decode rate in tokens per second")


# ---------------------------------------------------------------------------
# Step 8: Total throughput across concurrent requests
# ---------------------------------------------------------------------------


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
    # HINT: sum() the counts.
    raise NotImplementedError("TODO: compute the combined tokens per second")
