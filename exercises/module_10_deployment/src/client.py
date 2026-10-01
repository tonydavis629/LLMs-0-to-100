"""Talking to the model server over HTTP, provided for you.

You do NOT need to edit this file. It uses only the standard library, so
there is nothing to install: urllib POSTs the JSON body and hands back the
raw response. Any OpenAI-compatible server works (vLLM, Ollama, or
llama.cpp's llama-server), because they all speak the same API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from functools import cache

# Where to look for a server. The first URL that answers GET /models wins.
# vLLM serves on 8000, Ollama on 11434, llama.cpp's llama-server on 8080;
# all three speak the same OpenAI-compatible API, which is the point.
SERVER_CANDIDATES = [
    os.environ.get("LLM_SERVER_URL"),
    "http://localhost:8000/v1",
    "http://localhost:11434/v1",
    "http://localhost:8080/v1",
]


def _get_json(url: str, timeout: float = 3.0) -> dict:
    """GET a URL and parse its JSON body."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, body: dict, timeout: float = 300.0):
    """POST a JSON body and return the open response, to read all at once or line by line."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


@cache  # look for a server once, then reuse the answer in every step
def find_server() -> tuple[str, str] | None:
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


def send_request(base: str, body: dict) -> dict:
    """POST a chat request and wait for the whole reply as one JSON document."""
    with _post_json(base + "/chat/completions", body) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_request(base: str, body: dict, parse_line) -> tuple[float, list[float], str]:
    """Send a streaming request; return (start_time, token arrival times, full text).

    Every line the server sends goes through `parse_line` (your
    parse_stream_line from Step 6); each line that carries text counts as one
    token and gets a timestamp.
    """
    body = dict(body)
    body["stream"] = True
    start = time.perf_counter()
    token_times: list[float] = []
    pieces: list[str] = []
    with _post_json(base + "/chat/completions", body) as resp:
        for raw in resp:
            text = parse_line(raw.decode("utf-8"))
            if not text:
                continue  # role announcements, keep-alives, and [DONE] carry no text
            token_times.append(time.perf_counter())
            pieces.append(text)
    return start, token_times, "".join(pieces)
