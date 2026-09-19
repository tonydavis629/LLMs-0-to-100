"""Step 5: extract_reply()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import json

from tests.check import Check, bad, ok

# A complete response in the shape an OpenAI-compatible server sends back,
# as raw JSON text. The runner parses it with json.loads before calling you.
SAMPLE_RESPONSE = """
{
  "id": "chatcmpl-42",
  "object": "chat.completion",
  "model": "qwen2.5:0.5b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Paris is the capital of France."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
}
"""


def _nudge(got) -> str:
    """Name the usual wrong turn when the result is a dict or list instead of text."""
    if isinstance(got, dict) and "content" in got:
        return " (that is the whole message: take its \"content\")"
    if isinstance(got, (dict, list)):
        return " (keep going down: choices, then [0], then \"message\", then \"content\")"
    return ""


def check_extract_reply(extract_reply) -> list[Check]:
    """The text lives at response["choices"][0]["message"]["content"]."""
    checks = []

    # The smallest possible response: one choice, one message
    tiny = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    got = extract_reply(tiny)
    name = 'a one-choice response with content "hi" gives "hi"'
    checks.append(ok(name) if got == "hi" else bad(name, f"expected 'hi'\ngot      {got!r}{_nudge(got)}"))

    # A full server response: the id, usage, and finish_reason keys are ignored
    got = extract_reply(json.loads(SAMPLE_RESPONSE))
    want = "Paris is the capital of France."
    name = "a full server response (with id, usage, finish_reason) gives just the reply text"
    checks.append(ok(name) if got == want else bad(name, f"expected {want!r}\ngot      {got!r}{_nudge(got)}"))

    # Asking for n=2 samples returns two choices; the first is choice 0
    two = {"choices": [
        {"index": 0, "message": {"role": "assistant", "content": "first"}},
        {"index": 1, "message": {"role": "assistant", "content": "second"}},
    ]}
    got = extract_reply(two)
    name = 'with two choices it reads the first one: "first", not "second"'
    checks.append(ok(name) if got == "first" else bad(name, f"expected 'first'\ngot      {got!r}{_nudge(got)}"))
    return checks
