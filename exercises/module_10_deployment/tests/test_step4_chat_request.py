"""Step 4: build_chat_request()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import json

from tests.check import Check, bad, ok


def check_build_chat_request(build_chat_request) -> list[Check]:
    """The OpenAI chat completions body: settings plus a one-message list."""
    checks = []
    body = build_chat_request("tiny-model", "Hi there", 0.5, 16)

    # A common slip: set body["messages"] but never hand the body back
    name = "returns the request body as a dict"
    if not isinstance(body, dict):
        got = "None (did you forget `return body`?)" if body is None else f"{type(body).__name__}: {body!r}"
        return [bad(name, f"got {got}")]
    checks.append(ok(name))

    # The one line you wrote: a list holding a single user message.
    # (A system message in front of it, from the extra credit, is fine too.)
    expected = [{"role": "user", "content": "Hi there"}]
    name = 'prompt "Hi there" becomes messages = [{"role": "user", "content": "Hi there"}]'
    got = body.get("messages")
    fits = (isinstance(got, list) and len(got) >= 1 and got[-1] == expected[0]
            and all(isinstance(m, dict) and m.get("role") == "system" for m in got[:-1]))
    checks.append(ok(name) if fits else bad(name, f"expected {expected!r}\ngot      {got!r}"))

    # The settings the runner passed in must still be there
    kept = {k: body.get(k) for k in ("model", "temperature", "max_tokens")}
    wanted = {"model": "tiny-model", "temperature": 0.5, "max_tokens": 16}
    name = "keeps model, temperature, and max_tokens as given"
    checks.append(ok(name) if kept == wanted else bad(name, f"expected {wanted!r}\ngot      {kept!r}"))

    # Everything has to survive JSON encoding to travel over HTTP
    name = "the body survives a JSON round trip, as it must to go over the wire"
    try:
        same = json.loads(json.dumps(body)) == body
        checks.append(ok(name) if same else bad(name, "json.loads(json.dumps(body)) came back different"))
    except (TypeError, ValueError) as e:
        checks.append(bad(name, f"json.dumps failed: {e} (use a list [...] and a dict {{...}}, not a set)"))
    return checks
