"""Step 6: parse_stream_line()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok

# A short stream, line by line, the way a server sends it. The first chunk
# only announces the role, the last one only carries finish_reason, and the
# blank lines between events are part of the server-sent-events format.
SAMPLE_STREAM = [
    'data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}',
    "",
    'data: {"choices": [{"index": 0, "delta": {"content": "Hel"}}]}',
    "",
    'data: {"choices": [{"index": 0, "delta": {"content": "lo"}}]}',
    "",
    'data: {"choices": [{"index": 0, "delta": {"content": "!"}}]}',
    "",
    'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
    "",
    "data: [DONE]",
]


def _call(parse_stream_line, line: str):
    """Call the student's function, turning a KeyError into a readable result."""
    try:
        return parse_stream_line(line), None
    except KeyError as e:
        return None, f"raised KeyError: {e} (use .get(\"content\") so a chunk with no text gives None)"


def check_parse_stream_line(parse_stream_line) -> list[Check]:
    """Each data: line holds a JSON chunk; the new text is in choices[0].delta.content."""
    checks = []

    # One ordinary token, with its leading space kept
    got, err = _call(parse_stream_line, 'data: {"choices": [{"index": 0, "delta": {"content": " Paris"}}]}')
    name = 'a token chunk with delta {"content": " Paris"} gives " Paris", leading space kept'
    if err:
        checks.append(bad(name, err))
    else:
        nudge = " (that is the whole delta: take its \"content\")" if isinstance(got, dict) else ""
        checks.append(ok(name) if got == " Paris" else bad(name, f"expected ' Paris'\ngot      {got!r}{nudge}"))

    # The final chunk carries only finish_reason: its delta is empty
    got, err = _call(parse_stream_line, 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}')
    name = "the closing chunk with an empty delta {} gives None"
    if err:
        checks.append(bad(name, err))
    else:
        checks.append(ok(name) if got is None else bad(name, f"expected None\ngot      {got!r}"))

    # A whole stream: role chunk, three tokens, closing chunk, blank lines, [DONE]
    pieces, err = [], None
    for line in SAMPLE_STREAM:
        text, err = _call(parse_stream_line, line)
        if err:
            break
        if text:
            pieces.append(text)
    name = 'a whole 11-line stream (role chunk, 3 tokens, [DONE]) reassembles to "Hello!"'
    if err:
        checks.append(bad(name, err))
    else:
        joined = "".join(p if isinstance(p, str) else repr(p) for p in pieces)
        checks.append(ok(name) if joined == "Hello!" else bad(name, f"expected 'Hello!'\ngot      {joined!r}"))
    return checks
