"""
Quiz Q4: Which distribution has HIGHER entropy?

    Distribution A: {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    Distribution B: {"a": 0.97, "b": 0.01, "c": 0.01, "d": 0.01}

Run: uv run python -m exercises.module_01_introduction.quiz.q4_entropy
"""

# TODO: Replace with "A" or "B"
ANSWER = ...


if __name__ == "__main__":
    assert ANSWER in ("A", "B"), 'ANSWER must be "A" or "B".'
    assert ANSWER == "A", \
        "Uniform distributions have maximum entropy (most uncertainty). " \
        "Distribution A is uniform, Distribution B is concentrated on 'a'."
    print("Correct! Uniform distributions have maximum entropy.")
    print("Entropy measures uncertainty -- the more spread out, the higher.")
    print("Distribution B is very predictable (almost always 'a'), so low entropy.")
