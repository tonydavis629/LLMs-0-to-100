"""
Quiz Q3: Given this bigram model, what is P('e' | 'th')?

    model["th"] = Counter({"e": 50, "a": 20, "i": 15, "o": 10, "u": 5})

Run: uv run python -m exercises.module_01_introduction.quiz.q3_conditional_probability
"""

# TODO: Replace with the probability as a float (e.g., 0.75)
ANSWER = ...


if __name__ == "__main__":
    assert isinstance(ANSWER, (int, float)), "ANSWER must be a number."
    assert abs(ANSWER - 0.5) < 1e-6, \
        f"P('e'|'th') = 50 / (50+20+15+10+5) = 50/100 = 0.5, got {ANSWER}"
    print("Correct! P('e'|'th') = 50/100 = 0.5")
    print("Conditional probability = count of target / total count for that context.")
