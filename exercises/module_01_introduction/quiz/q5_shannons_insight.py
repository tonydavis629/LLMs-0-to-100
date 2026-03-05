"""
Quiz Q5: As we increase the order of our n-gram model, what happens
to the generated text?

    a) It becomes more random
    b) It becomes more coherent but may start copying the source
    c) It stays about the same
    d) It becomes shorter

Run: uv run python -m exercises.module_01_introduction.quiz.q5_shannons_insight
"""

# TODO: Replace with "a", "b", "c", or "d"
ANSWER = ...


if __name__ == "__main__":
    assert ANSWER in ("a", "b", "c", "d"), 'ANSWER must be "a", "b", "c", or "d".'
    assert ANSWER == "b", \
        "Higher-order models capture more context, producing more coherent text. " \
        "But with limited training data, very high orders memorize and reproduce " \
        "chunks of the original."
    print("Correct! Higher-order models produce more coherent text,")
    print("but with limited data they start copying the source verbatim.")
    print("This is the bias-variance tradeoff in language modeling.")
