"""
Quiz Q2: Which n-gram model order produced each sample?

Sample A: "tg nraoie shdi  aettn lsoe"
Sample B: "the was not a little she had"
Sample C: "xmqzj bvlkw npfrt gdyhc"

Run: uv run python -m exercises.module_01_introduction.quiz.q2_model_order
"""

# TODO: Set each to 0, 1, or 2
#   0 = uniform random
#   1 = unigram (character frequencies)
#   2 = bigram/trigram (higher-order)
SAMPLE_A_ORDER = ...
SAMPLE_B_ORDER = ...
SAMPLE_C_ORDER = ...


if __name__ == "__main__":
    for name, val in [("SAMPLE_A_ORDER", SAMPLE_A_ORDER),
                      ("SAMPLE_B_ORDER", SAMPLE_B_ORDER),
                      ("SAMPLE_C_ORDER", SAMPLE_C_ORDER)]:
        assert isinstance(val, int), f"{name} must be an integer (0, 1, or 2)."

    assert SAMPLE_A_ORDER == 1, \
        "Sample A has English-like letter frequencies but no word structure -> unigram (order 1)"
    assert SAMPLE_B_ORDER == 2, \
        "Sample B has recognizable words -> higher order model (order 2+)"
    assert SAMPLE_C_ORDER == 0, \
        "Sample C has uniform random characters with no frequency pattern -> order 0"

    print("Correct! You can identify model order by looking at the output:")
    print("  Order 0: random noise, all characters equally likely")
    print("  Order 1: right letter frequencies, but no word structure")
    print("  Order 2+: recognizable letter combinations and words")
