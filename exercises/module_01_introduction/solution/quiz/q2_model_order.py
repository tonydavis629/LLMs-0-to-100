"""
Quiz Q2: Which n-gram model order produced each sample?
"""

SAMPLE_A_ORDER = 1
SAMPLE_B_ORDER = 2
SAMPLE_C_ORDER = 0

if __name__ == "__main__":
    for name, val in [("SAMPLE_A_ORDER", SAMPLE_A_ORDER),
                      ("SAMPLE_B_ORDER", SAMPLE_B_ORDER),
                      ("SAMPLE_C_ORDER", SAMPLE_C_ORDER)]:
        assert isinstance(val, int), f"{name} must be an integer (0, 1, or 2)."
    assert SAMPLE_A_ORDER == 1
    assert SAMPLE_B_ORDER == 2
    assert SAMPLE_C_ORDER == 0
    print("Correct!")
