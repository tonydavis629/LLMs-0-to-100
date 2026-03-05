"""
Quiz Q1: What is the most common character in English text (including spaces)?
"""

ANSWER = " "

if __name__ == "__main__":
    assert isinstance(ANSWER, str) and len(ANSWER) == 1, \
        "ANSWER must be a single character string."
    assert ANSWER == " ", \
        "Not quite. Hint: it's not 'e'. Think about what separates every word."
    print("Correct! Space is the most common character in English text.")
    print("'e' is the most common letter, but space appears more often.")
