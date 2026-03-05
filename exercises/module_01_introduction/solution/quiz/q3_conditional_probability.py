"""
Quiz Q3: Given this bigram model, what is P('e' | 'th')?
"""

ANSWER = 0.5

if __name__ == "__main__":
    assert abs(ANSWER - 0.5) < 1e-6
    print("Correct! P('e'|'th') = 50/100 = 0.5")
