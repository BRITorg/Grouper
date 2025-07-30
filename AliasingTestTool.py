from rapidfuzz import fuzz

def dynamic_threshold(token1, token2, base_threshold=75, max_threshold=90):
    avg_len = (len(token1) + len(token2)) / 2
    if avg_len <= 5:
        return base_threshold
    elif avg_len >= 15:
        return max_threshold
    else:
        return base_threshold + ((avg_len - 5) / 10) * (max_threshold - base_threshold)

def test_aliasing():
    print("Token Aliasing Similarity Checker")
    print("Enter empty input to quit.\n")

    while True:
        token1 = input("Original token: ").strip()
        if not token1:
            break

        token2 = input("Canonical token: ").strip()
        if not token2:
            break

        length1 = len(token1)
        length2 = len(token2)
        score = fuzz.ratio(token1, token2)
        threshold = dynamic_threshold(token1, token2)

        print(f"\nOriginal Length: {length1}")
        print(f"Canonical Length: {length2}")
        print(f"Fuzzy Similarity: {score:.2f}")
        print(f"Dynamic Threshold: {threshold:.2f}")
        if score >= threshold:
            print("✅ ALIASING WOULD OCCUR")
        else:
            print("❌ ALIASING WOULD NOT OCCUR")
        print("-" * 40)

if __name__ == "__main__":
    test_aliasing()
