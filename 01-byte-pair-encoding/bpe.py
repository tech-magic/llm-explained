from collections import defaultdict

def get_pairs(symbols):
    """Return frequency count of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        pairs[pair] += 1
    return pairs

def merge_pair(symbols, merge):
    """Merge all occurrences of a given pair."""
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == merge:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged

def print_step(step, symbols, merge=None):
    print(f"\n### Step {step}")
    if merge:
        print(f"🔁 Merged: {merge[0]} + {merge[1]} → {merge[0]+merge[1]}")
    print("📌 Sequence:", symbols)

def bpe_demo(input_text, num_merges=3):
    print("## Byte Pair Encoding (BPE) Demo\n")
    print("📘 Input Text:", repr(input_text))

    # Initialize sequence of characters
    symbols = list(input_text)
    print_step(0, symbols)

    merges = []
    vocab = set(symbols)

    for step in range(1, num_merges + 1):
        pairs = get_pairs(symbols)
        if not pairs:
            break

        # Select most frequent pair
        most_frequent = max(pairs.items(), key=lambda x: x[1])[0]
        merges.append(most_frequent)

        # Apply merge
        symbols = merge_pair(symbols, most_frequent)
        vocab.update(symbols)
        print_step(step, symbols, most_frequent)

    print("\n✅ Final Output:")
    print("🧱 Final Tokenized Sequence:", symbols)

    print("\n📦 Final Vocabulary:")
    for token in sorted(vocab):
        print("  ", token)

    print("\n🧠 Learned Merges (in order):")
    for idx, merge in enumerate(merges, 1):
        print(f"  {idx}. {merge[0]} + {merge[1]} → {merge[0]+merge[1]}")

if __name__ == "__main__":
    input_text = "aaabdaaabac"
    bpe_demo(input_text, num_merges=3)