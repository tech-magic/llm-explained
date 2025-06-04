# Byte Pair Encoding (BPE)

## 🧩 What is BPE?

- **Byte Pair Encoding (BPE)** is a subword tokenization technique that helps reduce the overall vocabulary size while retaining frequently occurring character or subword patterns.
- Instead of representing each word as a unique token, BPE breaks words into smaller units (subwords), enabling the model to handle **rare or unseen words** effectively.

## 🤖 Why is BPE used in LLMs?

- BPE is widely used in **transformer-based language models** like **GPT**, **BERT**, **RoBERTa**, and more.
- These models use BPE to **convert input text into tokens** before processing. This step ensures that:
  - Frequent words are encoded efficiently.
  - Rare words are broken into smaller, known parts.
  - The tokenizer avoids the problem of out-of-vocabulary (OOV) words.

## 🏷️ How does it work?

- The process begins by breaking words into characters.
- Then, the **most frequent adjacent character pairs are merged iteratively**.
- This continues until a predefined vocabulary size is reached or a merge list is exhausted.

For example:
```
Input: a a a b d a a a b a c
Merge 1: a a → aa → aa a b d aa a b a c
Merge 2: aa a → aaa → aaa b d aaa b a c
...
```


## 🏁 Special Markers

- In practical implementations, BPE uses **special tokens like `</w>` (end-of-word)** to preserve word boundaries.
- This helps models learn meaningful merges like:
  - `play + ing</w>` → `playing</w>`
  - `inter + est + ing</w>` → `interesting</w>`

## 🧠 During Inference

- The same learned BPE rules are applied to new input text.
- The input is **split into characters**, and **merge rules are applied** to produce subword tokens.
- These tokens are then fed into the LLM for processing.

For example:
```
Input: unrecognizable
Tokens: un, recog, n, iz, able
```

## ✅ Benefits of BPE in LLMs

- 📉 **Smaller vocabulary size**  
- 🧠 **Better generalization to rare and unseen words**  
- 🔄 **Consistent and deterministic tokenization process**  
- 🚀 **Efficient training and inference due to fewer token splits**  

> 📝 BPE strikes a smart balance between word-level and character-level tokenization. It's one of the key ingredients that powers today's most powerful LLMs by making them faster, smaller, and smarter in handling human language.



---

## 🧮 Hand-Calculated Example

### 📘 Input

Toy corpus:
```
aaabdaaabac
```

We treat the corpus as a sequence of characters, and we can represent it as a list of symbols:

```
['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']
```

---

### 🔁 Applying BPE: Step-by-Step

We'll now apply **BPE** iteratively by following these steps:

1. **Count symbol pairs**
2. **Find the most frequent pair**
3. **Merge the pair**
4. **Repeat** until a stopping condition (e.g., when all pairs have a frequency of 1, after a fixed number of iterations etc.)

---

#### ✅ Step 0: Initial Sequence
```
['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']
```



#### 🔢 Step 1: Count symbol pairs

Sliding window of size 2 over the sequence:

| Pair  | Count |
|-------|-------|
| a a   | 4     |
| a b   | 2     |
| b d   | 1     |
| d a   | 1     |
| b a   | 1     |
| a c   | 1     |

🔝 **Most frequent pair: `a a` (count = 4)**

#### 🔁 Step 2: Merge `a a` → `aa`

After merging all `a a`:

**New sequence:**
```
['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']
```

#### 🔢 Step 3: Count symbol pairs again

| Pair   | Count |
|--------|-------|
| aa a   | 2     |
| a b    | 2     |
| b d    | 1     |
| d aa   | 1     |
| b a    | 1     |
| a c    | 1     |

🔝 **Most frequent pair: `aa a` (2 times)**

#### 🔁 Step 4: Merge `aa a` → `aaa`

**New sequence:**
```
['aaa', 'b', 'd', 'aaa', 'b', 'a', 'c']
```

#### 🔢 Step 5: Count symbol pairs

| Pair   | Count |
|--------|-------|
| aaa b  | 2     |
| b d    | 1     |
| d aaa  | 1     |
| b a    | 1     |
| a c    | 1     |

🔝 **Most frequent pair: `aaa b`**

#### 🔁 Step 6: Merge `aaa b` → `aaab`

**New sequence:**
```
['aaab', 'd', 'aaab', 'a', 'c']
```

#### 🔢 Step 7: Count symbol pairs

| Pair   | Count |
|--------|-------|
| aaab d | 1     |
| d aaab | 1     |
| aaab a | 1     |
| a c    | 1     |

All pairs have frequency 1 → could stop here or continue merging based on a fixed number of steps.

#### ✅ Final Output

**Learned merges (in order):**
1. `a a` → `aa`
2. `aa a` → `aaa`
3. `aaa b` → `aaab`

**Final tokenized sequence:**
```
['aaab', 'd', 'aaab', 'a', 'c']
```

---

## 🐍 Python Code

```python
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
```

#### 🧾 Sample Output
```
### Step 0
📌 Sequence: ['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']

### Step 1
🔁 Merged: a + a → aa
📌 Sequence: ['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']

### Step 2
🔁 Merged: aa + a → aaa
📌 Sequence: ['aaa', 'b', 'd', 'aaa', 'b', 'a', 'c']

### Step 3
🔁 Merged: aaa + b → aaab
📌 Sequence: ['aaab', 'd', 'aaab', 'a', 'c']

✅ Final Output:
🧱 Final Tokenized Sequence: ['aaab', 'd', 'aaab', 'a', 'c']

📦 Final Vocabulary:
   a
   aa
   aaa
   aaab
   b
   c
   d

🧠 Learned Merges (in order):
  1. a + a → aa
  2. aa + a → aaa
  3. aaa + b → aaab
```

---

## 📚 References

- [Byte Pair Encoding (Wikipedia)](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [Sennrich et al. 2016 – Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

---



