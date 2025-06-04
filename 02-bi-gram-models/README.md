# Bigram Language Models

## 📘 What is a Bigram?

A **bigram** is a pair of consecutive words in a sequence of text. In natural language processing (NLP), a bigram language model predicts the **next word based on the current word**. It’s one of the simplest **n-gram models** (where *n=2*).

For example, in the sentence:

> the sky is blue

The bigrams are:
- (`the`, `sky`)
- (`sky`, `is`)
- (`is`, `blue`)

A **bigram model** uses these pairs to learn how likely certain words follow others. It’s a foundational method in building statistical language models.

---

## 🧠 Hand Calculation Example

Let's say the input text contains this excerpt:

`The air was thick with danger and dread. Danger lingered like a shadow.`

#### Step 1: Clean and tokenize

Words after cleaning (lowercased and punctuation removed):
```
['the', 'air', 'was', 'thick', 'with', 'danger', 'and', 'dread', 'danger', 'lingered', 'like', 'a', 'shadow']
```


#### Step 2: Generate Bigrams

From the cleaned list, the bigrams are:

| Bigram (Key → Value)     |
|--------------------------|
| the → air                |
| air → was                |
| was → thick              |
| thick → with             |
| with → danger            |
| danger → and             |
| and → dread              |
| dread → danger           |
| danger → lingered        |
| lingered → like          |
| like → a                 |
| a → shadow               |

> Note: The word "danger" appears **twice** as a key. Its successors are both `and` and `lingered`.

#### Step 3: Final `successor_map`

```python
{
  'the': ['air'],
  'air': ['was'],
  'was': ['thick'],
  'thick': ['with'],
  'with': ['danger'],
  'danger': ['and', 'lingered'],
  'and': ['dread'],
  'dread': ['danger'],
  'lingered': ['like'],
  'like': ['a'],
  'a': ['shadow']
}
```

---

## 🐍 Python Code

```python
import random

PUNCTUATION = '.;,-"\'“”‘’:?!—()_'

# Open the input file
with open('data/sample_data.txt', 'r') as reader:
    successor_map = {}
    window = []

    # Process each line
    for line in reader:
        for word in line.split():
            clean_word = word.strip(PUNCTUATION).lower()
            if clean_word == '':
                continue
            window.append(clean_word)

            # Build bigram mapping
            if len(window) == 2:
                key = window[0]
                value = window[1]
                if key in successor_map:
                    successor_map[key].append(value)
                else:
                    successor_map[key] = [value]
                window.pop(0)

# Test: print successors of a known word
if 'danger' in successor_map:
    print("Successors of 'danger':", successor_map['danger'])
else:
    print("The word 'danger' is not in the text.")

# Generate a sequence using the bigram model
random.seed(42)  # For reproducibility
word = 'the'
print("\nGenerated sequence:")
print(word, end=' ')
for _ in range(15):
    successors = successor_map.get(word)
    if not successors:
        break
    word = random.choice(successors)
    print(word, end=' ')
print()
```

#### 🗣️ Sample Generated Output

Starting with the word "the":
```
the air was thick with danger and dread danger lingered like a shadow
```
This is one possible generated sentence using the bigram model and random selection.

---

## 📌 Notes
- The model is non-deterministic due to random choice (use **random.seed()** for reproducibility).
- Bigrams capture local word relationships; they don’t consider longer context.
- Extendable to tri-grams (3-word sequences) or higher-order models (n-grams) for better context modeling.

---