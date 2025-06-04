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
