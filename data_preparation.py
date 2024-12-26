import random
import numpy as np

def generate_scrambled_dataset(words):
    scrambled_dataset = []
    for word in words:
        scrambled = ''.join(random.sample(word, len(word)))
        scrambled_dataset.append((scrambled, word))
    return scrambled_dataset

def one_hot_encode(word, max_len, char_map):
    one_hot = np.zeros((max_len, len(char_map)), dtype=np.float32)
    for i, char in enumerate(word):
        if char in char_map:
            one_hot[i, char_map[char]] = 1.0
    return one_hot
