import numpy as np

from typing import List, Tuple


class OneHotEncoder:

    def __init__(self, chars):
        self._chars = sorted(set(chars))
        self._char_to_index = {}
        self._index_to_char = {}
        self._fill_translation_dicts()

    def _fill_translation_dicts(self):
        for idx, char in enumerate(self._chars):
            self._char_to_index[char] = idx
            self._index_to_char[idx] = char

    def get_index_of_char(self, char):
        return self._char_to_index[char]

    def get_charset_size(self) -> int:
        return len(self._chars)
    
    def encode_one_hot(self, word: str, max_length: int) -> List:
        one_hot = np.zeros((max_length, len(self._chars)), dtype=np.float32)
        for idx, char in enumerate(word):
            one_hot[idx, self._char_to_index[char]] = 1.0
        return one_hot

    def decode_one_hot(self, one_hot_vector: np.array) -> Tuple:
        indices = one_hot_vector.argmax(axis=-1)
        chars = ''.join(self._index_to_char[ind] for ind in indices)
        return indices, chars
