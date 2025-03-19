import numpy as np


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def is_valid_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def is_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

class DataAggregator:
    def __init__(self, word_list):
        self.word_list = word_list
        self.max_word_size = len(max(self.word_list, key=len))

        self.shuffled_word_list = self.generate_shuffled_words()
        self.trie = self.build_trie()
        self.words_and_targets, self.shuffled_words_and_targets = self.generate_dataset()
        self.word_padded_set = self.add_padding(self.words_and_targets)
        self.shuffled_word_padded_set = self.add_padding(self.shuffled_words_and_targets)

        self.vocab ="abcdefghijklmnopqrstuvwxyz"
        self.vocab_size = len(self.vocab)

        self.one_hot_word_set = self.convert_dataset_to_one_hot(self.word_padded_set)
        self.one_hot_shuffled_word_set = self.convert_dataset_to_one_hot(self.shuffled_word_padded_set)
        self.blank_space = [(["0"] * self.max_word_size, 0)]

        self.all_word_data = [self.one_hot_word_set, self.one_hot_shuffled_word_set, self.blank_space]

    def get_data(self):
        return self.all_word_data

    def generate_dataset(self):
        word_and_targets = []
        shuffled_word_and_targets = []
        for i in range(len(self.word_list)):
            word_targets = self.classify_word_positions(self.word_list[i])
            shuffled_word_targets = self.classify_word_positions(self.shuffled_word_list[i])

            word_and_targets.append(list(zip(list(self.word_list[i]), word_targets)))
            shuffled_word_and_targets.append(list(zip(list(self.shuffled_word_list[i]), shuffled_word_targets)))

        return word_and_targets, shuffled_word_and_targets

    def classify_word_positions(self, word):
        targets = [0] * len(word)

        for i in range(1, len(word) + 1):
            prefix = word[:i]
            if self.trie.is_word(prefix):
                targets[i-1] = 2
            elif self.trie.is_valid_prefix(prefix):
                targets[i-1] = 1
            else:
                targets[i-1] = 3
        return targets

    def build_trie(self):
        trie = Trie()
        for word in self.word_list:
            trie.insert(word)
        return trie

    def add_padding(self, word_set, pad_char='0', pad_target=0):
        """
        Add padding to a dataset where each item is a list of (character, target) pairs
        """
        processed_dataset = []

        for word_target_pairs in word_set:
            # Separate characters and targets
            word_with_separator = list(word_target_pairs) + [(pad_char, pad_target)]
            processed_dataset.append(word_with_separator)

        return processed_dataset

    def generate_shuffled_words(self):
        shuffled_words = self.word_list.copy()

        for idx, word in enumerate(shuffled_words):
            shuffled_word = np.array(list(word))
            np.random.shuffle(shuffled_word)
            shuffled_word = "".join(shuffled_word)
            shuffled_words[idx] = shuffled_word
        return shuffled_words

    def convert_to_one_hot(self, word_data):
        char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        vocab_size = len(self.vocab)

        one_hot_data = []

        for char, target in word_data:
            if char == '0':
                one_hot = np.zeros(vocab_size)
            else:
                one_hot = np.zeros(vocab_size)
                char_idx = char_to_idx.get(char.lower(), 0)
                one_hot[char_idx] = 1
            one_hot_data.append((one_hot, target))
        return one_hot_data

    def convert_dataset_to_one_hot(self, char_data):
        one_hot_data = []
        for i in range(len(char_data)):
            one_hot_data.append(self.convert_to_one_hot(char_data[i]))
        return one_hot_data
