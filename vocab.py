# Python imports.
import pdb
from collections import defaultdict


class Vocab(object):

    PAD_TOKEN = 0
    START_TOKEN = 1

    def __init__(self, input_file, padding=False):
        self.input_file = input_file
        self.padding = padding
        self.training_data = self._get_training_data()

        self.num_special_tokens = 2 if padding else 0
        self.vocab, self.reverse_vocab = self._create_vocab(self.num_special_tokens)

    def _get_training_data(self):
        train_data = []
        with open(self.input_file) as _file:
            for line in _file:
                train_data.append(line)
        return train_data

    def get_vocab_size(self):
        return len(self.vocab.keys())

    def _create_vocab(self, num_special_tokens=0):
        all_words = []
        for line in self.training_data:
            words = line.split()
            all_words += words
        word_set = set(all_words)
        vocab = defaultdict()
        reverse_vocab = defaultdict()
        for idx, word in enumerate(word_set):
            # Note: we are adding 2 so as to not conflict with the special tokens
            vocab[word] = idx + num_special_tokens
            reverse_vocab[idx + num_special_tokens] = word
        return vocab, reverse_vocab
