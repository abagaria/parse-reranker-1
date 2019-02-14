# Python imports.
import pdb
import pickle
from collections import defaultdict


class Vocab(object):

    def __init__(self, input_file):
        self.input_file = input_file
        self.training_data = self._get_training_data()
        self.vocab, self.reverse_vocab = self._create_vocab()
        self.save_vocab()

    def _get_training_data(self):
        train_data = []
        with open(self.input_file) as _file:
            for line in _file:
                train_data.append(line)
        return train_data

    def get_vocab_size(self):
        return len(self.vocab.keys())

    def _create_vocab(self):
        all_words = []
        for line in self.training_data:
            words = line.split()
            all_words += words
        word_set = set(all_words)
        vocab = defaultdict()
        reverse_vocab = defaultdict()
        for idx, word in enumerate(word_set):
            vocab[word] = idx
            reverse_vocab[idx] = word
        return vocab, reverse_vocab

    def get_unk_token(self):
        return self.vocab["*UNK"]

    def get_pad_token(self):
        return self.vocab["STOP"]

    def save_vocab(self):
        with open("vocab.pkl", "wb") as vf:
            pickle.dump(self.vocab, vf)
        with open("reverse_vocab.pkl", "wb") as ivf:
            pickle.dump(self.reverse_vocab, ivf)


if __name__ == "__main__":
    f = "data/reranker_train.txt"
    v = Vocab(f)
