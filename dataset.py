# Python imports.
from collections import defaultdict
import pdb
import numpy as np
from tqdm import tqdm

# PyTorch imports.
import torch
from torch.utils.data import Dataset

PAD_TOKEN = 0
START_TOKEN = 1


def create_vocab(input_file):
    with open(input_file, "r") as _file:
        all_words = []
        for line in _file:
            words = line.split()
            all_words += words
        word_set = set(all_words)
        vocab = defaultdict()
        for idx, word in enumerate(word_set):
            # Note: we are adding 2 so as to not conflict with the special tokens
            vocab[word] = idx + 2
        return vocab


class ParserDataset(Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.vocab = create_vocab(input_file)
        self.unpadded_sentences = self.read_sentences(input_file)
        self.padded_sentences = self.pad_sentences(self.unpadded_sentences)
        self.padded_input_sentences = self.get_padded_input_sentences(self.unpadded_sentences)
        self.padded_label_sentences = self.get_padded_label_sentences(self.unpadded_sentences)

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab.keys())

    def read_sentences(self, filename):
        print("Reading sentences..")
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = [self.vocab[word] for word in line.split()]
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def pad_sentences(sentences):
        pad_token = PAD_TOKEN
        X_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(X_lengths)
        batch_size = len(sentences)
        padded_sentences = torch.ones(batch_size, longest_sentence, dtype=torch.long) * pad_token

        # copy over the actual sequences
        print("Padding input sentences..")
        for i, x_len in tqdm(enumerate(X_lengths)):
            sequence = sentences[i]
            padded_sentences[i, 0:x_len] = sequence[:x_len]
        return padded_sentences

    @staticmethod
    def get_padded_input_sentences(sentences):
        X_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(X_lengths)
        batch_size = len(sentences)
        padded_sentences = torch.ones(batch_size, longest_sentence+1, dtype=torch.long) * PAD_TOKEN

        # copy over the actual sequences
        print("Padding input sentences..")
        for i, x_len in tqdm(enumerate(X_lengths)):
            sequence = sentences[i]
            padded_sentences[i, 0] = START_TOKEN
            padded_sentences[i, 1:x_len] = sequence[:x_len-1]
        return padded_sentences

    @staticmethod
    def get_padded_label_sentences(sentences):
        X_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(X_lengths)
        batch_size = len(sentences)
        padded_sentences = torch.ones(batch_size, longest_sentence - 1, dtype=torch.long) * PAD_TOKEN

        # copy over the actual sequences
        print("Padding input sentences..")
        for i, x_len in tqdm(enumerate(X_lengths)):
            sequence = sentences[i]
            padded_sentences[i, 0:x_len-1] = sequence[1:x_len]
        return padded_sentences

    def __len__(self):
        return len(self.unpadded_sentences)

    def __getitem__(self, i):
        # pdb.set_trace()
        # sentence = self.padded_input_sentences[i]
        # label_sentence = self.padded_label_sentences[i]
        sentence = self.padded_sentences[i]
        seq_length = torch.tensor([len(self.unpadded_sentences[i])], dtype=torch.long)
        return sentence, seq_length
