# Python imports.
import pdb
from tqdm import tqdm
from collections import defaultdict
import pickle

# PyTorch imports.
import torch
from torch.utils.data import Dataset

# Other imports.
from vocab import Vocab


class ParserDataset(Dataset):
    """ Dataset module for our parse re-ranker. """

    def __init__(self, raw_sentences, vocab, reverse_vocab):
        """
        Args:
            raw_sentences (list): list of sentences in training / dev data
            vocab (defaultdict)
            reverse_vocab (defaultdict)
        """
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.pad_token = vocab["STOP"]
        self.unk_token = vocab["*UNK"]

        self.sentences_tensor = self.read_sentences(raw_sentences)
        self.input_sentences = self.get_padded_input_sentences(self.sentences_tensor)
        self.label_sentences = self.get_padded_label_sentences(self.sentences_tensor)

    def read_sentences(self, sentences):
        sentence_vectors = []
        for sentence in tqdm(sentences):
            if sentence != "STOP\n":
                sentence_vectors.append(self.read_sentence(sentence))
        return sentence_vectors

    def read_sentence(self, sentence):
        sentence_list = sentence.split()
        word_ids = [self.vocab[word] if word in self.vocab else self.unk_token for word in sentence_list]
        return torch.tensor(word_ids, dtype=torch.long)

    def get_padded_input_sentences(self, sentences):
        x_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(x_lengths)
        batch_size = len(sentences)

        padded_sentences = torch.ones(batch_size, longest_sentence - 1, dtype=torch.long) * self.pad_token

        # copy over the actual sequences
        print("Padding input sentences..")
        for i, x_len in tqdm(enumerate(x_lengths)):
            sequence = sentences[i]

            # We do not include the last character in a seq/sentence as input
            padded_sentences[i, :x_len - 1] = sequence[:x_len - 1]
        return padded_sentences

    def get_padded_label_sentences(self, sentences):
        x_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(x_lengths)
        batch_size = len(sentences)

        padded_sentences = torch.ones(batch_size, longest_sentence - 1, dtype=torch.long) * self.pad_token

        # copy over the actual sequences
        print("Padding output sentences..")
        for i, x_len in tqdm(enumerate(x_lengths)):
            sequence = sentences[i]

            # We do not include the first character in a seq/sentence as label
            padded_sentences[i, :x_len-1] = sequence[1:x_len]
        return padded_sentences

    def __len__(self):
        return len(self.sentences_tensor)

    def __getitem__(self, i):
        sequence_length = len(self.sentences_tensor[i]) - 1
        input_tensor = self.input_sentences[i]
        label_tensor = self.label_sentences[i]

        return input_tensor, label_tensor, sequence_length


def get_training_data(input_file):
    train_data = []
    with open(input_file) as _file:
        for line in _file:
            train_data.append(line)
    return train_data


if __name__ == "__main__":
    in_file = "data/reranker_train.txt"
    t_data = get_training_data(in_file)
    with open("vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open("reverse_vocab.pkl", "rb") as f:
        rv = pickle.load(f)
    d_set = ParserDataset(t_data, v, rv)
