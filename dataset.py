# Python imports.
import pdb
from tqdm import tqdm
from collections import defaultdict

# PyTorch imports.
import torch
from torch.utils.data import Dataset

# Other imports.
from vocab import Vocab


class ParserDataset(Dataset):
    """ Dataset module for our parse re-ranker. """

    def __init__(self, raw_sentences, vocab, reverse_vocab, pad_token, start_token, unk_token):
        """
        Args:
            raw_sentences (list): list of sentences in training / dev data
            vocab (defaultdict)
            reverse_vocab (defaultdict)
            pad_token (int)
            start_token (int)
            unk_token (int)
        """
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.pad_token = pad_token
        self.start_token = start_token
        self.unk_token = unk_token

        self.sentences_tensor = self.read_sentences(raw_sentences)
        self.input_sentences = self.get_padded_input_sentences(self.sentences_tensor)
        self.label_sentences = self.get_padded_label_sentences(self.sentences_tensor)

    def read_sentences(self, sentences):
        sentence_vectors = []
        for sentence in tqdm(sentences):
            if sentence != "STOP\n":
                sentence_vectors.append(self.read_sentence(sentence))
        return sentence_vectors
        # return [self.read_sentence(sentence) for sentence in tqdm(sentences)]

    def read_sentence(self, sentence):
        sentence_list = sentence.split()
        word_ids = [self.vocab[word] if word in self.vocab else self.unk_token for word in sentence_list]
        return torch.tensor(word_ids, dtype=torch.long)

    def get_padded_input_sentences(self, sentences):
        x_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(x_lengths)
        batch_size = len(sentences)

        # +1 to the num characters per sentence to account for the <START> token
        padded_sentences = torch.ones(batch_size, longest_sentence + 1, dtype=torch.long) * self.pad_token

        # copy over the actual sequences
        print("Padding input sentences..")
        for i, x_len in tqdm(enumerate(x_lengths)):
            sequence = sentences[i]
            padded_sentences[i, 0] = self.start_token

            # We do not include the last character in a seq/sentence as input
            padded_sentences[i, 1:x_len] = sequence[:x_len - 1]
        return padded_sentences

    def get_padded_label_sentences(self, sentences):
        x_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(x_lengths)
        batch_size = len(sentences)

        # -1 to account for the fact that the label will not contain the 1st char of a seq
        padded_sentences = torch.ones(batch_size, longest_sentence, dtype=torch.long) * self.pad_token

        # copy over the actual sequences
        print("Padding output sentences..")
        for i, x_len in tqdm(enumerate(x_lengths)):
            sequence = sentences[i]

            # We do not include the first character in a seq/sentence as label
            padded_sentences[i, 0] = self.pad_token
            padded_sentences[i, 1:x_len] = sequence[1:x_len]
        return padded_sentences

    def __len__(self):
        return len(self.sentences_tensor)

    def __getitem__(self, i):
        sequence_length = len(self.sentences_tensor[i])
        input_tensor = self.input_sentences[i]
        label_tensor = self.label_sentences[i]

        return input_tensor, label_tensor, sequence_length


if __name__ == "__main__":
    input_file = "data/reranker_toy.txt"
    vocab_obj = Vocab(input_file, padding=True)
    dset = ParserDataset(vocab_obj.training_data, vocab_obj.vocab, vocab_obj.reverse_vocab, vocab_obj.PAD_TOKEN,
                         vocab_obj.START_TOKEN, vocab_obj.get_unk_token())

