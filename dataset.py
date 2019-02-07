# Python imports.
import pdb

# PyTorch imports.
import torch
from torch.utils.data import Dataset

# Other imports.
from vocab import Vocab
from hyperparameters import WINDOW_LENGTH


class ParserDataset(Dataset):
    """ Dataset module for our parse re-ranker. """

    def __init__(self, vocab_object):
        """
        Args:
            vocab_object (Vocab)
        """
        self.vocab = vocab_object.vocab
        self.reverse_vocab = vocab_object.reverse_vocab
        self.sentences = vocab_object.training_data
        self.training_data = self._create_string_windows()

    def _create_string_windows(self):
        """
        Convert the training data (list of sentences) into 1 string and then create
        chunks / windows of certain length.
        Returns:
            windows (list): list of lists of symbols in the current data window
        """
        all_words = " ".join(self.sentences).split()
        num_windows = len(all_words) // WINDOW_LENGTH
        windows = []
        for i in range(num_windows):
            start_idx = WINDOW_LENGTH * i
            end_idx = start_idx + WINDOW_LENGTH
            window = all_words[start_idx:end_idx]
            windows.append(window)
        return windows

    def __len__(self):
        assert isinstance(self.training_data[0], list), "Expected data as LoL, got {}".format(self.training_data)
        return len(self.training_data)

    def __getitem__(self, i):
        sub_string = self.training_data[i]

        # Input doesn't include the last element and label doesn't include the first element.
        input_sub_string = sub_string[:-1]
        label_sub_string = sub_string[1:]

        # Convert strings to list of word ids.
        input_sequence = [self.vocab[word] for word in input_sub_string]
        label_sequence = [self.vocab[word] for word in label_sub_string]

        # Eventually we want to return tensors from the dataset.
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        label_tensor = torch.tensor(label_sequence, dtype=torch.long)

        return input_tensor, label_tensor
