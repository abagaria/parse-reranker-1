# Python imports.
import sys
import pdb

# PyTorch imports.
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Other imports.
from hyperparameters import DROPOUT_PROBABILITY


class NeuralParser(nn.Module):
    """ Parse re-ranker network. """
    def __init__(self, vocab_size, embedding_size, hidden_size, rnn_layers, batch_size, device=torch.device("cpu")):
        super(NeuralParser, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_layers = rnn_layers
        self.batch_size = batch_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(p=DROPOUT_PROBABILITY)
        self.rnn_encoder = nn.GRU(self.embedding_size, self.hidden_size, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(device)

    def forward(self, tokens, seq_lens):
        curr_batch_size = seq_lens.shape[0]
        embeds = self.dropout(self.embedding(tokens))
        sorted_seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        sorted_seq_tensor = embeds[perm_idx]

        packed_input = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens, batch_first=True)
        output, final_hidden = self.rnn_encoder(packed_input, self.init_hidden(curr_batch_size))
        unpacked_output, _ = pad_packed_sequence(output, batch_first=True)

        logits = self.fc(unpacked_output)

        _, unperm_idx = perm_idx.sort(0)
        unsorted_logits = logits[unperm_idx].squeeze(1)
        return unsorted_logits

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=self.device)
