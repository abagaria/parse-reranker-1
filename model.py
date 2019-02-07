# Python imports.
import sys
import pdb

# PyTorch imports.
import torch
import torch.nn as nn


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
        self.rnn_encoder = nn.GRU(self.embedding_size, self.hidden_size, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(device)

    def forward(self, tokens, hidden):
        embeds = self.embedding(tokens)
        output, new_hidden = self.rnn_encoder(embeds, hidden)
        logits = self.fc(output)
        return logits, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=self.device)
