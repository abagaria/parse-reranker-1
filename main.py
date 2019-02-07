# Python imports.
import sys
import pdb
from tqdm import tqdm
import numpy as np

# PyTorch imports.
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Other imports.
from vocab import Vocab
from dataset import ParserDataset
from model import NeuralParser
from hyperparameters import *


def train(input_file):
    vocab_object = Vocab(input_file, padding=False)
    vocab_size = vocab_object.get_vocab_size()
    dataset = ParserDataset(vocab_object)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    network = NeuralParser(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, RNN_LAYERS, BATCH_SIZE, device)
    network.train()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters())
    training_loss, validation_loss = [], []
    n_iterations = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        hidden_state = network.init_hidden(BATCH_SIZE)
        for input_sequence, label_sequence in tqdm(loader, desc='{}/{}'.format(epoch, NUM_EPOCHS)):
            input_sequence = input_sequence.to(device)
            label_sequence = label_sequence.to(device)

            hidden_state = hidden_state.detach()
            network.zero_grad()
            logits, hidden_state = network(input_sequence, hidden_state)

            loss = loss_function(logits.view(-1, vocab_size), label_sequence.view(-1))
            perplexity = np.exp(loss.item())

            loss.backward()
            optimizer.step()

            n_iterations = n_iterations + 1
            training_loss.append(loss.item())

            # Logging
            writer.add_scalar("TrainingLoss", loss.item(), n_iterations)
            writer.add_scalar("TrainingPerplexity", perplexity, n_iterations)

    return training_loss


def evaluate():
    pass


def main():
    training_loss_history = train(input_file_name)
    evaluate()

    return training_loss_history


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    device = torch.device("cpu")
    writer = SummaryWriter()
    t_loss = main()
