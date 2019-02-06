# Python imports.
import sys
import pdb
from tqdm import tqdm

# PyTorch imports.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence

# Other imports.
from dataset import ParserDataset
from parse_reranker_1 import NeuralParser
from hyperparameters import *


def train(input_file):
    device = torch.device("cpu")
    dset = ParserDataset(input_file)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vocab_size = dset.get_vocab_size()
    model = NeuralParser(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, RNN_LAYERS)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    training_loss, validation_loss = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        for vectorized_seq, seq_len in tqdm(loader, desc='{}/{}'.format(epoch, NUM_EPOCHS)):
            vectorized_seq = vectorized_seq.to(device)

            start_tokens = torch.ones(vectorized_seq.shape[0], 1, dtype=torch.long, device=device) * START_TOKEN
            input_sequence = torch.cat((start_tokens, vectorized_seq), 1)


            pdb.set_trace()

            model.train()
            model.zero_grad()
            logits = model(vectorized_seq, seq_len)
            loss = loss_function(logits, label_seq)

            loss.backward()
            optimizer.step()


def evaluate():
    pass


def main():
    train(input_file_name)
    evaluate()


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    main()
