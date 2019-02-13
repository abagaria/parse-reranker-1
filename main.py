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
from utils import create_data_splits
from hyperparameters import *


def train(input_file):
    modes = ["train", "dev"]

    # We create the vocabulary over the entire input file and later perform the training vs validation splits.
    vocab_object = Vocab(input_file, padding=False)
    vocab_size = vocab_object.get_vocab_size()
    assert vocab_object.PAD_TOKEN == 0, "Padding logits with 0s"

    # 90% of the sentences should be in the training_split and the remaining in the validation split.
    training_split, validation_split = create_data_splits(input_file)
    data = {"train": training_split, "dev": validation_split}
    params = vocab_object.vocab, vocab_object.reverse_vocab, vocab_object.PAD_TOKEN, vocab_object.START_TOKEN, vocab_object.get_unk_token()
    datasets = {mode: ParserDataset(data[mode], *params) for mode in modes}
    loaders = {mode: DataLoader(datasets[mode], batch_size=BATCH_SIZE, shuffle=True, drop_last=True) for mode in modes}
    network = NeuralParser(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, RNN_LAYERS, BATCH_SIZE, device)

    loss_function = nn.CrossEntropyLoss(ignore_index=vocab_object.PAD_TOKEN).to(device)
    optimizer = optim.Adam(network.parameters())
    # hidden_state = network.init_hidden(BATCH_SIZE)

    training_loss, validation_loss = [], []
    n_training_iterations = 0
    n_validation_iterations = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        for mode in modes:
            for input_sequence, label_sequence, seq_length in tqdm(loaders[mode], desc='{}:{}/{}'.format(mode, epoch,
                                                                                                         NUM_EPOCHS)):
                input_sequence = input_sequence.to(device)
                label_sequence = label_sequence.to(device)

                if mode == "train":
                    network.train()
                    # hidden_state = hidden_state.detach()
                    network.zero_grad()

                    logits = network(input_sequence, seq_length)

                    padded_logits = torch.zeros(BATCH_SIZE, label_sequence.shape[1], vocab_size, device=device)
                    padded_logits[:, :logits.shape[1], :] = logits

                    loss = loss_function(padded_logits.view(-1, vocab_size), label_sequence.view(-1))
                    perplexity = np.exp(loss.item())

                    loss.backward()
                    optimizer.step()

                    n_training_iterations = n_training_iterations + 1
                    training_loss.append(loss.item())

                    # Logging
                    writer.add_scalar("TrainingLoss", loss.item(), n_training_iterations)
                    writer.add_scalar("TrainingPerplexity", perplexity, n_training_iterations)

                else:
                    network.eval()
                    with torch.no_grad():
                        logits = network(input_sequence, seq_length)
                        padded_logits = torch.zeros(BATCH_SIZE, label_sequence.shape[1], vocab_size, device=device)
                        padded_logits[:, :logits.shape[1], :] = logits

                        loss = loss_function(padded_logits.view(-1, vocab_size), label_sequence.view(-1))
                        perplexity = np.exp(loss.item())
                    n_validation_iterations = n_validation_iterations + 1
                    validation_loss.append(loss.item())

                    # Logging
                    writer.add_scalar("ValidationLoss", loss.item(), n_validation_iterations)
                    writer.add_scalar("ValidationPerplexity", perplexity, n_validation_iterations)
        torch.save(network.state_dict(), "{}_weights.pt".format(epoch))

    return training_loss, validation_loss


def main():
    training_loss_history, validation_loss_history = train(input_file_name)
    return training_loss_history, validation_loss_history


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    t_loss, v_loss = main()

    print()
    print("=" * 80)
    print("Training perplexity = {:.2f}".format(np.exp(np.mean(t_loss))))
    print("Validation perplexity = {:.2f}".format(np.exp(np.mean(v_loss))))
    print("=" * 80)

