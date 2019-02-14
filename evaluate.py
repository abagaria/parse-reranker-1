# Python imports.
import numpy as np
import pdb
from collections import defaultdict
import pickle

# PyTorch imports.
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Other imports.
from model import NeuralParser
from hyperparameters import *


def get_vocab():
    with open("vocab.pkl", "rb") as _f:
        v = pickle.load(_f)
    with open("reverse_vocab.pkl", "rb") as _f:
        inverse_vocab = pickle.load(_f)
    v_size = len(v)
    unk = v["UNK*"]
    print("Using vocab size {}, UNK token {}".format(v_size, unk))
    return v, inverse_vocab, v_size, unk


def read_line(line, vocab, unk_token):
    """
    Args:
        line (str): raw text string from the generated parses file
        vocab (defaultdict): Vocab to map strings to word IDs
        unk_token (int): ID for an unseen word

    Returns:
        num_correct (int): the number of correct constituent tags in this parse (compared to the gold standard)
        num_total (int): the total number of constituent tags in this parse
        parse (torch.tensor): the actual parse itself as a tensor of vocab IDs
    """
    parse_start_idx = line.find("(")
    numbers = line[:parse_start_idx].split()
    num_correct = int(numbers[0])
    num_total = int(numbers[1])
    parse_string = line[parse_start_idx:].split()
    parse_ids = [vocab[word] if word in vocab else unk_token for word in parse_string]
    parse_tensor = torch.tensor(parse_ids, dtype=torch.long)

    return num_correct, num_total, parse_tensor


def get_num_gold_tags(gold_lines, sentence_number):
    gold_parse = gold_lines[sentence_number]  # type: str
    return gold_parse.count("(")


def get_n_best(lines, vocab, unk_token):
    """
    Given the raw lines corresponding to the same sentence, output the parses as a list of tuples.
    Args:
        lines (list): list of lines corresponding to the n best parses of the same sentence
        vocab (defaultdict): map from word to word ID
        unk_token (int): *UNK*

    Returns:
        n_best (list): list of tuples containing:
                       a) the number of correct constituent tags in this parse
                       b) the total number of constituent tags in this parse
                       c) the actual parse itself
    """
    # pdb.set_trace()
    num_corrects, totals, parses = [], [], []
    for line in lines:
        n1, n2, parse = read_line(line, vocab, unk_token)
        num_corrects.append(n1)
        totals.append(n2)
        parses.append(parse)
    return num_corrects, totals, parses


def get_sequence_probability(model, sequence):
    seq = sequence.unsqueeze(0)
    seq = seq.to(device)
    with torch.no_grad():
        length = torch.tensor(seq.shape[1], device=device).unsqueeze(0)
        logits = model(seq, length)
        probabilities = F.softmax(logits, dim=2)
        selected_probabilities = probabilities.gather(2, seq.unsqueeze(2))
        sequence_probability = selected_probabilities.log().sum()
    return sequence_probability


def evaluate_n_best(model, trees, sentence_number):
    model.eval()
    probabilities = []
    for tree in trees:
        seq_probability = get_sequence_probability(model, tree)
        probabilities.append(seq_probability)
    writer.add_scalar("BestParseProbability", np.max(probabilities), sentence_number)
    return np.argmax(probabilities)


def compute_precision(num_correct_tags, total_num_tags):
    return (1. * num_correct_tags) / total_num_tags


def compute_recall(num_correct_tags, num_gold_tags):
    return (1. * num_correct_tags) / num_gold_tags


def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.
    return (2. * precision * recall) / (precision + recall)


def get_all_lines(file_name):
    with open(file_name, "r") as _file:
        return [line for line in _file]


def main():
    # Model
    model = NeuralParser(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, RNN_LAYERS, BATCH_SIZE, device)
    model.load_state_dict(torch.load(saved_model, map_location=device))

    sentence_number, end_idx, line_number = 0, 0, 0
    chosen_corrects, chosen_totals, num_golds, chosen_trees = [], [], [], []
    while line_number < len(gen_lines):
        assert len(gen_lines[line_number]) == 3, "Expected 2 digits + new line, got {}".format(gen_lines[line_number])

        num_sentence_parses = int(gen_lines[line_number])
        start_idx = end_idx + 1
        end_idx = start_idx + num_sentence_parses
        num_corrects, num_totals, n_best_trees = get_n_best(gen_lines[start_idx:end_idx], vocab, unk_token)
        best_idx = evaluate_n_best(model, n_best_trees, sentence_number)

        chosen_tree = n_best_trees[best_idx]
        chosen_num_correct = num_corrects[best_idx]
        chosen_num_total = num_totals[best_idx]
        num_gold_tags = get_num_gold_tags(gold_lines, sentence_number)

        precision = compute_precision(chosen_num_correct, chosen_num_total)
        recall = compute_recall(chosen_num_correct, num_gold_tags)
        f1_score = compute_f1_score(precision, recall)

        print("Sentence number: {} \t Num correct: {} \t Num total: {} \t Num gold: {}".format(sentence_number,
                                                                                               chosen_num_correct,
                                                                                               chosen_num_total,
                                                                                               num_gold_tags))

        writer.add_scalar("Precision", precision, sentence_number)
        writer.add_scalar("Recall", recall, sentence_number)
        writer.add_scalar("F1-Score", f1_score, sentence_number)

        line_number += num_sentence_parses + 1
        sentence_number += 1

        chosen_trees.append(chosen_tree)
        chosen_corrects.append(chosen_num_correct)
        chosen_totals.append(chosen_num_total)
        num_golds.append(num_gold_tags)

    return chosen_corrects, chosen_totals, num_golds, chosen_trees


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("Evaluation")
    gen_lines = get_all_lines("data/conv.txt")
    gold_lines = get_all_lines("data/gold.txt")
    saved_model = "1_weights.pt"

    # Vocab
    vocab, reverse_vocab, vocab_size, unk_token = get_vocab()
    cc, ct, ng, cp = main()

    p = compute_precision(np.mean(cc), np.mean(ct))
    r = compute_recall(np.mean(cc), np.mean(ng))
    f = compute_f1_score(p, r)
    print("F1-Score = {}".format(f))
