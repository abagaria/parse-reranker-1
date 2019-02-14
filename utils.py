# Python imports.

# ------------------------------------
# Data pre-processing functions
# ------------------------------------


def get_num_lines(input_file):
    with open(input_file, "r") as _file:
        num_lines = 0
        for line in _file:
            num_lines += 1
        return num_lines


def create_data_splits(input_file):
    num_lines = get_num_lines(input_file)
    num_training_lines = int(0.9 * num_lines)

    training_sentences, validation_lines = [], []
    print("Creating training and validation splits from {}".format(input_file))
    with open(input_file, "r") as _file:
        for i, line in enumerate(_file):
            if i < num_training_lines:
                training_sentences.append(line)
            else:
                validation_lines.append(line)
        print("Loaded {} training sentences and {} validation sentences".format(len(training_sentences),
                                                                                len(validation_lines)))
        return training_sentences, validation_lines
