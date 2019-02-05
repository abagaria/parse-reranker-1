import sys

input_file_name = sys.argv[1]
train_data = []
with open(input_file_name) as file:
    for line in file:
        train_data.append(line)

# TODO: Make your language model!
# -------------------------
print("I am a language model.")
