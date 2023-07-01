import random
with open('train_nn1.txt', 'r') as f:
    text = f.read()
records = text.splitlines()
split_data = [record.split("   ") for record in records]

train_idx = random.sample(range(19000), k=14000)
train = [split_data[i] for i in train_idx]

validation = [split_data[i] for i in range(len(split_data)) if i not in train_idx]

with open('test_nn1.txt', 'r') as f:
    text = f.read()
records = text.splitlines()
test = [record.split("   ") for record in records]


# save the groups in global variables
TRAIN_STR = [string[0] for string in train]
ACTUAL_CLASS = [float(string[1]) for string in train]
TEST_STR = [string[0] for string in validation]
TEST_CLASS = [float(string[1]) for string in validation]

NUM_WEIGHT = 217


