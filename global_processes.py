import random

with open('nn1.txt', 'r') as f:
    text = f.read()
records = text.splitlines()
split_data = [record.split("   ") for record in records]

#split the data to train and test groups
train_idx = random.sample(range(20000), k=14000)
train = [split_data[i] for i in train_idx]

remaining_records = [split_data[i] for i in range(len(split_data)) if i not in train_idx]

validation_idx = random.sample(range(6000), k=5000)
validation = [remaining_records[i] for i in validation_idx]
test = [remaining_records[i] for i in range(len(remaining_records)) if i not in validation_idx]

# save the groups in global variables
TRAIN_STR = [string[0] for string in train]
ACTUAL_CLASS = [float(string[1]) for string in train]
TEST_STR = [string[0] for string in validation]
TEST_CLASS = [float(string[1]) for string in validation]

with open('test.txt', 'w') as f_test:
    for record in test:
        f_test.write(f"{record[0]}\t{record[1]}\n")

NUM_WEIGHT = 217


