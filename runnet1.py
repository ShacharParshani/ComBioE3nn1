import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
	return max(0.0, x)

def cal_classification(strings, weights):
    class_list = []
    # build the neuron network by the weights
    hidden_layer_1 = [None for i in range(8)]
    hidden_layer_2 = [None for i in range(8)]
    for string in strings:
        index_weights = 0
        # cal hidden layer 1
        for i, n in enumerate(hidden_layer_1):
            summ = 0
            for j in range(len(string)):
                summ = summ + (float(string[j]) * float(weights[index_weights]))
                index_weights += 1
            summ = summ + (1 * float(weights[index_weights]))  # bias
            index_weights += 1
            # output = sig(summ)
            output = relu(summ)
            hidden_layer_1[i] = output

        # cal hidden layer 2
        for i, n in enumerate(hidden_layer_2):
            summ = 0
            for j in range(len(hidden_layer_1)):
                summ = summ + (hidden_layer_1[j] * float(weights[index_weights]))
                index_weights += 1
            summ = summ + (1 * float(weights[index_weights]))  # bias
            index_weights += 1
            #output = sig(summ)
            output = relu(summ)
            hidden_layer_2[i] = output

        # cal output neuron
        summ = 0
        for j in range(len(hidden_layer_2)):
            summ = summ + (hidden_layer_2[j] * float(weights[index_weights]))
            index_weights += 1
        summ = summ + (1 * float(weights[index_weights]))  # bias
        index_weights += 1
        output = sig(summ)
        # output = relu(summ)
        output_neuron = output
        if output_neuron > 0.5:
            classification = 1
        else:
            classification = 0
        class_list.append(classification)
    return class_list

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))

if __name__ == '__main__':
    with open('test.txt', 'r') as f:
        text_test = f.read()
    records_test = text_test.splitlines()
    split_data = [record.split("\t") for record in records_test]
    with open('testnet1.txt', 'w') as f:
        for record in split_data:
            f.write(f"{record[0]}\n")
    actual_class = [float(record[1]) for record in split_data]

    #until here delete in submit!!!!!

    with open('testnet1.txt', 'r') as f:
        text = f.read()
    test_strings = text.splitlines()

    with open('wnet1.txt', 'r') as f:
        weights_data = f.read()
    weights = weights_data.splitlines()

    class_list = cal_classification(test_strings, weights)

    with open('classification.txt', 'w') as f:
        for record in class_list:
            f.write(f"{record}\n")
    print(accuracy_metric(actual_class, class_list)) # delete in submit!!!!

