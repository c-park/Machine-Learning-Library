#!/usr/bin/env python

"""
Testing function for perceptron algorithms in Machine Learning

Cade Parkison
"""
from perceptron import *


def test():
    # Import and prepare data

    test_data = pd.read_csv('bank-note/test.csv', header=None)
    train_data = pd.read_csv('bank-note/train.csv', header=None)
    columns = ['var', 'skew', 'curt', 'ent', 'label']
    features = columns[:-1]
    output = columns[-1]
    test_data.columns = columns
    train_data.columns = columns
    train_inputs = train_data.iloc[:, :-1].values
    test_inputs = test_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values
    test_labels = test_data.iloc[:, -1].values

    # Standard Perceptron Testing
    perceptron_s = Perceptron(4)
    perceptron_s.train(train_inputs, train_labels)
    error_s = perceptron_s.evaluate(test_inputs, test_labels)
    print("Standard Perceptron Test Error: " + str(error_s))
    weights = []
    errors = []
    for i in range(100):
        perceptron = Perceptron(4)
        weights.append(perceptron.train(train_inputs, train_labels))
        errors.append(perceptron.evaluate(test_inputs, test_labels))

    print(np.mean(weights, axis=0)), print(np.mean(errors))

    # Voted Perceptron Testing
    perceptron_v = VotedPerceptron(4)
    perceptron_v.train(train_inputs, train_labels)
    error_v = perceptron_v.evaluate(test_inputs, test_labels)
    print("Voted Perceptron Test Error: " + str(error_v))
    weights = []
    errors = []
    for i in range(100):
        perceptron = VotedPerceptron(4)
        weights.append(perceptron.train(train_inputs, train_labels))
        errors.append(perceptron.evaluate(test_inputs, test_labels))
    print(np.mean(weights, axis=0)), print(np.mean(errors))

    # Average Perceptron Testing
    perceptron_a = AvgPerceptron(4)
    perceptron_a.train(train_inputs, train_labels)
    error_a = perceptron_a.evaluate(test_inputs, test_labels)
    print("Average Perceptron Test Error: " + str(error_a))
    weights = []
    errors = []
    for i in range(100):
        perceptron = AvgPerceptron(4)
        weights.append(perceptron.train(train_inputs, train_labels))
        errors.append(perceptron.evaluate(test_inputs, test_labels))
    print(np.mean(weights, axis=0)), print(np.mean(errors))


if __name__ == "__main__":
    test()
