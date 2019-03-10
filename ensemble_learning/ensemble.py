#!/usr/bin/env python
from decision_tree import *
import decision_tree_ada as ada

from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test_data = pd.read_csv('bank/test.csv', header=None)
train_data = pd.read_csv('bank/train.csv', header=None)

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
           'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_data.columns = columns
test_data.columns = columns
num_2_binary(train_data)
num_2_binary(test_data)


def ada_boost(train_data, test_data, max_iter=5):

    # keep track of alphas and hypotheses
    alphas = []
    trees = []

    attrs = list(train_data.columns.values)[:-1]
    label = list(train_data.columns.values)[-1]

    M_train, N_train = train_data.shape
    M_test, N_test = test_data.shape

    # initialize weights, D_1
    weights = np.ones(M_train)/M_train

    # initialize hypotheses, one each for training and testing data
    h_train = np.zeros(M_train)
    h_test = np.zeros(M_test)

    for t in range(max_iter):
        print('')
        print('iter: {}'.format(t))
        # Train weak learner using training data weighted according to D_t
        tree = ada.id3(train_data, train_data, attrs, label, gain_method=ada.entropy,
                       parent_label=None, weights=weights, current_depth=0, max_depth=2)
        # get weak hypothesis
        h = weak_hypothesis(tree, train_data)
        h_t = np.array(list(map(lambda x: 1 if x else -1, h)))

        # convert "yes" and "no" to 1 and -1 respectively in 'y' column
        y = np.array(
            list(map(lambda x: 1 if x == 'yes' else -1, train_data['y'].tolist())))

        # calculate error w.r.t. D_t
        error_t = np.sum((h_t != y) * weights)
        print('error check: {}'.format(error_t))

        if error_t > 0.5:
            invert = -1
            error_t = 1 - error_t
            #weights = - weights
            #tree = id3(train_data, train_data, attrs, label, gain_method=entropy, parent_label=None, weights=weights, current_depth=0, max_depth=2)
            #h = weak_hypothesis(tree, train_data)
            #h_t = np.array(list(map(lambda x: 1 if x else -1, h)))
            #error_t = np.sum((h_t != y) * weights)
        else:
            invert = 1

        # get alpha
        alpha_t = 0.5*invert*np.log((1-error_t)/float(error_t))

        print('error {}: {}'.format(t, error_t))
        print('alpha {}: {}'.format(t, alpha_t))

        alphas.append(alpha_t)
        trees.append(tree)

        # update weights
        weights = weights*np.exp(-alpha_t*y*h_t)
        weights = weights / np.sum(weights)

        #print('weights {}: {}'.format(t,weights))
        #print('weight avg: {}'.format(np.mean(weights)))

    # final classifier is summation of alphas and hypotheses
    #H = np.sign(np.sum(np.array(alphas)*np.array(hypos), axis=1))

    return alphas, trees


def weak_hypothesis(tree, data):
    N = data.shape[0]
    h = np.zeros(N)

    for i in range(N):
        h[i] = predict_label(data.iloc[i], tree, 'y')

    return h


def evaluate_ada(alphas, trees, train_data, test_data):

    train_error = []
    test_error = []

    n_train = train_data.shape[0]
    n_test = test_data.shape[0]

    max_iter = len(alphas)

    for t in range(max_iter):
        train_sum = np.zeros(n_train)
        test_sum = np.zeros(n_test)

        for i in range(t):
            a_t = alphas[i]
            tree = trees[i]
            h_train = np.array(
                list(map(lambda x: 1 if x else -1, weak_hypothesis(tree, train_data))))
            h_test = np.array(
                list(map(lambda x: 1 if x else -1, weak_hypothesis(tree, test_data))))
            train_sum += (a_t*h_train)
            test_sum += (a_t*h_test)

        # print('t={}'.format(t))
        #print('train_sum: {}'.format(train_sum))
        #print('train_sum avg: {}'.format(np.mean(train_sum)))
        #print('test_sum: {}'.format(test_sum))

        train_final = np.sign(train_sum)
        test_final = np.sign(test_sum)

        #print('train_final: {}'.format(train_final))
        #print('train_final avg: {}'.format(np.mean(train_final)))
        #print('test_final avg: {}'.format(np.mean(test_final)))

        y_train = np.array(
            list(map(lambda x: 1 if x == 'yes' else -1, train_data['y'].tolist())))
        y_test = np.array(
            list(map(lambda x: 1 if x == 'yes' else -1, test_data['y'].tolist())))

        train_error.append(np.sum(train_final != y_train)/n_train)
        test_error.append(np.sum(test_final != y_test)/n_test)

    return train_error, test_error


def bagging(train_data, test_data, n_samples=1000, max_iter=10):

    M_train, N_train = train_data.shape
    M_test, N_test = test_data.shape

    attrs = list(train_data.columns.values)[:-1]
    label = list(train_data.columns.values)[-1]

    trees = []
    predictions = []

    for t in range(max_iter):
        # print('')
        #print('iter: ' + str(t))
        sample = train_data.sample(n_samples, replace=True)

        tree = id3(sample, sample, attrs, label, gain_method=entropy,
                   parent_label=None, current_depth=0, max_depth=100)
        trees.append(tree)
        # pprint(tree)

        h_t = np.array(
            list(map(lambda x: 1 if x else -1, weak_hypothesis(tree, train_data))))
        predictions.append(h_t)

    # take average of predictions
    H = np.median(predictions, axis=0)

    return H


def eval_bag(H, data):
    n = train_data.shape[0]
    y = np.array(
        list(map(lambda x: 1 if x == 'yes' else -1, data['y'].tolist())))

    return np.sum(H != y)/float(n)


def weak_hypothesis(tree, data):
    N = data.shape[0]
    h = np.zeros(N)

    for i in range(N):
        h[i] = predict_label(data.iloc[i], tree, 'y')

    return h


def random_forest(train_data, test_data, max_iter=10):

    M_train, N_train = train_data.shape
    M_test, N_test = test_data.shape

    attrs = list(train_data.columns.values)[:-1]
    label = list(train_data.columns.values)[-1]

    trees = []
    predictions = []

    for t in tqdm(range(max_iter)):
        # print('')
        #print('iter: ' + str(t))
        sample = train_data.sample(M_train, replace=True)

        tree = rand_tree_learn(sample, sample, attrs,
                               label, gain_method=entropy, parent_label=None)

        # tree = id3(sample, sample, attrs, label, gain_method=entropy,
        #           parent_label=None, current_depth=0, max_depth=100)
        # trees.append(tree)

        h_t = np.array(
            list(map(lambda x: 1 if x else -1, weak_hypothesis(tree, train_data))))
        predictions.append(h_t)

    # take average of predictions
    H = np.median(predictions, axis=0)

    return H


def rand_tree_learn(data, original_data, attrs, target_attr, gain_method, parent_label, current_depth=0, max_depth=100):
    """ ID3 Algorithm

    Args:
        data (pandas dataframe): input data
        original_data (pandas dataframe): copy of original, untouched data 
        attrs (list): list of strings of attributes, all but the target attribute
        target_attr (str): name of attribute to be used at the target labels
        gain_method (function name): Information Gain method, either entropy, maj_error, or gini
        parent_label (int): attribute label of parent node in recursive algorithm.
        current_depth (int): current tree depth
        max_depth (int): maximum tree depth

    Returns:
        tree (dict): dictionary structure represented the decision tree

    """

    M, N = data.shape

    # if all target labels are the same, stop and return this value
    unique_labels = np.unique(data[target_attr])
    if len(unique_labels) == 1:
        #
        return unique_labels[0]

    # if the data is empty, return the label that occurs the most in the origional data
    elif len(data) == 0:
        vals, freqs = np.unique(original_data[target_attr], return_counts=True)
        return vals[np.argmax(freqs)]

    # if there are no more attributes, return the parent label
    elif len(attrs) == 0:
        return parent_label
    else:
        current_depth += 1
        # set value for this node to the mode of the target feature values
        vals, freqs = np.unique(data[target_attr], return_counts=True)
        parent_label = vals[np.argmax(freqs)]

        # if max depth is reached, return label that occurs the most
        if current_depth == max_depth+1:
            return parent_label

        # Randomly sample subset of Features
        features = data.iloc[:, :N-1]
        # feature subset g random features, where g is square root of number of features
        G_count = int(np.floor(np.sqrt(N-1)))
        # subset of features
        subdata = features.sample(G_count, axis=1)
        subdata['y'] = data.iloc[:, -1]

        subattrs = list(subdata.columns.values)[:-1]

        # Find best attribute to split data on using Subdata
        info_gains = [info_gain(subdata, attr, target_attr, gain_method)
                      for attr in subattrs]
        best_attr = attrs[info_gains.index(max(info_gains))]

        # create new subtree
        tree = dict()
        tree[best_attr] = dict()

        # remove best attribute from attribute list
        attrs = [i for i in attrs if i != best_attr]

        # grow tree
        for val in np.unique(original_data[best_attr]):
            val = val
            new_data = dict(data)

            # split dataset on the best attribute and remove this column from dataset
            new_data = data.where(data[best_attr] == val).dropna()

            # Recursion
            new_tree = id3(new_data, original_data, attrs, target_attr,
                           gain_method, parent_label, current_depth, max_depth)

            # Add subtree to parents tree
            tree[best_attr][val] = new_tree

        return tree


def test_ada():
    alphas, trees = ada_boost(train_data, test_data, max_iter=10)

    plt.figure()
    plt.plot(range(100), alphas)
    plt.title("Alphas as a funtion of t")
    plt.show()

    train_error, test_error = evaluate_ada(
        alphas, trees, train_data, test_data)

    print(train_error)
    print(test_error)


def test_bag():
    train_errors = []
    test_errors = []

    for i in range(1000):
        H_bag = bagging(train_data, test_data, n_samples=1000, max_iter=i)
        train_errors.append(eval_bag(H_bag, train_data))
        test_errors.append(eval_bag(H_bag, test_data))

    print(train_errors)
    print(test_errors)


def test_rf():
    H = random_forest(train_data, test_data, max_iter=10)
    print(eval_bag(H, train_data))


if __name__ == "__main__":
    train_errors_bag, test_errors_bag = test_bag()

    print(train_errors)
    print(test_errors)
