#!/usr/bin/env python
"""
Perceptron Algorithms for Machine Learning

Author: Cade Parkison
University of Utah
Machine Learning

"""

import numpy as np
import pandas as pd


class Perceptron(object):

    def __init__(self, no_of_inputs, epoch=10, rate=0.01):
        self.epoch = epoch
        self.rate = rate   # learning rate
        self.weights = np.zeros(no_of_inputs + 1)  # initialize weights to zero

    def predict(self, inputs):
        # predicts the label of one training example input with current weights
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, train_inputs, labels):
        # trains perceptron weights on training dataset
        labels = np.expand_dims(labels, axis=1)
        data = np.hstack((train_inputs, labels))
        for e in range(self.epoch):
            #print("Epoch: "+ str(e))
            #print("Weights: " + str(self.weights))
            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                prediction = self.predict(inputs)
                self.weights[1:] += self.rate * (label - prediction) * inputs
                self.weights[0] += self.rate * (label - prediction)

        return self.weights

    def evaluate(self, test_inputs, labels):
        # calculates average prediction error on testing dataset
        errors = []
        for inputs, label in zip(test_inputs, labels):
            prediction = self.predict(inputs)
            errors.append(np.abs(label-prediction))

        return sum(errors) / float(test_inputs.shape[0])


class VotedPerceptron(object):

    def __init__(self, no_of_inputs, epoch=10, rate=0.01):
        self.epoch = epoch
        self.rate = rate   # learning rate
        self.weights = np.zeros(no_of_inputs + 1)  # initialize weights to zero
        #self.weights_set = [np.zeros(no_of_inputs + 1)]
        self.C = [0]

    def predict(self, inputs, weights):
        # predicts the label of one training example input with current weights
        summation = np.dot(inputs, weights[1:]) + weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, train_inputs, labels):
        # trains perceptron weights on training dataset
        weights = np.zeros(train_inputs.shape[1] + 1)
        weights_set = [np.zeros(train_inputs.shape[1]+1)]
        labels = np.expand_dims(labels, axis=1)
        data = np.hstack((train_inputs, labels))
        m = 0
        for e in range(self.epoch):
            #print("Epoch: "+ str(e))

            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                prediction = self.predict(inputs, weights)
                error = label - prediction
                if error:
                    #weights_a = self.rate * (label - prediction) * inputs
                    #weights_b = self.rate * (label - prediction)
                    #self.weights[1:] += weights_a
                    #self.weights[0] += weights_b
                    weights[1:] += self.rate * (label - prediction) * inputs
                    weights[0] += self.rate * (label - prediction)
                    # print('Error!')
                    # print(weights)
                    weights_set.append(np.copy(weights))

                    self.C.append(1)
                    m += 1

                else:
                    self.C[m] += 1

        self.weights = weights
        self.weights_set = weights_set

        return self.weights

    def evaluate(self, test_inputs, labels):
        # calculates average prediction error on testing dataset
        errors = []
        n_weights = len(self.weights_set)
        for inputs, label in zip(test_inputs, labels):
            predictions = []
            for k in range(n_weights):
                pred = self.predict(inputs, weights=self.weights_set[k])
                if not pred:
                    pred = -1
                predictions.append(self.C[k]*pred)

            prediction = np.sign(sum(predictions))
            if prediction == -1:
                prediction = 0

            errors.append(np.abs(label-prediction))

        return sum(errors) / float(test_inputs.shape[0])


class AvgPerceptron(object):

    def __init__(self, no_of_inputs, epoch=10, rate=0.01):
        self.epoch = epoch
        self.rate = rate   # learning rate
        self.weights = np.zeros(no_of_inputs + 1)  # initialize weights to zero
        #self.weights_set = [np.zeros(no_of_inputs + 1)]
        self.a = np.zeros(no_of_inputs + 1)

    def predict(self, inputs, weights):
        # predicts the label of one training example input with current weights
        summation = np.dot(inputs, weights[1:]) + weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, train_inputs, labels):
        # trains perceptron weights on training dataset
        weights = np.zeros(train_inputs.shape[1] + 1)
        weights_set = [np.zeros(train_inputs.shape[1]+1)]
        labels = np.expand_dims(labels, axis=1)
        data = np.hstack((train_inputs, labels))
        m = 0
        for e in range(self.epoch):
            #print("Epoch: "+ str(e))

            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                prediction = self.predict(inputs, weights)
                error = label - prediction
                weights[1:] += self.rate * (label - prediction) * inputs
                weights[0] += self.rate * (label - prediction)
                self.a += np.copy(weights)

        self.weights = weights

        return self.a

    def evaluate(self, test_inputs, labels):
        # calculates average prediction error on testing dataset
        errors = []
        for inputs, label in zip(test_inputs, labels):
            prediction = self.predict(inputs, weights=self.a)
            errors.append(np.abs(label-prediction))

        return sum(errors) / float(test_inputs.shape[0])
