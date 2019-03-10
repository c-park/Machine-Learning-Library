"""
id3 Algorithm for learning Decision tree.

Author: Cade Parkison
University of Utah
Machine Learning

"""
import numpy as np
import pandas as pd
import matplotlib as mpl


class DecisionTree(object):
    """Decision Tree

    Args:
        object ([type]): [description]

    Raises:
        NotImplementedError: [description]
    """

    def __init__(self, X, y, max_depth=10):
        self.root = None
        self.tree = None
        self.X = X
        self.y = y

    def entropy(self, labels):
        vals, freqs = np.unique(labels, return_counts=True)
        probs = freqs / len(labels)
        entropy = - probs.dot(np.log2(probs))

        return entropy

    def maj_error(self, abels):
        vals, freqs = np.unique(labels, return_counts=True)
        probs = freqs / len(labels)
        me = 1 - probs.max()

        return me

    def gini(self, labels):
        vals, freqs = np.unique(labels, return_counts=True)
        probs = freqs / len(labels)
        gi = 1 - probs.dot(probs)

        return gi

    def info_gain(self, data, split_attr, target_attr, gain_method):

        #print('method: {}'.format(gain_method))
        #print('type: {}'.format(type(gain_method)))
        total_e = gain_method(data[target_attr])

        # Calculate the values and counts for the split feature
        vals, freqs = np.unique(data[split_attr], return_counts=True)

        # Calculate new entropy for split
        new_e = 0
        for i in range(len(vals)):
            split_data = data.where(data[split_attr] == vals[i]).dropna()[
                target_attr]
            new_e += (freqs[i]/np.sum(freqs))*gain_method(split_data)

        # Calculate info gain
        info_gain = total_e - new_e
        return info_gain

    def train(self, data, original_data, attrs, target_attr, gain_method, parent_label, current_depth=0, max_depth=6):
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

        # if all target labels are the same, stop and return this value
        unique_labels = np.unique(data[target_attr])
        if len(unique_labels) == 1:
            #
            return unique_labels[0]

        # if the data is empty, return the label that occurs the most in the origional data
        elif len(data) == 0:
            vals, freqs = np.unique(
                original_data[target_attr], return_counts=True)
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

            # Find best attribute to split data on
            info_gains = [self.info_gain(data, attr, target_attr, gain_method)
                          for attr in attrs]
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
                new_tree = self.train(new_data, original_data, attrs, target_attr,
                                      gain_method, parent_label, current_depth, max_depth)

                # Add subtree to parents tree
                tree[best_attr][val] = new_tree

            return tree

    def predict(self, ex, tree, label):
        """
        Returns True if actual label matches label from trained descision tree
        """
        for key, val in tree.items():
            attr_value = ex[key]
            new_val = val[attr_value]

            if isinstance(new_val, dict):
                # current node is not and endnode, keep recursion going
                return self.predict_label(ex, new_val, label)
            else:
                return ex[label] == new_val

    def evaluate(self, data, tree, label):
        """
            Loops over each data example to caclulate accuracy of learned tree.
        """
        N = data.shape[0]

        correct_counter = 0

        for i in range(N):
            # print(i)
            correct_counter += self.predict_label(data.iloc[i], tree, label)

        return 1 - (correct_counter / float(N))


def num_2_binary(data):
    for attr in attrs:
        vals = data[attr]
        if np.unique(vals).dtype == 'int64':
            data[attr] = data[attr] >= data[attr].median()
